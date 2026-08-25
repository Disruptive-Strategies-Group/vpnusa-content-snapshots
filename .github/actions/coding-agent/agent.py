#!/usr/bin/env python3
"""DSG Coding Agent — agentic loop for the CI/CD pipeline.

Replaces anthropics/claude-code-action@v1 with a provider-agnostic agentic
coding loop. Uses any OpenAI-compatible API (DeepSeek, OpenAI, etc).

Environment variables (set by the composite action):
    AGENT_API_KEY         — API key for the LLM provider
    AGENT_API_BASE_URL    — Base URL (default: https://openrouter.ai/api/v1)
    AGENT_MODEL           — Model name (default: deepseek/deepseek-v4-flash-0731)
    AGENT_ISSUE_CONTEXT   — Path to .agent/issue-context.json
    AGENT_BRANCH_NAME     — Git branch to work on
    AGENT_ISSUE_NUMBER    — GitHub issue number
    AGENT_ISSUE_TITLE     — GitHub issue title
    AGENT_MAX_TURNS       — Max agentic loop iterations (default: 40)
    AGENT_MODE            — implement | revise (default: implement)
    AGENT_REVIEW_CONCERNS — Review concerns text (revise mode)
    AGENT_REQUIRED_FILES  — JSON-encoded array of file paths the agent must
                            restrict its changes to; "*" means no restriction
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time

from openai import OpenAI

from tools import TOOL_SCHEMAS, execute_tool

VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.environ["AGENT_API_KEY"]
API_BASE_URL = os.environ.get("AGENT_API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.environ.get("AGENT_MODEL", "deepseek/deepseek-v4-flash-0731")
ISSUE_CONTEXT_PATH = os.environ["AGENT_ISSUE_CONTEXT"]
BRANCH_NAME = os.environ["AGENT_BRANCH_NAME"]
ISSUE_NUMBER = os.environ["AGENT_ISSUE_NUMBER"]
ISSUE_TITLE = os.environ.get("AGENT_ISSUE_TITLE", "")
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "40"))
AGENT_MODE = os.environ.get("AGENT_MODE", "implement")
AGENT_REVIEW_CONCERNS = os.environ.get("AGENT_REVIEW_CONCERNS", "")
AGENT_REQUIRED_FILES = os.environ.get("AGENT_REQUIRED_FILES", "*")

# Context budget for the message window. When total serialized chars exceed
# this, _compact_messages() reclaims space (~45K tokens, approaching 64K limit).
CONTEXT_BUDGET_CHARS = 180_000


def log(msg: str) -> None:
    print(f"[agent] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Issue context loading
# ---------------------------------------------------------------------------


def load_issue_context(path: str) -> dict:
    """Load the issue context JSON produced by the workflow's 'Gather issue context' step."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_latest_plan(issue_data: dict) -> str | None:
    """Find the latest <!-- AGENT_PLAN --> comment body."""
    comments = issue_data.get("comments", [])
    plan = None
    for comment in comments:
        body = comment.get("body", "")
        if "<!-- AGENT_PLAN -->" in body:
            plan = body
    return plan


def build_issue_summary(issue_data: dict) -> str:
    """Build a concise issue summary for the system prompt.

    Cap raised 8,000 -> 32,000 chars: Aeris now files issues carrying an
    auto-generated design spec plus attachment links, and the old cap
    silently truncated exactly the spec the agent needed to implement
    faithfully. 32k chars (~8k tokens) persists in the system prompt every
    turn, which still leaves ample model context for tool outputs; going
    much higher would start squeezing the agent's working context instead.
    """
    title = issue_data.get("title", "")
    body = issue_data.get("body", "")

    if len(body) > 32000:
        body = body[:32000] + "\n\n... [truncated]"

    return f"## Issue Title\n{title}\n\n## Issue Body\n{body}"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
YOU ARE AN AUTONOMOUS CODING AGENT WORKING IN A GIT REPOSITORY.

HARD REQUIREMENTS:
- Work ONLY on the current branch: {branch_name}
- Make the minimal correct changes to satisfy the issue and the latest approved plan.
- You MUST commit your changes locally using the bash tool (git add + git commit). Do NOT push. Do NOT open a PR.
- If no code changes are required, do NOT commit.
- Do not modify .github/workflows unless the issue explicitly requires it.
- Do not modify .claude/ directory.
- Never use [skip ci], [ci skip], [no ci], or skip-checks: true in commit messages.
- NEVER use `python3 -c` or `python -c` to validate file changes. The runner shell is dash on Ubuntu and cannot handle nested parentheses in -c one-liners, which causes stuck retry loops. To verify a change landed, use `read_file` or `grep_search` instead. Once `grep_search` confirms the expected content is present, commit and stop — do not attempt further validation.

WORKFLOW:
1. Read the issue and the approved plan carefully.
2. Explore the repository structure to understand the codebase (use list_files, read_file, grep_search).
3. Implement the changes described in the plan.
4. Use edit_file for targeted changes to existing files. Use write_file for new files.
5. Test your changes if appropriate (run linters, type checks, unit tests via bash).
6. Commit all changes with a clear commit message referencing the issue number.

TRIGGERING ISSUE (SOURCE OF TRUTH):
Issue #: {issue_number}
Title: {issue_title}

{issue_summary}

{plan_section}
END TRIGGERING ISSUE
"""


REVISION_SYSTEM_PROMPT = """\
YOU ARE AN AUTONOMOUS CODING AGENT PERFORMING A TARGETED PR REVISION.

HARD REQUIREMENTS:
- Work ONLY on the current branch: {branch_name}
- You MUST commit your changes locally (git commit). Do NOT push. Do NOT open a PR.
- Do not modify .github/workflows unless the review concerns explicitly require it.
- Do not modify .claude/ directory.
- Never use [skip ci], [ci skip], [no ci], or skip-checks: true in commit messages.
- NEVER use `python3 -c` or `python -c` to validate file changes. The runner shell is dash on Ubuntu and cannot handle nested parentheses in -c one-liners, which causes stuck retry loops. To verify a change landed, use `read_file` or `grep_search` instead. Once `grep_search` confirms the expected content is present, commit and stop — do not attempt further validation.

WORKFLOW:
1. Run `git diff origin/main...HEAD` to understand what this branch has changed.
2. Run `git log --oneline -10` to see commit history.
3. For EACH file mentioned in the review concerns, read the CURRENT file content before editing.
4. Read .agent/issue-context.json if present for original issue context.
5. Address ONLY the review concerns below. Do not refactor or touch anything else.
6. After editing, verify with `git diff` that changes look correct.
7. Commit with a message listing the specific files changed and what was changed.

LARGE FILE HANDLING:
If read_file returns a truncated view of a file (you will see a TRUNCATED message), use grep_search to find the exact line numbers of the code you need to modify, then use read_file_lines to read that specific section. This is essential for files over 50,000 characters. Never attempt edit_file on code you have not directly read — the old_text will not match and the edit will fail.

REVIEW CONCERNS TO ADDRESS:
{review_concerns}

{required_files_section}
END REVIEW CONCERNS"""


def build_required_files_section() -> str:
    """Build the constrained-file-scope section for the system prompt.

    When the calling workflow passes a JSON-encoded list of file paths
    (AGENT_REQUIRED_FILES), the agent must restrict its edits to exactly
    those files so revisions stay on-target for Gate 1 feedback. The default
    value "*" (or an empty/invalid list) means no restriction.
    """
    if not AGENT_REQUIRED_FILES or AGENT_REQUIRED_FILES == "*":
        return ""
    try:
        files = json.loads(AGENT_REQUIRED_FILES)
    except json.JSONDecodeError:
        log(f"Warning: AGENT_REQUIRED_FILES is not valid JSON: {AGENT_REQUIRED_FILES}")
        return ""
    if not isinstance(files, list):
        return ""
    file_list = "\n".join(f"- {f}" for f in files if isinstance(f, str) and f)
    if not file_list:
        return ""
    return (
        "CONSTRAINED FILE SCOPE (MUST FOLLOW):\n"
        "Restrict your changes to ONLY the following files. Do not modify any other "
        "files unless the review concerns explicitly require it. If the requested "
        "change cannot be made within these files, say so and stop rather than "
        "editing unrelated files.\n"
        f"{file_list}\n"
    )


def build_system_prompt(issue_data: dict) -> str:
    if AGENT_MODE == "revise":
        return REVISION_SYSTEM_PROMPT.format(
            branch_name=BRANCH_NAME,
            review_concerns=AGENT_REVIEW_CONCERNS,
            required_files_section=build_required_files_section(),
        )

    issue_summary = build_issue_summary(issue_data)
    plan = extract_latest_plan(issue_data)
    if plan:
        plan_section = f"APPROVED PLAN (follow this exactly):\n{plan}"
    else:
        plan_section = "NO APPROVED PLAN FOUND — implement based on the issue body."

    return (
        SYSTEM_PROMPT.format(
            branch_name=BRANCH_NAME,
            issue_number=ISSUE_NUMBER,
            issue_title=ISSUE_TITLE,
            issue_summary=issue_summary,
            plan_section=plan_section,
        )
        + build_required_files_section()
    )


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------


def run_agent() -> tuple[bool, int]:
    """Run the agentic loop. Returns (success: bool, turns_used: int)."""
    log(f"Loading issue context from {ISSUE_CONTEXT_PATH}")
    issue_data = load_issue_context(ISSUE_CONTEXT_PATH)

    system_prompt = build_system_prompt(issue_data)
    log(f"System prompt built ({len(system_prompt)} chars)")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    log(f"Initialized API client: {API_BASE_URL} / model={MODEL}")

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
    ]
    if AGENT_MODE == "revise":
        messages.append({
            "role": "user",
            "content": (
                "Address the review concerns listed in the system prompt. "
                "Start by running git diff to see what this branch has changed, "
                "then modify the flagged files to resolve each concern."
            ),
        })
    else:
        messages.append({
            "role": "user",
            "content": (
                f"Implement the changes for issue #{ISSUE_NUMBER}. "
                "Start by exploring the repository structure, then follow the approved plan."
            ),
        })

    turns = 0
    consecutive_errors = 0
    max_consecutive_errors = 5

    # Stuck-loop detection: track signatures of consecutive identical failing tool calls
    repeated_fail_sigs: list[str] = []
    stuck_redirects = 0

    # Stuck-loop detection: track consecutive identical successful tool calls that
    # never make progress (no intervening edit_file/write_file).
    repeated_success_sigs: list[str] = []
    stuck_success_redirects = 0
    wrote_since_success_reset = False

    # Track whether the agent has attempted any file edits, to detect
    # text-only responses before any work was done.
    has_attempted_edit = False
    text_only_redirects = 0
    max_text_only_redirects = 2

    while turns < MAX_TURNS:
        turns += 1
        log(f"--- Turn {turns}/{MAX_TURNS} ---")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.0,
            )
        except Exception as e:
            log(f"API call failed: {e}")
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                log(f"Too many consecutive API errors ({consecutive_errors}), aborting")
                return False, turns
            log("Retrying in 5 seconds...")
            time.sleep(5)
            continue

        consecutive_errors = 0
        choice = response.choices[0]

        # If the model produced a text response with no tool calls, it's done
        if choice.finish_reason == "stop" or not choice.message.tool_calls:
            # Redirect text-only responses when no edits have been made yet
            if not has_attempted_edit and text_only_redirects < max_text_only_redirects:
                text_only_redirects += 1
                log(
                    f"Agent returned text-only without making any edits "
                    f"(redirect {text_only_redirects}/{max_text_only_redirects}). "
                    f"Injecting redirect."
                )
                messages.append({
                    "role": "assistant",
                    "content": choice.message.content or "",
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "You have not made any code changes yet. Do not respond with "
                        "text only - you must use your tools to explore the repository, "
                        "make the necessary edits with edit_file or write_file, and commit "
                        "your changes with git. Start now by calling a tool."
                    ),
                })
                continue
            if choice.message.content:
                log(f"Agent finished: {choice.message.content[:200]}")
            else:
                log("Agent finished (no final message)")
            messages.append(choice.message.model_dump())
            return True, turns

        # Process tool calls
        messages.append(choice.message.model_dump())

        for tool_call in choice.message.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}
                log(f"  Warning: failed to parse args for {fn_name}")

            log(f"  Tool: {fn_name}({_summarize_args(fn_args)})")

            result = execute_tool(fn_name, fn_args)

            # Any edit/write call breaks the "read loop that never writes" signal.
            if fn_name in ("edit_file", "write_file"):
                has_attempted_edit = True
                wrote_since_success_reset = True

            if result["is_error"]:
                log(f"  Error: {result['output'][:200]}")
                # Track consecutive identical failing tool calls
                sig = hashlib.md5(
                    f"{fn_name}:{json.dumps(fn_args, sort_keys=True)}".encode()
                ).hexdigest()
                repeated_fail_sigs.append(sig)
            else:
                output_preview = result["output"][:100].replace("\n", " ")
                log(f"  OK: {output_preview}...")
                # Successful tool call resets failure tracking
                repeated_fail_sigs = []
                # Track consecutive identical successful tool calls using the same
                # signature (fn_name + sorted args) as the failing-call detector.
                sig = hashlib.md5(
                    f"{fn_name}:{json.dumps(fn_args, sort_keys=True)}".encode()
                ).hexdigest()
                if repeated_success_sigs and repeated_success_sigs[-1] == sig:
                    repeated_success_sigs.append(sig)
                else:
                    repeated_success_sigs = [sig]
                    wrote_since_success_reset = False

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result["output"],
                }
            )

        # Stuck-loop detection: check for repeated identical failing tool calls
        if len(repeated_fail_sigs) >= 3 and len(set(repeated_fail_sigs[-3:])) == 1:
            log(
                f"WARNING: Agent repeated the same failing tool call "
                f"{len(repeated_fail_sigs)} times consecutively"
            )
            if stuck_redirects >= 2:
                log(
                    "Agent stuck in repeated failure loop after redirect "
                    "attempts, aborting"
                )
                return False, turns
            # Inject a redirect message to nudge the model toward a different approach
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "SYSTEM: You have repeated the same failing tool call "
                        f"{len(repeated_fail_sigs)} times with the same error each "
                        "time. This approach is not working. Try a completely "
                        "different strategy — use a different tool, different "
                        "command syntax, or different approach to accomplish the task."
                    ),
                }
            )
            stuck_redirects += 1
            repeated_fail_sigs = []
            log(f"Redirect injected (attempt {stuck_redirects}/2)")

        # Stuck-loop detection: check for repeated identical successful tool calls
        # with no intervening edit_file/write_file (a read loop that never writes).
        if (
            len(repeated_success_sigs) >= 3
            and len(set(repeated_success_sigs[-3:])) == 1
            and not wrote_since_success_reset
        ):
            log(
                f"WARNING: Agent repeated the same successful tool call "
                f"{len(repeated_success_sigs)} times consecutively with no "
                f"edit_file/write_file (success repetition loop)"
            )
            if stuck_success_redirects >= 2:
                log(
                    "Agent stuck in success repetition loop after redirect "
                    "attempts, aborting"
                )
                return False, turns
            # Inject a redirect message to nudge the model toward making progress
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "SYSTEM: You have repeated the same successful tool call "
                        f"{len(repeated_success_sigs)} times with no edits or writes. "
                        "You are not making progress. Try a completely different "
                        "strategy — make the actual code changes required, or use a "
                        "different approach to accomplish the task."
                    ),
                }
            )
            stuck_success_redirects += 1
            repeated_success_sigs = []
            wrote_since_success_reset = False
            log(f"Success-loop redirect injected (attempt {stuck_success_redirects}/2)")

        # Context window management: if messages are getting very large,
        # summarize older tool results to stay within limits
        total_chars = sum(
            len(json.dumps(m)) for m in messages
        )
        if total_chars > CONTEXT_BUDGET_CHARS:
            log(f"Context approaching limit ({total_chars} chars), compacting...")
            messages = _compact_messages(messages)

    log(f"Hit max_turns limit ({MAX_TURNS})")
    return False, turns


def _summarize_args(args: dict) -> str:
    """Produce a short summary of tool call arguments for logging."""
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 60:
            parts.append(f"{k}='{v[:60]}...'")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _total_chars(messages: list[dict]) -> int:
    """Total serialized character count of a message list."""
    return sum(len(json.dumps(m)) for m in messages)


def _truncate_content(msg: dict, limit: int) -> None:
    """Truncate a message's string content in place to `limit` chars."""
    content = msg.get("content", "")
    if isinstance(content, str) and len(content) > limit:
        msg["content"] = content[:limit] + "\n[TRUNCATED for context management]"


def _compact_messages(messages: list[dict]) -> list[dict]:
    """Compact messages to stay within the context budget.

    Progressive, budget-aware compaction that operates on every message
    except the system prompt (messages[0]), which is always preserved.
    Reclaims space in phases, stopping as soon as the total char count
    drops under CONTEXT_BUDGET_CHARS:

    1. Truncate all reclaimable messages (tool/assistant/user) to 2000 chars.
    2. Truncate all reclaimable messages to 500 chars.
    3. Drop the oldest tool messages entirely, one by one (oldest first).
    4. Truncate assistant/user messages to 200 chars.
    5. Drop the oldest assistant/user messages until the budget is met.

    Tool output is sacrificed before assistant reasoning, and user messages
    last. If the budget still cannot be met after exhausting all reclaimable
    content, a distinct WARNING is logged so the budget constants can be
    tuned rather than failing silently.
    """
    if len(messages) <= 1:
        return messages

    before_chars = _total_chars(messages)
    compacted = [dict(m) for m in messages]
    reclaimable = ("tool", "assistant", "user")

    def _under_budget() -> bool:
        return _total_chars(compacted) <= CONTEXT_BUDGET_CHARS

    def _finish() -> list[dict]:
        after_chars = _total_chars(compacted)
        reclaimed = before_chars - after_chars
        log(
            f"Compacted messages: {before_chars} -> {after_chars} chars "
            f"({reclaimed} reclaimed)"
        )
        return compacted

    # Phase 1 — Generative truncation: shrink all reclaimable messages to 2000 chars.
    for msg in compacted[1:]:
        if msg.get("role") in reclaimable:
            _truncate_content(msg, 2000)
    if _under_budget():
        return _finish()

    # Phase 2 — Aggressive truncation: shrink all reclaimable messages to 500 chars.
    for msg in compacted[1:]:
        if msg.get("role") in reclaimable:
            _truncate_content(msg, 500)
    if _under_budget():
        return _finish()

    # Phase 3 — Drop the oldest tool messages entirely (oldest first).
    for msg in list(compacted[1:]):
        if msg.get("role") == "tool":
            compacted.remove(msg)
            if _under_budget():
                return _finish()

    # Phase 4 — Truncate assistant/user messages to 200 chars.
    for msg in compacted[1:]:
        if msg.get("role") in ("assistant", "user"):
            _truncate_content(msg, 200)
    if _under_budget():
        return _finish()

    # Phase 5 — Drop the oldest assistant/user messages until under budget.
    for msg in list(compacted[1:]):
        if msg.get("role") in ("assistant", "user"):
            compacted.remove(msg)
            if _under_budget():
                return _finish()

    # Exhausted all reclaimable content and still over budget — log distinctly.
    log(
        f"WARNING: Compaction exhausted all reclaimable content but context is "
        f"still over budget: {_total_chars(compacted)} chars "
        f"(threshold {CONTEXT_BUDGET_CHARS}). Budget constants need tuning."
    )
    return _finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def set_output(name: str, value: str) -> None:
    """Write a GitHub Actions output variable."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{name}={value}\n")


def main() -> None:
    log(f"DSG Coding Agent v{VERSION} starting")
    log(f"  Model: {MODEL}")
    log(f"  API base: {API_BASE_URL}")
    log(f"  Max turns: {MAX_TURNS}")
    log(f"  Issue: #{ISSUE_NUMBER} — {ISSUE_TITLE}")
    log(f"  Branch: {BRANCH_NAME}")
    log(f"  Mode: {AGENT_MODE}")
    if AGENT_REQUIRED_FILES and AGENT_REQUIRED_FILES != "*":
        log(f"  Required files: {AGENT_REQUIRED_FILES}")

    success, turns_used = run_agent()

    set_output("turns_used", str(turns_used))

    if success:
        log(f"Agent completed successfully in {turns_used} turns")
        set_output("outcome", "success")
    else:
        log(f"Agent did not complete (used {turns_used} turns)")
        set_output("outcome", "failure")
        # Exit 1 so the workflow's safety-net autocommit step can
        # detect failure and create a WIP commit
        sys.exit(1)


if __name__ == "__main__":
    main()
