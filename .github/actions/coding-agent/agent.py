#!/usr/bin/env python3
"""DSG Coding Agent — agentic loop for the CI/CD pipeline.

Replaces anthropics/claude-code-action@v1 with a provider-agnostic agentic
coding loop. Uses any OpenAI-compatible API (DeepSeek, OpenAI, etc).

Environment variables (set by the composite action):
    AGENT_API_KEY         — API key for the LLM provider
    AGENT_API_BASE_URL    — Base URL (default: https://api.deepseek.com)
    AGENT_MODEL           — Model name (default: deepseek-chat)
    AGENT_ISSUE_CONTEXT   — Path to .agent/issue-context.json
    AGENT_BRANCH_NAME     — Git branch to work on
    AGENT_ISSUE_NUMBER    — GitHub issue number
    AGENT_ISSUE_TITLE     — GitHub issue title
    AGENT_MAX_TURNS       — Max agentic loop iterations (default: 40)
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
API_BASE_URL = os.environ.get("AGENT_API_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("AGENT_MODEL", "deepseek-chat")
ISSUE_CONTEXT_PATH = os.environ["AGENT_ISSUE_CONTEXT"]
BRANCH_NAME = os.environ["AGENT_BRANCH_NAME"]
ISSUE_NUMBER = os.environ["AGENT_ISSUE_NUMBER"]
ISSUE_TITLE = os.environ.get("AGENT_ISSUE_TITLE", "")
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "40"))
AGENT_MODE = os.environ.get("AGENT_MODE", "implement")
AGENT_REVIEW_CONCERNS = os.environ.get("AGENT_REVIEW_CONCERNS", "")


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
    """Build a concise issue summary for the system prompt."""
    title = issue_data.get("title", "")
    body = issue_data.get("body", "")

    # Truncate very long bodies
    if len(body) > 8000:
        body = body[:8000] + "\n\n... [truncated]"

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

WORKFLOW:
1. Run `git diff origin/main...HEAD` to understand what this branch has changed.
2. Run `git log --oneline -10` to see commit history.
3. For EACH file mentioned in the review concerns, read the CURRENT file content before editing.
4. Read .agent/issue-context.json if present for original issue context.
5. Address ONLY the review concerns below. Do not refactor or touch anything else.
6. After editing, verify with `git diff` that changes look correct.
7. Commit with a message listing the specific files changed and what was changed.

REVIEW CONCERNS TO ADDRESS:
{review_concerns}

END REVIEW CONCERNS"""


def build_system_prompt(issue_data: dict) -> str:
    if AGENT_MODE == "revise":
        return REVISION_SYSTEM_PROMPT.format(
            branch_name=BRANCH_NAME,
            review_concerns=AGENT_REVIEW_CONCERNS,
        )

    issue_summary = build_issue_summary(issue_data)
    plan = extract_latest_plan(issue_data)
    if plan:
        plan_section = f"APPROVED PLAN (follow this exactly):\n{plan}"
    else:
        plan_section = "NO APPROVED PLAN FOUND — implement based on the issue body."

    return SYSTEM_PROMPT.format(
        branch_name=BRANCH_NAME,
        issue_number=ISSUE_NUMBER,
        issue_title=ISSUE_TITLE,
        issue_summary=issue_summary,
        plan_section=plan_section,
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
        {
            "role": "user",
            "content": (
                f"Implement the changes for issue #{ISSUE_NUMBER}. "
                "Start by exploring the repository structure, then follow the approved plan."
            ),
        },
    ]

    turns = 0
    consecutive_errors = 0
    max_consecutive_errors = 5

    # Stuck-loop detection: track signatures of consecutive identical failing tool calls
    repeated_fail_sigs: list[str] = []
    stuck_redirects = 0

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

        # Context window management: if messages are getting very large,
        # summarize older tool results to stay within limits
        total_chars = sum(
            len(json.dumps(m)) for m in messages
        )
        if total_chars > 180_000:  # ~45K tokens, approaching 64K limit
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


def _compact_messages(messages: list[dict]) -> list[dict]:
    """Truncate older tool results to free up context space.

    Keeps the system prompt and last 10 messages intact; truncates tool
    content in older messages to 500 chars each.
    """
    if len(messages) <= 12:
        return messages

    compacted = [messages[0]]  # system prompt
    boundary = len(messages) - 10

    for i, msg in enumerate(messages[1:], 1):
        if i < boundary and msg.get("role") == "tool":
            content = msg.get("content", "")
            if len(content) > 500:
                msg = dict(msg)
                msg["content"] = content[:500] + "\n[TRUNCATED for context management]"
        compacted.append(msg)

    log(f"Compacted messages from {len(messages)} to {len(compacted)} entries")
    return compacted


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
