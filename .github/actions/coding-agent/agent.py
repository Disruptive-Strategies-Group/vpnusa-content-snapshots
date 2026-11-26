#!/usr/bin/env python3
"""DSG Coding Agent — agentic loop for the CI/CD pipeline.

Replaces anthropics/claude-code-action@v1 with a provider-agnostic agentic
coding loop. Uses any OpenAI-compatible API (DeepSeek, OpenAI, etc).

Environment variables (set by the composite action):
    AGENT_PROVIDER        — "openai" (default, any OpenAI-compatible API) or "bedrock"
                            (AWS Bedrock Converse via bedrock_client.py; US-only, uses the
                            runner's AWS credentials, no API key needed)
    AGENT_BEDROCK_REGION  — Bedrock region (default: us-east-2); model IDs must be us.* /
                            in-region IDs, never global.*
    AGENT_API_KEY         — API key for the LLM provider (not used when AGENT_PROVIDER=bedrock)
    AGENT_API_BASE_URL    — Base URL (default: https://openrouter.ai/api/v1)
    AGENT_MODEL           — Model name (default: deepseek/deepseek-v4-flash-0731)
    AGENT_ESCALATION_MODEL — Model to escalate to on no_progress (default: deepseek/deepseek-v4-pro)
    AGENT_ISSUE_CONTEXT   — Path to .agent/issue-context.json
    AGENT_BRANCH_NAME     — Git branch to work on
    AGENT_ISSUE_NUMBER    — GitHub issue number
    AGENT_ISSUE_TITLE     — GitHub issue title
    AGENT_MAX_TURNS       — Max agentic loop iterations (default: 120)
    AGENT_MODE            — implement | revise (default: implement)
    AGENT_REVIEW_CONCERNS — Review concerns text (revise mode)
    AGENT_REQUIRED_FILES  — JSON-encoded array of file paths the agent must
                            restrict its changes to; "*" means no restriction
"""

from __future__ import annotations

import collections
import hashlib
import httpx
import json
import os
import re
import signal
import subprocess
import sys
import time

from openai import OpenAI

from tools import TOOL_SCHEMAS, execute_tool

VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AGENT_PROVIDER = os.environ.get("AGENT_PROVIDER", "openai").strip().lower()
if AGENT_PROVIDER not in ("openai", "bedrock"):
    raise SystemExit(f"Unsupported AGENT_PROVIDER: {AGENT_PROVIDER!r} (expected 'openai' or 'bedrock')")
# The key is only needed for OpenAI-compatible providers; Bedrock uses AWS credentials.
API_KEY = os.environ["AGENT_API_KEY"] if AGENT_PROVIDER == "openai" else os.environ.get("AGENT_API_KEY", "")
API_BASE_URL = os.environ.get("AGENT_API_BASE_URL", "https://openrouter.ai/api/v1")
# Provider-appropriate model defaults. Bedrock needs explicit Bedrock model IDs
# (e.g. us.anthropic.claude-haiku-5-5, zai.glm-5); set AGENT_MODEL for real use.
if AGENT_PROVIDER == "bedrock":
    DEFAULT_MODEL = "zai.glm-5"
    DEFAULT_ESCALATION_MODEL = "zai.glm-5"
else:
    DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
    DEFAULT_ESCALATION_MODEL = "deepseek/deepseek-v4-pro"
ISSUE_CONTEXT_PATH = os.environ["AGENT_ISSUE_CONTEXT"]
BRANCH_NAME = os.environ["AGENT_BRANCH_NAME"]
ISSUE_NUMBER = os.environ["AGENT_ISSUE_NUMBER"]
ISSUE_TITLE = os.environ.get("AGENT_ISSUE_TITLE", "")
MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "120"))
AGENT_MODE = os.environ.get("AGENT_MODE", "implement")
AGENT_REVIEW_CONCERNS = os.environ.get("AGENT_REVIEW_CONCERNS", "")
AGENT_REQUIRED_FILES = os.environ.get("AGENT_REQUIRED_FILES", "*")

# Write-progress and escalation configuration (see SECURITY.md rationale).
# WRITE_THRESHOLD_PCT: if no write has happened by this percent of max_turns,
# the agent escalates to AGENT_ESCALATION_MODEL.
WRITE_THRESHOLD_PCT = int(os.environ.get("AGENT_WRITE_THRESHOLD_PCT", "60"))
# TURNS_PER_FILE: additional turns granted per required file beyond the first.
TURNS_PER_FILE = int(os.environ.get("AGENT_TURNS_PER_FILE", "8"))
# FIRST_WRITE_BY_TURN: hard deadline for producing the first write/edit in revise
# mode; used in the system prompt to force early progress.
FIRST_WRITE_BY_TURN = int(os.environ.get("AGENT_FIRST_WRITE_BY_TURN", "8"))
# Up-front file content injection caps for revise mode (chars). Per-file cap
# prevents a single huge file from dominating the prompt; the total cap bounds
# the whole injection block. Truncated files are marked in the prompt.
INJECT_PER_FILE_CAP = int(os.environ.get("AGENT_INJECT_PER_FILE_CAP", "12000"))
INJECT_TOTAL_CAP = int(os.environ.get("AGENT_INJECT_TOTAL_CAP", "40000"))

# Context budget for the message window. When total serialized chars exceed
# this, _compact_messages() reclaims space down to
# COMPACT_TARGET_RATIO * CONTEXT_BUDGET_CHARS (headroom so the next few tool
# results do not immediately re-trigger compaction on the following turn).
# Tunable per workflow via the AGENT_CONTEXT_BUDGET_CHARS env var; the default
# 300000 chars (~75K tokens) is safe for any model with a 128K+ token window.
CONTEXT_BUDGET_CHARS = int(os.environ.get("AGENT_CONTEXT_BUDGET_CHARS", "300000"))
# Compact to this fraction of the budget so compaction does not run every turn.
COMPACT_TARGET_RATIO = 0.6
# The most recent tool messages (and the assistant messages that issued them)
# are never truncated or dropped, so freshly-read file content survives
# compaction long enough to be reproduced in an edit_file(old_text=...) call.
PROTECTED_RECENT_TOOL_MESSAGES = 8
# Truncation limits for OLDER tool messages (phases 1 and 2 of compaction).
OLDER_TOOL_LIMIT_PHASE1 = 6000
OLDER_TOOL_LIMIT_PHASE2 = 1500

# Per-request timeout for the OpenAI client, in seconds. DeepSeek V4 Flash's
# best-provider P50 latency is ~0.41s and even long tool-loaded prompts
# complete well under a minute; anything past 90s is almost certainly a
# stalled OpenRouter provider connection worth aborting so the existing retry
# loop can land on a different provider. Without this timeout a silent stall
# (no response, no error) would hang the agent until the job-level timeout.
# Tunable via AGENT_API_REQUEST_TIMEOUT.
API_REQUEST_TIMEOUT = float(os.environ.get("AGENT_API_REQUEST_TIMEOUT", "90"))

# Per-turn wall clock timeout (seconds). Wraps the entire turn body — API call
# plus tool execution — so no single turn can hang indefinitely regardless of
# what stalls. Must exceed API_REQUEST_TIMEOUT to allow at least one full API
# attempt before the alarm fires. A turn includes tool execution (bash, git),
# so the default is not lowered below 120s. Tunable via AGENT_TURN_TIMEOUT.
TURN_WALL_CLOCK_TIMEOUT = int(os.environ.get("AGENT_TURN_TIMEOUT", "150"))  # 2.5 minutes (exceeds API_REQUEST_TIMEOUT by 60s)

# No-progress detection: hard-stop when the agent burns too many turns without
# any edit_file/write_file call (e.g. an endless successful-but-non-writing
# exploration loop of bash awk/sed/od-style byte inspections). The agent gets
# NO_PROGRESS_GRACE_TURNS of exploration before the detector arms; once past
# that, sys.exit(4) fires when turns_since_last_write reaches
# NO_PROGRESS_THRESHOLD. Exit code 4 is distinct from exit code 3 (error-loop
# detection) so the workflow can surface ROOT_CAUSE: no_progress.
NO_PROGRESS_THRESHOLD = 25
NO_PROGRESS_GRACE_TURNS = 30
NO_PROGRESS_NUDGE_AT = 15

# ---------------------------------------------------------------------------
# LLM usage tracking (issue #980)
#
# Records per-call and per-run token usage so runner model spend is trackable
# in CloudWatch next to the Lambda-side usage logged by the compute service
# (dsg-agentic-assistant-compute#594, message `llm_usage`). This changes no
# model, provider or prompt — it only reads response.usage and logs it.
# The call function is kept provider-neutral because it will later be replaced
# by a Bedrock adapter.
# ---------------------------------------------------------------------------

# Per-model USD rates per 1M tokens (input / output). Seeded with the models
# the coding agent uses; an unknown model reports estimatedCostUsd: null and
# never raises. Optional "cached" rate prices cache-read input tokens and
# "cache_write" prices cache-write input tokens (default for both: the normal
# input rate). Haiku 5.5 bills in two tiers chosen per request: requests with
# <= threshold_tokens prompt tokens use the standard rates below; requests
# with more bill EVERY token of that request at the "long_context" rates
# (Anthropic pricing page; both usage types appear on our Bedrock bill).
# The Bedrock rates are AWS Cost Explorer effective rates (us-east-2, billed
# usage, checked 2026-11-14, issue #990).
MODEL_RATES_PER_1M: dict[str, dict[str, float | int | dict[str, float | int]]] = {
    "deepseek/deepseek-v4-flash-0731": {"input": 0.14, "output": 0.28},
    "deepseek/deepseek-v4-pro": {"input": 1.25, "output": 2.50},
    "zai.glm-5": {"input": 1.0, "output": 3.2},
    "zai.glm-4.7": {"input": 0.6, "output": 2.2},
    "zai.glm-4.7-flash": {"input": 0.07, "output": 0.4},
    "us.anthropic.claude-haiku-5-5": {
        "input": 0.11,
        "output": 0.55,
        "cached": 0.011,
        "cache_write": 0.1375,
        "long_context": {
            "input": 0.55,
            "output": 2.75,
            "cached": 0.055,
            "cache_write": 0.6875,
            "threshold_tokens": 100_000,
        },
    },
    "us.anthropic.claude-sonnet-5-5": {
        "input": 2.2,
        "output": 11.0,
        "cached": 0.11,
        "cache_write": 2.75,
    },
}

# Per-model accumulated run totals:
#   model -> {"calls": int, "promptTokens": int, "completionTokens": int,
#             "cachedTokens": int, "cacheWriteTokens": int,
#             "estimatedCostUsd": float, "longContextCalls": int,
#             "maxPromptTokens": int}
# Escalation can swap the model mid-run, so totals are kept per model.
USAGE_STATE: dict[str, dict[str, int | float]] = {}


def _usage_totals(model: str) -> dict[str, int | float]:
    return USAGE_STATE.setdefault(
        model,
        {
            "calls": 0,
            "promptTokens": 0,
            "completionTokens": 0,
            "cachedTokens": 0,
            "cacheWriteTokens": 0,
            "estimatedCostUsd": 0.0,
            "longContextCalls": 0,
            "maxPromptTokens": 0,
        },
    )


def _cache_write_tokens(usage) -> int:
    """Cache-write input tokens from a Bedrock response usage; 0 when absent."""
    return int(getattr(usage, "cache_write_tokens", 0) or 0)


def _uses_long_context_tier(model: str, prompt_size: int) -> bool:
    """True when prompt_size (input + cache read + cache write) exceeds the
    model's long-context threshold, so every token of the call is billed at
    the long-context rates. ASSUMPTION (issue #990): cache tokens count toward
    the 100K threshold; AWS docs do not state this either way, but Cost
    Explorer lists long-context cache usage that way, so we follow it."""
    long_context = MODEL_RATES_PER_1M.get(model, {}).get("long_context")
    return long_context is not None and prompt_size > long_context.get(
        "threshold_tokens", 0
    )


def estimated_cost_usd(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Estimate USD cost for a single API call; None when the rate is unknown.

    prompt_tokens is the call's non-cached input token count (Bedrock
    ``inputTokens``; for OpenAI-compatible providers that is the reported
    ``usage.prompt_tokens`` minus ``cached_tokens``). cached_tokens and
    cache_write_tokens are the cache-read and cache-write input tokens
    (default 0); the call's total prompt size is the sum of the three.

    When the model defines a ``long_context`` tier and the call's prompt size
    exceeds ``threshold_tokens``, every token of that call — input, output,
    cache read and cache write — is priced at the long-context rates.
    """
    rate = MODEL_RATES_PER_1M.get(model)
    if rate is None:
        return None
    input_tokens = max(prompt_tokens, 0)
    completion_tokens = max(completion_tokens, 0)
    cached = max(cached_tokens, 0)
    cache_write = max(cache_write_tokens, 0)
    prompt_size = input_tokens + cached + cache_write

    if _uses_long_context_tier(model, prompt_size):
        rate = rate["long_context"]
    cached_rate = rate.get("cached", rate["input"])
    cache_write_rate = rate.get("cache_write", rate["input"])
    return (
        (input_tokens / 1_000_000) * rate["input"]
        + (cached / 1_000_000) * cached_rate
        + (cache_write / 1_000_000) * cache_write_rate
        + (completion_tokens / 1_000_000) * rate["output"]
    )


def _usage_counts(usage) -> tuple[int, int, int]:
    """Extract (prompt, completion, cached) token counts, defaulting missing fields to 0."""
    if usage is None:
        return 0, 0, 0
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    # OpenAI-compatible providers expose cached tokens as
    # usage.prompt_tokens_details.cached_tokens; 0 when absent.
    cached_tokens = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
    return prompt_tokens, completion_tokens, cached_tokens


def record_usage(model: str, usage) -> None:
    """Accumulate one completion call's token totals and per-call cost into USAGE_STATE.

    Cost is computed per API call (not from run totals) because Haiku 5.5's
    long-context tier depends on each call's individual prompt size (issue #990).
    """
    prompt_tokens, completion_tokens, cached_tokens = _usage_counts(usage)
    cache_write_tokens = _cache_write_tokens(usage)
    totals = _usage_totals(model)
    totals["calls"] += 1
    totals["promptTokens"] += prompt_tokens
    totals["completionTokens"] += completion_tokens
    totals["cachedTokens"] += cached_tokens
    totals["cacheWriteTokens"] += cache_write_tokens

    # Non-cached input is what Bedrock reports as inputTokens; for
    # OpenAI-compatible providers prompt_tokens already includes the cached
    # tokens, so subtract them to avoid double-charging.
    input_tokens = max(prompt_tokens - cached_tokens - cache_write_tokens, 0)
    prompt_size = input_tokens + cached_tokens + cache_write_tokens
    totals["maxPromptTokens"] = max(totals["maxPromptTokens"], prompt_size)
    cost = estimated_cost_usd(
        model, input_tokens, completion_tokens, cached_tokens, cache_write_tokens
    )
    if cost is not None:
        totals["estimatedCostUsd"] += cost
        if _uses_long_context_tier(model, prompt_size):
            totals["longContextCalls"] += 1


def print_llm_usage(model: str, turn: int, usage, duration_ms: float, finish_reason: str | None) -> None:
    """Print one JSON line per LLM call to stdout, prefixed 'LLM_USAGE '.

    Field names match the Lambda-side schema (dsg-agentic-assistant-compute#594).
    Never logs message content.
    """
    prompt_tokens, completion_tokens, cached_tokens = _usage_counts(usage)
    cache_write_tokens = _cache_write_tokens(usage)
    record_usage(model, usage)
    input_tokens = max(prompt_tokens - cached_tokens - cache_write_tokens, 0)
    prompt_size = input_tokens + cached_tokens + cache_write_tokens
    payload = {
        "role": "coding_agent",
        "mode": AGENT_MODE,
        "model": model,
        "turn": turn,
        "promptTokens": prompt_tokens,
        "completionTokens": completion_tokens,
        "cachedTokens": cached_tokens,
        "cacheWriteTokens": cache_write_tokens,
        "maxPromptTokens": prompt_size,
        "longContextCalls": 1 if _uses_long_context_tier(model, prompt_size) else 0,
        "durationMs": int(duration_ms),
        "finishReason": finish_reason,
    }
    print(f"LLM_USAGE {json.dumps(payload, sort_keys=True)}", flush=True)


def build_usage_summary() -> dict:
    """Per-model totals plus run metadata for the LLM_USAGE_SUMMARY line."""
    models = []
    total_calls = 0
    total_cost = 0.0
    any_known = False
    for model, totals in USAGE_STATE.items():
        # Per-call costs are accumulated in record_usage (the long-context
        # tier is decided per call, so run totals can't be priced in bulk).
        cost = totals["estimatedCostUsd"]
        if MODEL_RATES_PER_1M.get(model) is None:
            cost = None
        if cost is not None:
            total_cost += cost
            any_known = True
        models.append({
            "model": model,
            "calls": totals["calls"],
            "promptTokens": totals["promptTokens"],
            "completionTokens": totals["completionTokens"],
            "cachedTokens": totals["cachedTokens"],
            "cacheWriteTokens": totals["cacheWriteTokens"],
            "longContextCalls": totals["longContextCalls"],
            "maxPromptTokens": totals["maxPromptTokens"],
            "estimatedCostUsd": cost,
        })
        total_calls += totals["calls"]
    models.sort(key=lambda m: m["model"])
    return {
        "role": "coding_agent",
        "mode": AGENT_MODE,
        "repo": os.environ.get("GITHUB_REPOSITORY", ""),
        "issueNumber": ISSUE_NUMBER,
        "runId": os.environ.get("GITHUB_RUN_ID", ""),
        "models": models,
        "totalCalls": total_calls,
        "estimatedCostUsd": round(total_cost, 6) if any_known else None,
    }


def print_usage_summary() -> None:
    """Print the LLM_USAGE_SUMMARY line, set the usage-json output and append
    a small cost table to $GITHUB_STEP_SUMMARY. Never raises."""
    try:
        summary = build_usage_summary()
    except Exception as e:
        log(f"Failed to build LLM usage summary: {e}")
        return

    print(f"LLM_USAGE_SUMMARY {json.dumps(summary, sort_keys=True)}", flush=True)

    try:
        set_output("usage-json", json.dumps(summary, sort_keys=True))
    except Exception as e:
        log(f"Failed to write usage-json output: {e}")

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return
    try:
        rows = []
        for m in summary["models"]:
            cost = m["estimatedCostUsd"]
            cost_str = f"${cost:.6f}" if cost is not None else "n/a"
            rows.append(
                f"| {m['model']} | {m['calls']} | {m['promptTokens']} | "
                f"{m['completionTokens']} | {cost_str} |"
            )
        table = (
            "## LLM Usage Summary\n\n"
            "| Model | Calls | Input tokens | Output tokens | Estimated cost (USD) |\n"
            "|---|---|---|---|---|\n"
            + "\n".join(rows)
            + "\n"
        )
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(table)
    except Exception as e:
        log(f"Failed to append LLM usage table to job summary: {e}")


class TurnTimeoutError(Exception):
    """Raised when a single agentic turn exceeds TURN_WALL_CLOCK_TIMEOUT."""
    pass

def _turn_timeout_handler(signum, frame):
    raise TurnTimeoutError(f"Turn exceeded {TURN_WALL_CLOCK_TIMEOUT}s wall clock")


def log(msg: str) -> None:
    print(f"[agent] {msg}", flush=True)


def _extract_error_signature(tool_result_text: str) -> str | None:
    """Normalize common import/mock error patterns from tool output.

    Returns a stable signature string used by the repeated-error-loop
    detector, or None when the text matches no known pattern:

      - ImportError: <module_name>                          -> import:<module_name>
      - ModuleNotFoundError: <module_name>                  -> import:<module_name>
      - AttributeError: module '<x>' has no attribute '<y>' -> attr:<x>.<y>
      - TypeError: ... argument '<name>' ...                -> typearg:<name>
    """
    if not tool_result_text:
        return None
    text = tool_result_text.strip()
    # ModuleNotFoundError: No module named 'pandas'
    m = re.search(
        r"(?:ImportError|ModuleNotFoundError):\s*No module named\s+'([^']+)'",
        text,
    )
    if m:
        return f"import:{m.group(1)}"
    # ImportError: numpy / ModuleNotFoundError: numpy
    m = re.search(r"(?:ImportError|ModuleNotFoundError):\s*([A-Za-z_][\w.]*)", text)
    if m:
        return f"import:{m.group(1)}"
    m = re.search(
        r"AttributeError:\s*module\s+'([^']+)'\s+has no attribute\s+'([^']+)'",
        text,
    )
    if m:
        return f"attr:{m.group(1)}.{m.group(2)}"
    m = re.search(r"TypeError:.*?argument\s*:?\s+'([^']+)'", text)
    if m:
        return f"typearg:{m.group(1)}"
    return None


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
- Never use [skip ci], [ci skip], [no ci], or skip-checks: true in commit messages. The git commit hook will REJECT any commit message containing these markers. If your commit fails with 'forbidden CI-skip marker', remove the marker and retry — do not force-add it.
- NEVER use `python3 -c` or `python -c` to validate file changes. The runner shell is dash on Ubuntu and cannot handle nested parentheses in -c one-liners, which causes stuck retry loops. To verify a change landed, use `read_file` or `grep_search` instead. Once `grep_search` confirms the expected content is present, commit and stop — do not attempt further validation.
- If the issue body or approved plan specifies exact line numbers for a file you need to modify, skip the repository surveying step. Use read_file_lines to read ONLY those line ranges, then call edit_file immediately. Do not read the file from the top — you will waste turns on large files and trigger the no-progress detector.

STOP RULE: After you have made the requested changes AND (if tests were requested) all specified tests pass, commit and end your turn immediately. Do NOT continue exploring, grepping, or reading files. If you find yourself running grep_search, ls, or read_file after tests have passed, STOP — commit and end.

STOP RULE (TEST IMPORT ERRORS): If you run a test or Python script that fails with importing errors (ModuleNotFoundError, ImportError), STOP. Do NOT retry the same failing test. Instead:
1. Read the file requirements-test.txt. If the file exists, install it with: pip install --quiet -r requirements-test.txt. If requirements-test.txt does not exist, read requirements.txt and install it via: pip install --quiet -r requirements.txt.
2. After installing, re-run the test exactly once. If it still fails with imports, commit a SKIP.md explaining which imports failed and end your turn.
3. If no requirements file exists and the missing module is from the same repo, create the missing __init__.py or fix the import path, commit, and end your turn.
Do NOT repeatedly run the same failing test -- this triggers the error-loop detector.

WORKFLOW:
1. Read the issue and the approved plan carefully.
2. If the issue body or approved plan specifies exact file paths and line numbers to modify, skip ahead — use read_file_lines to read only those line ranges and jump directly to implementation (step 3). Only survey the broader repository (list_files, read_file, grep_search) if the issue does not specify exact locations.
3. Implement the changes described in the plan.
4. Use edit_file for targeted changes to existing files. Use write_file for new files.
5. Test your changes if appropriate (run linters, type checks, unit tests via bash).
6. Commit all changes with a clear commit message referencing the issue number.

LARGE FILE HANDLING:
If read_file returns a truncated view of a file (you will see a TRUNCATED message), use grep_search to find the exact line numbers of the code you need to modify, then use read_file_lines to read that specific section. This is essential for files over 50,000 characters. If edit_file reports that old_text appears multiple times, use read_file_lines to read more surrounding context until you find a unique string that appears exactly once, then retry edit_file with that larger old_text. Never attempt edit_file on code you have not directly read — the old_text will not match and the edit will fail.

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
- Never use [skip ci], [ci skip], [no ci], or skip-checks: true in commit messages. The git commit hook will REJECT any commit message containing these markers. If your commit fails with 'forbidden CI-skip marker', remove the marker and retry — do not force-add it.
- NEVER use `python3 -c` or `python -c` to validate file changes. The runner shell is dash on Ubuntu and cannot handle nested parentheses in -c one-liners, which causes stuck retry loops. To verify a change landed, use `read_file` or `grep_search` instead. Once `grep_search` confirms the expected content is present, commit and stop — do not attempt further validation.
- If the issue body or approved plan specifies exact line numbers for a file you need to modify, skip the repository surveying step. Use read_file_lines to read ONLY those line ranges, then call edit_file immediately. Do not read the file from the top — you will waste turns on large files and trigger the no-progress detector.

FIRST WRITE DEADLINE: You MUST produce your first file write (edit_file or write_file) by turn {first_write_by_turn}. The turn budget is limited to {max_turns}. If you have not written anything by turn {first_write_by_turn}, STOP reading/grepping and write SOMETHING immediately — even a partial draft, a skeleton file, or a placeholder. Exploration is not progress; only write_file and edit_file count as progress.

NO RE-READ RULE: Do NOT re-read any file that you have already read during this session. Once you have read a file, you already know its contents. Reading it again wastes turns and will cause you to exhaust the turn budget before finishing the required deliverables. This is especially important after context compaction.

STOP RULE: After you have made the requested changes AND (if tests were requested) all specified tests pass, commit and end your turn immediately. Do NOT continue exploring, grepping, or reading files. If you find yourself running grep_search, ls, or read_file after tests have passed, STOP — commit and end.

STOP RULE (TEST IMPORT ERRORS): If you run a test or Python script that fails with importing errors (ModuleNotFoundError, ImportError), STOP. Do NOT retry the same failing test. Instead:
1. Read the file requirements-test.txt. If the file exists, install it with: pip install --quiet -r requirements-test.txt. If requirements-test.txt does not exist, read requirements.txt and install it via: pip install --quiet -r requirements.txt.
2. After installing, re-run the test exactly once. If it still fails with imports, commit a SKIP.md explaining which imports failed and end your turn.
3. If no requirements file exists and the missing module is from the same repo, create the missing __init__.py or fix the import path, commit, and end your turn.
Do NOT repeatedly run the same failing test -- this triggers the error-loop detector.

WORKFLOW:
1. Know the exact files to create/edit from the review concerns AND the required-files section below. These are your targets.
2. Produce your first write_file or edit_file by turn {first_write_by_turn}. Do not spend early turns exploring — read one or two key files, then write.
3. Run `git diff origin/main...HEAD` to understand what this branch has changed.
4. Run `git log --oneline -10` to see commit history.
5. For EACH file mentioned in the review concerns, read the CURRENT file content before editing. If the review concerns specify exact line numbers for the changes needed, use read_file_lines to read only those line ranges instead of reading the entire file from the top. Reading large files from the top wastes turns and triggers the no-progress abort.
6. Read .agent/issue-context.json if present for original issue context.
7. Address ONLY the review concerns below. Do not refactor or touch anything else.
8. After editing, verify with `git diff` that changes look correct.
9. Commit with a message listing the specific files changed and what was changed.

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
        "REQUIRED DELIVERABLES (MUST WRITE ALL OF THESE):\n"
        "The following files are required deliverables. You MUST write each one "
        "with edit_file or write_file. If a required file does not exist yet, "
        "create it with write_file. Do not modify any other files unless the "
        "review concerns explicitly require it. If a requested change cannot be "
        "made within these files, say so and stop rather than editing unrelated "
        "files.\n"
        f"{file_list}\n"
    )


def parse_required_files() -> list[str]:
    """Parse AGENT_REQUIRED_FILES (a JSON-encoded list) into file paths.

    Returns an empty list for the special value "*", empty strings, and
    unparseable/non-list JSON, meaning "no required files".
    """
    if not AGENT_REQUIRED_FILES or AGENT_REQUIRED_FILES == "*":
        return []
    try:
        parsed = json.loads(AGENT_REQUIRED_FILES)
    except json.JSONDecodeError:
        log(f"Warning: AGENT_REQUIRED_FILES is not valid JSON: {AGENT_REQUIRED_FILES}")
        return []
    if not isinstance(parsed, list):
        return []
    return [f.strip() for f in parsed if isinstance(f, str) and f.strip()]


def compute_effective_max_turns(required_files_list: list[str]) -> int:
    """Scale max_turns for revise mode based on the number of required files.

    Baseline: 25 turns for 1 file, +TURNS_PER_FILE per additional file, capped
    at 60. When the required-files list is empty (e.g. "*"), N = 1. Implement
    mode keeps the workflow-provided MAX_TURNS unchanged.
    """
    if AGENT_MODE != "revise":
        return MAX_TURNS
    n = len(required_files_list) if required_files_list else 1
    scaled = 25 + TURNS_PER_FILE * (n - 1)
    return min(60, max(MAX_TURNS, scaled))


def build_injected_file_contents(required_files_list: list[str]) -> str:
    """Read required files and build the up-front content injection string.

    Injects the current contents of each required file into the initial user
    message so the agent does not need to discover them with read_file/grep
    loops. Caps at INJECT_PER_FILE_CAP chars per file and INJECT_TOTAL_CAP chars
    total; truncated files are marked, and files that do not exist yet are
    marked "(does not exist yet - create it)".
    """
    if not required_files_list:
        return ""
    injected_parts: list[str] = []
    total_injected = 0
    for fpath in required_files_list:
        if total_injected >= INJECT_TOTAL_CAP:
            injected_parts.append(
                "<file path=\"...\">[remaining files omitted — total injection cap reached]</file>"
            )
            break
        if os.path.isfile(fpath):
            try:
                with open(fpath) as _f:
                    content = _f.read()
                truncated = False
                if len(content) > INJECT_PER_FILE_CAP:
                    content = content[:INJECT_PER_FILE_CAP]
                    truncated = True
                if total_injected + len(content) > INJECT_TOTAL_CAP:
                    content = content[: INJECT_TOTAL_CAP - total_injected]
                    truncated = True
                marker = " [TRUNCATED]" if truncated else ""
                injected_parts.append(f"<file path=\"{fpath}\"{marker}>\n{content}\n</file>")
                total_injected += len(content)
                log(f"Injected {fpath} ({len(content)} chars) into initial prompt")
            except Exception as e:
                log(f"Could not read required file {fpath}: {e}")
        else:
            # File does not exist yet (to be created by the agent).
            injected_parts.append(
                f"<file path=\"{fpath}\" status=\"missing\">(does not exist yet - create it)</file>"
            )
    if not injected_parts:
        return ""
    return (
        "\n\n--- UP-FRONT FILE CONTENTS (already provided — do NOT re-read these files) ---\n"
        + "\n".join(injected_parts)
        + "\n--- END UP-FRONT FILE CONTENTS ---\n"
    )


def build_system_prompt(issue_data: dict) -> str:
    if AGENT_MODE == "revise":
        return REVISION_SYSTEM_PROMPT.format(
            branch_name=BRANCH_NAME,
            review_concerns=AGENT_REVIEW_CONCERNS,
            required_files_section=build_required_files_section(),
            first_write_by_turn=FIRST_WRITE_BY_TURN,
            max_turns=MAX_TURNS,
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


def _repo_has_agent_changes(base_ref: str = "origin/main") -> bool:
    """True when the working tree has uncommitted changes, or HEAD has commits
    that are not on the base branch. Changes under .agent/ are ignored.
    Never raises; any git failure returns False.
    """
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            capture_output=True, text=True, timeout=30,
        ).stdout.splitlines()
        for line in status:
            path = line[3:].strip()
            if path and not path.startswith(".agent/"):
                return True
        ahead = subprocess.run(
            ["git", "rev-list", "--count", f"{base_ref}..HEAD"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return ahead.isdigit() and int(ahead) > 0
    except Exception:
        return False



def run_agent() -> tuple[bool, int]:
    """Run the agentic loop. Returns (success: bool, turns_used: int)."""
    log(f"Loading issue context from {ISSUE_CONTEXT_PATH}")
    issue_data = load_issue_context(ISSUE_CONTEXT_PATH)

    # Parse the required-files list (JSON-encoded) and scale max_turns for
    # revise mode. Baseline: 25 turns for 1 file, +TURNS_PER_FILE per additional
    # file, capped at 60. Implement mode keeps MAX_TURNS unchanged.
    required_files_list = parse_required_files()
    effective_max_turns = compute_effective_max_turns(required_files_list)
    if effective_max_turns != MAX_TURNS:
        log(f"Scaled max_turns: {MAX_TURNS} -> {effective_max_turns} ({len(required_files_list) or 1} required files)")

    # Up-front file content injection for revise mode: read required files and
    # inject their contents into the initial user message to eliminate
    # discovery-oriented read_file/grep loops.
    injected_file_contents = build_injected_file_contents(required_files_list) if AGENT_MODE == "revise" else ""

    # Compaction ledger: tracks files read and files written across compaction
    # events so the agent never re-reads files it already knows.
    files_read: set[str] = set()
    files_written: set[str] = set()
    wrote_anything = False

    system_prompt = build_system_prompt(issue_data)
    log(f"System prompt built ({len(system_prompt)} chars)")

    MODEL = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
    AGENT_ESCALATION_MODEL = os.environ.get("AGENT_ESCALATION_MODEL", DEFAULT_ESCALATION_MODEL)

    if AGENT_PROVIDER == "bedrock":
        from bedrock_client import BedrockClient, assert_us_only

        for _m in (MODEL, AGENT_ESCALATION_MODEL):
            assert_us_only(_m)
        client = BedrockClient()
        log(f"Initialized Bedrock client: region={os.environ.get('AGENT_BEDROCK_REGION', 'us-east-2')} / model={MODEL}")
    else:
        client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE_URL,
            timeout=httpx.Timeout(
                connect=30.0,  # 30s to establish connection
                read=90.0,  # 90s max between received chunks
                write=30.0,  # 30s to send request body
                pool=10.0,  # 10s to acquire connection from pool
            ),
            max_retries=0,
        )
        log(f"Initialized API client: {API_BASE_URL} / model={MODEL}")

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
    ]
    if AGENT_MODE == "revise":
        user_content = (
            "Address the review concerns listed in the system prompt. "
            "Start by running git diff to see what this branch has changed, "
            "then modify the flagged files to resolve each concern."
        )
        if injected_file_contents:
            user_content += injected_file_contents
        messages.append({
            "role": "user",
            "content": user_content,
        })
    else:
        user_content = (
            f"Implement the changes for issue #{ISSUE_NUMBER}. "
            "Start by exploring the repository structure, then follow the approved plan."
        )
        if injected_file_contents:
            user_content += injected_file_contents
        messages.append({
            "role": "user",
            "content": user_content,
        })

    turns = 0
    consecutive_errors = 0
    max_consecutive_errors = 3

    # Stuck-loop detection: track signatures of consecutive identical failing tool calls
    repeated_fail_sigs: list[str] = []
    stuck_redirects = 0

    # Stuck-loop detection: track consecutive identical successful tool calls that
    # never make progress (no intervening edit_file/write_file).
    repeated_success_sigs: list[str] = []
    stuck_success_redirects = 0
    wrote_since_success_reset = False

    # Import/mock error-loop detection: track normalized error signatures from
    # tool output (see _extract_error_signature) for the pivot/hard-stop
    # triggers. Initialized once per run.
    error_signature_counts: dict[str, int] = {}
    error_signature_turns: list[tuple[int, str]] = []
    pivoted_error_signatures: set[str] = set()

    # Track whether the agent has attempted any file edits, to detect
    # text-only responses before any work was done.
    has_attempted_edit = False
    text_only_redirects = 0
    max_text_only_redirects = 2

    # No-progress detection: turns since the last edit_file/write_file call.
    # Resets to 0 on any write; hard-stops with sys.exit(4) once the grace
    # period has passed and the counter reaches NO_PROGRESS_THRESHOLD.
    turns_since_last_write = 0
    # Last 5 tool call names, for the no-progress abort diagnostic.
    recent_tools = collections.deque(maxlen=5)
    # Cap compaction-driven no-progress resets to prevent infinite
    # read-only loops where every compaction resets the counter and the
    # agent endlessly re-reads the same large file.
    compaction_resets = 0
    # Turn on which the last compaction ran; used to extend the no-progress
    # grace period so the nudge/abort never fire right after a compaction.
    last_compaction_turn = 0
    # Number of timeout hints already injected (capped at 2).
    timeout_hints_sent = 0

    has_made_tool_call = False
    tool_choice_required_failed = False

    while turns < effective_max_turns:
        turns += 1
        turns_since_last_write += 1
        log(f"--- Turn {turns}/{effective_max_turns} ---")

        # Force tool use after text-only redirects to break the
        # "answers without tools" loop where the model returns text-only
        # on every turn without ever calling a tool.
        if text_only_redirects > 0 and not has_made_tool_call and not tool_choice_required_failed:
            effective_tool_choice = "required"
        else:
            effective_tool_choice = "auto"

        signal.signal(signal.SIGALRM, _turn_timeout_handler)
        signal.alarm(TURN_WALL_CLOCK_TIMEOUT)

        try:
            _turn_start = time.monotonic()
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice=effective_tool_choice,
                temperature=0.0,
            )
            _turn_duration_ms = (time.monotonic() - _turn_start) * 1000
        except TurnTimeoutError:
            signal.alarm(0)
            log(f"⚠ Turn {turns} exceeded {TURN_WALL_CLOCK_TIMEOUT}s wall clock timeout")
            consecutive_errors += 1
            # On the 2nd consecutive timeout, compact the message window so the
            # retry sends a smaller request (less likely to stall again).
            if consecutive_errors == 2:
                log("Compacting messages after consecutive timeout...")
                messages = _compact_messages(messages)
            # Switch model to avoid re-hitting a stalled provider
            if MODEL == AGENT_ESCALATION_MODEL:
                MODEL = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
            else:
                MODEL = AGENT_ESCALATION_MODEL
            log(f"Switched model to {MODEL} for retry")
            if consecutive_errors >= max_consecutive_errors:
                log(f"Max consecutive errors ({max_consecutive_errors}) reached, aborting")
                return False, turns
            # Encourage smaller steps so the retry is less likely to stall again.
            if timeout_hints_sent < 2:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "SYSTEM: The last request timed out. Continue in smaller "
                            "steps: make one edit_file call (under 80 lines of new "
                            "text) or one read_file_lines call, then continue."
                        ),
                    }
                )
                timeout_hints_sent += 1
                log(f"Timeout hint injected ({timeout_hints_sent}/2)")
            log("Retrying in 5 seconds...")
            time.sleep(5)
            continue
        except Exception as e:
            signal.alarm(0)
            if effective_tool_choice == "required":
                log(f"tool_choice='required' not supported ({e}), falling back to 'auto'")
                tool_choice_required_failed = True
                continue
            log(f"API call failed: {e}")
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                log(f"Too many consecutive API errors ({consecutive_errors}), aborting")
                return False, turns
            log("Retrying in 5 seconds...")
            time.sleep(5)
            continue
        finally:
            signal.alarm(0)  # always cancel the alarm after the API attempt

        consecutive_errors = 0
        choice = response.choices[0]

        # Record per-call LLM usage (issue #980) — tokens + timing only,
        # never message content. Uses whichever model actually served this
        # turn (escalation can swap the model mid-run).
        print_llm_usage(
            model=MODEL,
            turn=turns,
            usage=getattr(response, "usage", None),
            duration_ms=_turn_duration_ms,
            finish_reason=getattr(choice, "finish_reason", None),
        )

        # If the model produced a text response with no tool calls, it's done
        if choice.finish_reason == "stop" or not choice.message.tool_calls:
            # Git state check: the agent may have changed files via bash
            # (cp + git commit) instead of edit_file/write_file.
            repo_changed = _repo_has_agent_changes()
            # Redirect text-only responses when no edits have been made yet
            if not (has_attempted_edit or repo_changed) and text_only_redirects < max_text_only_redirects:
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
            # If the model never made any edits, return failure (not success)
            # so the workflow correctly reports BLOCKED instead of masking
            # the no-op as a successful run.
            if not (has_attempted_edit or repo_changed):
                log("Agent returned text-only without making any edits (all redirects exhausted)")
                return False, turns
            if not has_attempted_edit:
                log("Agent finished with changes made outside edit_file/write_file (bash); treating as success")
            if choice.message.content:
                log(f"Agent finished: {choice.message.content[:200]}")
            else:
                log("Agent finished (no final message)")
            messages.append(choice.message.model_dump())
            return True, turns

        # Process tool calls
        messages.append(choice.message.model_dump())
        has_made_tool_call = True

        for tool_call in choice.message.tool_calls:
            fn_name = tool_call.function.name
            recent_tools.append(fn_name)
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
                turns_since_last_write = 0
                compaction_resets = 0

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

            # Import/mock error-loop detection: check point runs after each
            # tool result, before appending to messages.
            err_sig = _extract_error_signature(result["output"])
            if err_sig:
                error_signature_counts[err_sig] = error_signature_counts.get(err_sig, 0) + 1
                error_signature_turns.append((turns, err_sig))
                count = error_signature_counts[err_sig]

                # Pivot trigger: 3 occurrences of the same signature within a
                # rolling 6-turn window -> inject a system-role redirect
                # (fires exactly once per signature).
                if count >= 3 and err_sig not in pivoted_error_signatures:
                    recent_turns = [t for t, s in error_signature_turns if s == err_sig]
                    if len(recent_turns) >= 3 and (turns - recent_turns[-3]) <= 6:
                        pivoted_error_signatures.add(err_sig)
                        log(
                            f"WARNING: Repeated error signature {err_sig} "
                            f"({count} occurrences in 6 turns). Injecting pivot redirect."
                        )
                        messages.append({
                            "role": "system",
                            "content": (
                                "SYSTEM: You keep hitting the same import/mock error "
                                f"({err_sig}).\n\n"
                                "- If it's a missing third-party package, install it via "
                                "pip install <package>. Check for requirements-test.txt or "
                                "requirements.txt first.\n"
                                "- If the module is from the same repo but can't be imported "
                                "(missing __init__.py), create the file, commit it, and end "
                                "your turn.\n"
                                "- Alternatively, commit a SKIP.md explaining why this "
                                "cannot be done and end your turn."
                            ),
                        })

                # Hard-stop trigger: 5 occurrences of the same signature within
                # a rolling 10-turn window -> exit with a distinct code so the
                # workflow surfaces ROOT_CAUSE: repeated_error_loop:<signature>.
                # Does NOT consume remaining max_turns.
                if count >= 5:
                    recent_turns = [t for t, s in error_signature_turns if s == err_sig]
                    if len(recent_turns) >= 5 and (turns - recent_turns[-5]) <= 10:
                        log(
                            f"Repeated error loop detected for signature {err_sig} "
                            f"({count} occurrences in 10 turns), aborting"
                        )
                        sys.exit(3)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result["output"],
                }
            )

        # Stuck-loop detection: check for 5+ repeated identical failing tool calls
        if len(repeated_fail_sigs) >= 5 and len(set(repeated_fail_sigs[-5:])) == 1:
            log(
                "WARNING: Agent repeated the same failing tool call "
                "5 times consecutively"
            )
            if stuck_redirects >= 2:
                if MODEL != AGENT_ESCALATION_MODEL:
                    log(
                        f"Escalating from {MODEL} to {AGENT_ESCALATION_MODEL} "
                        f"after stuck loop"
                    )
                    MODEL = AGENT_ESCALATION_MODEL
                    turns_since_last_write = 0
                    repeated_fail_sigs = []
                    repeated_success_sigs = []
                    stuck_redirects = 0
                    stuck_success_redirects = 0
                    continue  # back to top of while loop, now using pro model
                else:
                    log(
                        f"Already escalated to {AGENT_ESCALATION_MODEL} -- "
                        f"stuck loop persists, aborting"
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

        # Stuck-loop detection: check for 5+ repeated identical successful tool
        # calls with no intervening edit_file/write_file (a read loop that never
        # writes).
        if (
            len(repeated_success_sigs) >= 5
            and len(set(repeated_success_sigs[-5:])) == 1
            and not wrote_since_success_reset
        ):
            log(
                "WARNING: Agent repeated the same successful tool call "
                "5 times consecutively with no "
                "edit_file/write_file (success repetition loop)"
            )
            if stuck_success_redirects >= 2:
                if MODEL != AGENT_ESCALATION_MODEL:
                    log(
                        f"Escalating from {MODEL} to {AGENT_ESCALATION_MODEL} "
                        f"after stuck loop"
                    )
                    MODEL = AGENT_ESCALATION_MODEL
                    turns_since_last_write = 0
                    repeated_fail_sigs = []
                    repeated_success_sigs = []
                    stuck_redirects = 0
                    stuck_success_redirects = 0
                    continue  # back to top of while loop, now using pro model
                else:
                    log(
                        f"Already escalated to {AGENT_ESCALATION_MODEL} -- "
                        f"stuck loop persists, aborting"
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

        # Extend the no-progress grace period after a compaction so the nudge
        # and abort never fire within NO_PROGRESS_GRACE_TURNS turns of one.
        # The window is anchored to the compaction turn itself (rather than
        # computed from the remaining turns), so the checks stay silent for
        # the full NO_PROGRESS_GRACE_TURNS turns after a mid-run compaction.
        effective_grace = _no_progress_effective_grace(
            last_compaction_turn, NO_PROGRESS_GRACE_TURNS
        )

        # No-progress nudge: one-time warning when the agent has stalled for
        # exactly NO_PROGRESS_NUDGE_AT turns after the grace period. Equality
        # test so it fires at most once per stall; the counter resets on any
        # write and on escalation, so the nudge re-arms naturally.
        if (
            turns > effective_grace
            and turns_since_last_write == NO_PROGRESS_NUDGE_AT
        ):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"SYSTEM: You have gone {turns_since_last_write} turns "
                        "without writing a file. Stop reading. Write your first "
                        "change now with edit_file or write_file, even if partial, "
                        "then continue in small steps. If you cannot finish the "
                        "whole task, commit what you have and end your turn."
                    ),
                }
            )
            log(f"No-progress nudge injected at turn {turns}")

        # No-progress detection: hard-stop when the agent burns too many turns
        # without any edit_file/write_file call. Fires only after the grace
        # period so early exploration turns are not penalized. Exit code 4 is
        # distinct from exit code 3 (error-loop) so the workflow can surface
        # ROOT_CAUSE: no_progress.
        if (
            turns > effective_grace
            and turns_since_last_write >= NO_PROGRESS_THRESHOLD
        ):
            if MODEL != AGENT_ESCALATION_MODEL:
                log(
                    f"Escalating from {MODEL} to {AGENT_ESCALATION_MODEL} "
                    f"after {turns_since_last_write} turns without progress"
                )
                MODEL = AGENT_ESCALATION_MODEL
                turns_since_last_write = 0
                # Clear stuck-loop detectors so old flash failures do not
                # cause premature pro aborts
                repeated_fail_sigs = []
                repeated_success_sigs = []
                stuck_redirects = 0
                stuck_success_redirects = 0
                continue  # back to top of while loop, now using pro model
            else:
                log(
                    f"Already escalated to {AGENT_ESCALATION_MODEL} -- "
                    f"no progress persists after {turns_since_last_write} turns, aborting"
                )
                log(
                    f"No-progress abort at turn {turns}; "
                    f"last tool calls: {list(recent_tools)}"
                )
                sys.exit(4)

        # Context window management: if messages are getting very large,
        # summarize older tool results to stay within limits
        total_chars = sum(
            len(json.dumps(m)) for m in messages
        )
        if total_chars > CONTEXT_BUDGET_CHARS:
            log(f"Context approaching limit ({total_chars} chars), compacting...")
            messages = _compact_messages(messages)
            last_compaction_turn = turns
            # Reset no-progress counter: compaction drops tool results and forces the
            # agent to re-read files it had already read. Those recovery turns are
            # legitimate work, not agent inactivity, and must not count toward the
            # no-progress hard-stop. However, unlimited resets enable infinite
            # read-only loops; after 2 compaction-driven resets without a file write,
            # stop resetting so the no-progress detector can fire.
            if compaction_resets < 2:
                turns_since_last_write = 0
                compaction_resets += 1
                log(f"Compaction reset counter {compaction_resets}/2")
            else:
                log("WARNING: Compaction limit reached - no-progress counter will not be reset again")
            # Also clear the success-repetition-loop tracker. Compaction drops
            # tool results and forces the agent to re-read files it had already
            # read; those recovery re-reads are legitimate work and must not be
            # misclassified as a "success repetition loop" that triggers a
            # spurious BLOCKED abort.
            repeated_success_sigs = []
            wrote_since_success_reset = False
            stuck_success_redirects = 0
            log("No-progress and success-loop trackers reset after compaction")

    log(f"Hit max_turns limit ({effective_max_turns})")
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


def _no_progress_effective_grace(
    last_compaction_turn: int, grace_turns: int
) -> int:
    """Earliest turn on which the no-progress nudge/abort may fire.

    The run loop guards the nudge and abort with ``turns >
    effective_grace``. Compaction drops tool results and forces the
    agent to re-read files it had already read; those recovery turns are
    legitimate work, so the checks must never fire within ``grace_turns``
    turns of a compaction. Anchoring the window to the compaction turn
    keeps the checks silent from the compaction up to (but excluding)
    ``last_compaction_turn + grace_turns``. With no compaction
    (``last_compaction_turn == 0``) this degenerates to the plain
    startup grace period, preserving the original behavior.
    """
    return last_compaction_turn + grace_turns


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
    drops under COMPACT_TARGET_RATIO * CONTEXT_BUDGET_CHARS (headroom so
    the next few tool results do not immediately re-trigger compaction):

    1. Truncate older tool messages to OLDER_TOOL_LIMIT_PHASE1 chars and
       assistant/user messages to 2000 chars.
    2. Truncate older tool messages to OLDER_TOOL_LIMIT_PHASE2 chars and
       assistant/user messages to 500 chars.
    3. Drop the oldest tool messages entirely, one by one (oldest first).
    4. Truncate assistant/user messages to 200 chars.
    5. Drop the oldest assistant/user messages until the budget is met.

    The most recent PROTECTED_RECENT_TOOL_MESSAGES tool messages and the
    assistant messages that issued them are never truncated or dropped, so
    freshly-read file content survives compaction long enough to be
    reproduced in an edit_file(old_text=...) call.

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
    target = int(CONTEXT_BUDGET_CHARS * COMPACT_TARGET_RATIO)

    # Identify protected messages: the last PROTECTED_RECENT_TOOL_MESSAGES
    # tool messages plus the assistant messages that issued them. Tracked by
    # object identity so removals in later phases do not shift indices.
    protected = set()
    tool_positions = [
        (i, m) for i, m in enumerate(compacted) if m.get("role") == "tool"
    ]
    for i, tm in tool_positions[-PROTECTED_RECENT_TOOL_MESSAGES:]:
        protected.add(id(tm))
        tool_call_id = tm.get("tool_call_id")
        if not tool_call_id:
            continue
        for j in range(i - 1, -1, -1):
            m = compacted[j]
            if m.get("role") != "assistant":
                continue
            calls = m.get("tool_calls") or []
            if any(
                (c.get("id") if isinstance(c, dict) else None) == tool_call_id
                for c in calls
            ):
                protected.add(id(m))
                break

    def _under_budget() -> bool:
        return _total_chars(compacted) <= target

    def _finish() -> list[dict]:
        after_chars = _total_chars(compacted)
        reclaimed = before_chars - after_chars
        log(
            f"Compacted messages: {before_chars} -> {after_chars} chars "
            f"({reclaimed} reclaimed)"
        )
        return compacted

    # Phase 1 — Generative truncation: shrink older tool messages to
    # OLDER_TOOL_LIMIT_PHASE1 chars and assistant/user messages to 2000 chars.
    for msg in compacted[1:]:
        if id(msg) in protected:
            continue
        if msg.get("role") == "tool":
            _truncate_content(msg, OLDER_TOOL_LIMIT_PHASE1)
        elif msg.get("role") in ("assistant", "user"):
            _truncate_content(msg, 2000)
    if _under_budget():
        return _finish()

    # Phase 2 — Aggressive truncation: shrink older tool messages to
    # OLDER_TOOL_LIMIT_PHASE2 chars and assistant/user messages to 500 chars.
    for msg in compacted[1:]:
        if id(msg) in protected:
            continue
        if msg.get("role") == "tool":
            _truncate_content(msg, OLDER_TOOL_LIMIT_PHASE2)
        elif msg.get("role") in ("assistant", "user"):
            _truncate_content(msg, 500)
    if _under_budget():
        return _finish()

    # Phase 3 — Drop the oldest tool messages entirely (oldest first),
    # skipping protected ones.
    for msg in list(compacted[1:]):
        if id(msg) in protected:
            continue
        if msg.get("role") == "tool":
            compacted.remove(msg)
            if _under_budget():
                return _finish()

    # Phase 4 — Truncate assistant/user messages to 200 chars.
    for msg in compacted[1:]:
        if id(msg) in protected:
            continue
        if msg.get("role") in ("assistant", "user"):
            _truncate_content(msg, 200)
    if _under_budget():
        return _finish()

    # Phase 5 — Drop the oldest assistant/user messages until under budget.
    for msg in list(compacted[1:]):
        if id(msg) in protected:
            continue
        if msg.get("role") in ("assistant", "user"):
            compacted.remove(msg)
            if _under_budget():
                return _finish()

    # Exhausted all reclaimable content and still over budget — log distinctly.
    log(
        f"WARNING: Compaction exhausted all reclaimable content but context is "
        f"still over budget: {_total_chars(compacted)} chars "
        f"(threshold {target}). Budget constants need tuning."
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
    log(f"  Model: {os.environ.get('AGENT_MODEL', DEFAULT_MODEL)}")
    log(f"  Escalation model: {os.environ.get('AGENT_ESCALATION_MODEL', DEFAULT_ESCALATION_MODEL)}")
    log(f"  API base: {API_BASE_URL}")
    log(f"  Max turns: {MAX_TURNS}")
    log(f"  Issue: #{ISSUE_NUMBER} — {ISSUE_TITLE}")
    log(f"  Branch: {BRANCH_NAME}")
    log(f"  Mode: {AGENT_MODE}")
    if AGENT_REQUIRED_FILES and AGENT_REQUIRED_FILES != "*":
        log(f"  Required files: {AGENT_REQUIRED_FILES}")

    try:
        success, turns_used = run_agent()
    finally:
        # Publish the per-run usage summary on every exit path — including
        # the error_loop (3) and no_progress (4) sys.exit()s raised inside
        # run_agent — without ever masking the agent's real exit status.
        try:
            print_usage_summary()
        except Exception as e:  # pragma: no cover - defensive
            log(f"Failed to publish LLM usage summary: {e}")

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
