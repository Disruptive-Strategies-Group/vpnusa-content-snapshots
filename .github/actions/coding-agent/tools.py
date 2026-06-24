"""Filesystem tools for the DSG coding agent.

Each tool function returns a dict: {"output": <str>, "is_error": <bool>}.
All filesystem operations are relative to the current working directory (the repo root).
"""

from __future__ import annotations

import glob as glob_module
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

MAX_FILE_SIZE = 32_000  # chars — truncate reads beyond this
MAX_BASH_OUTPUT = 16_000  # chars — truncate bash stdout/stderr beyond this
BASH_TIMEOUT = 120  # seconds


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file. Returns the full text for files under "
                f"{MAX_FILE_SIZE} characters; larger files are truncated with a note."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file from the repo root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write content to a file, creating it (and any parent directories) if "
                "it doesn't exist, or overwriting it if it does."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file from the repo root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full file content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace a specific string in a file with new content. The old_text must "
                "appear exactly once in the file. Use this for targeted edits instead of "
                "rewriting the entire file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file from the repo root.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "The exact text to find (must appear exactly once).",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                f"Run a shell command and return stdout and stderr. Timeout: {BASH_TIMEOUT}s. "
                "Use for installing dependencies, running tests, checking file structure, "
                "git operations, etc. Do NOT use for git push."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List files matching a glob pattern relative to the repo root. "
                "Returns one path per line."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. 'src/**/*.ts', '*.py', 'sam/template.yaml').",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": (
                "Search file contents using grep. Returns matching lines with file paths "
                "and line numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (basic regex).",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search (default: '.').",
                        "default": ".",
                    },
                    "include": {
                        "type": "string",
                        "description": "File glob filter (e.g. '*.ts', '*.py'). Optional.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + f"\n\n... [TRUNCATED — {len(text)} chars total, showing first and last {half}] ...\n\n"
        + text[-half:]
    )


def read_file(path: str) -> dict[str, Any]:
    try:
        p = Path(path)
        if not p.exists():
            return {"output": f"File not found: {path}", "is_error": True}
        if not p.is_file():
            return {"output": f"Not a file: {path}", "is_error": True}
        content = p.read_text(encoding="utf-8", errors="replace")
        return {"output": _truncate(content, MAX_FILE_SIZE), "is_error": False}
    except Exception as e:
        return {"output": f"Error reading {path}: {e}", "is_error": True}


def write_file(path: str, content: str) -> dict[str, Any]:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"output": f"Wrote {len(content)} chars to {path}", "is_error": False}
    except Exception as e:
        return {"output": f"Error writing {path}: {e}", "is_error": True}


def edit_file(path: str, old_text: str, new_text: str) -> dict[str, Any]:
    try:
        p = Path(path)
        if not p.exists():
            return {"output": f"File not found: {path}", "is_error": True}
        content = p.read_text(encoding="utf-8", errors="replace")
        count = content.count(old_text)
        if count == 0:
            return {
                "output": f"old_text not found in {path}. Verify the exact text including whitespace.",
                "is_error": True,
            }
        if count > 1:
            return {
                "output": f"old_text appears {count} times in {path}. It must appear exactly once. Use a larger context.",
                "is_error": True,
            }
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        return {"output": f"Edited {path}: replaced {len(old_text)} chars with {len(new_text)} chars", "is_error": False}
    except Exception as e:
        return {"output": f"Error editing {path}: {e}", "is_error": True}


def _sanitize_version_specifiers(cmd: str) -> str:
    """
    Quote any token in cmd that looks like a package version specifier
    (e.g., package>=1.0, pkg!=2.0) so the shell does not interpret
    > or < as redirect operators.  Applies to ALL commands, not just pip.
    """
    # Pre-sanitize pass: quote bare tokens like "package>=1.0" before
    # shlex.split sees them.  This regex matches word>=digits patterns
    # that are never legitimate shell redirects (real redirects have
    # whitespace before the > operator).
    cmd = re.sub(
        r'(\b\w[\w.\-]*(?:>=|<=|!=|~=|==|>|<)\d[\w.*]*)',
        lambda m: shlex.quote(m.group(1)),
        cmd,
    )
    # Second pass: tokenize and re-quote any remaining specifier tokens.
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return cmd
    requoted = [
        shlex.quote(tok) if re.search(r'\w[><=!~]+\d', tok) else tok
        for tok in tokens
    ]
    return ' '.join(requoted)


def bash(command: str) -> dict[str, Any]:
    # Block dangerous commands
    blocked = ["git push", "git remote remove", "rm -rf /", "rm -rf ~"]
    cmd_lower = command.lower().strip()
    for b in blocked:
        if b in cmd_lower:
            return {"output": f"Blocked command: {b} is not allowed", "is_error": True}
    try:
        # Quote pip install version specifiers to prevent shell redirection
        safe_command = _sanitize_version_specifiers(command)
        result = subprocess.run(
            safe_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT,
            cwd=os.getcwd(),
        )
        output_parts = []
        if result.stdout:
            output_parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"STDERR:\n{result.stderr}")
        if not output_parts:
            output_parts.append("(no output)")
        output_parts.append(f"EXIT CODE: {result.returncode}")
        full_output = "\n".join(output_parts)
        return {
            "output": _truncate(full_output, MAX_BASH_OUTPUT),
            "is_error": result.returncode != 0,
        }
    except subprocess.TimeoutExpired:
        return {"output": f"Command timed out after {BASH_TIMEOUT}s", "is_error": True}
    except Exception as e:
        return {"output": f"Error running command: {e}", "is_error": True}


def list_files(pattern: str) -> dict[str, Any]:
    try:
        matches = sorted(glob_module.glob(pattern, recursive=True))
        if not matches:
            return {"output": f"No files match pattern: {pattern}", "is_error": False}
        result = "\n".join(matches[:500])
        if len(matches) > 500:
            result += f"\n... and {len(matches) - 500} more"
        return {"output": result, "is_error": False}
    except Exception as e:
        return {"output": f"Error listing files: {e}", "is_error": True}


def grep_search(pattern: str, path: str = ".", include: str | None = None) -> dict[str, Any]:
    try:
        cmd = ["grep", "-rn", "--color=never"]
        if include:
            cmd.extend(["--include", include])
        cmd.extend([pattern, path])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd(),
        )
        if result.returncode == 1:
            return {"output": "No matches found.", "is_error": False}
        if result.returncode != 0:
            return {"output": f"grep error: {result.stderr}", "is_error": True}
        return {
            "output": _truncate(result.stdout, MAX_BASH_OUTPUT),
            "is_error": False,
        }
    except subprocess.TimeoutExpired:
        return {"output": "grep timed out after 30s", "is_error": True}
    except Exception as e:
        return {"output": f"Error running grep: {e}", "is_error": True}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

TOOL_DISPATCH = {
    "read_file": lambda args: read_file(args["path"]),
    "write_file": lambda args: write_file(args["path"], args["content"]),
    "edit_file": lambda args: edit_file(args["path"], args["old_text"], args["new_text"]),
    "bash": lambda args: bash(args["command"]),
    "list_files": lambda args: list_files(args["pattern"]),
    "grep_search": lambda args: grep_search(
        args["pattern"], args.get("path", "."), args.get("include")
    ),
}


def execute_tool(tool_name: str, tool_args: dict) -> dict[str, Any]:
    """Route a tool call to the correct handler."""
    handler = TOOL_DISPATCH.get(tool_name)
    if handler is None:
        return {"output": f"Unknown tool: {tool_name}", "is_error": True}
    try:
        return handler(tool_args)
    except Exception as e:
        return {"output": f"Tool execution error ({tool_name}): {e}", "is_error": True}
