# DSG Coding Agent

A composite GitHub Action that runs an LLM-powered coding agent inside a workflow. It reads issue context and an approved plan, then iterates through a ReAct-style tool-call loop — exploring the repository, editing files, running commands, and committing changes locally. The agent defaults to DeepSeek (`deepseek-chat` via `https://api.deepseek.com`) but accepts any OpenAI-compatible API provider.

## Inputs

| Input | Description | Default | Required |
|-------|-------------|---------|----------|
| `api-key` | API key for the LLM provider (DeepSeek, OpenAI, etc). | — | yes |
| `issue-context` | Path to the issue context JSON file (`.agent/issue-context.json`). | — | yes |
| `branch-name` | Git branch the agent is working on. | — | yes |
| `issue-number` | GitHub issue number for commit messages. | — | yes |
| `issue-title` | GitHub issue title for commit messages. | `""` | no |
| `max-turns` | Maximum number of agentic loop iterations. | `"40"` | no |
| `model` | Model identifier (e.g. deepseek-chat, gpt-4o). | `"deepseek-chat"` | no |
| `api-base-url` | Base URL for the OpenAI-compatible API. | `"https://api.deepseek.com"` | no |

## Outputs

| Output | Description |
|--------|-------------|
| `outcome` | `success` if the agent completed normally, `failure` if it hit max_turns or errored. |
| `turns_used` | Number of agentic loop turns consumed. |

## Tools available to the agent

- `read_file` — Read the contents of a file. Returns the full text for files under 32000 characters; larger files are truncated with a note.
- `write_file` — Write content to a file, creating it (and any parent directories) if it doesn't exist, or overwriting it if it does.
- `edit_file` — Replace a specific string in a file with new content. The old_text must appear exactly once in the file. Use this for targeted edits instead of rewriting the entire file.
- `bash` — Run a shell command and return stdout and stderr. Timeout: 120s. Use for installing dependencies, running tests, checking file structure, git operations, etc. Do NOT use for git push.
- `list_files` — List files matching a glob pattern relative to the repo root. Returns one path per line.
- `grep_search` — Search file contents using grep. Returns matching lines with file paths and line numbers.

## Provider configuration

By default the agent uses DeepSeek (`api-base-url: "https://api.deepseek.com"`, `model: "deepseek-chat"`). You can switch to any OpenAI-compatible API by overriding the `api-base-url` and `model` inputs.

## Usage example

**Default (DeepSeek):**

```yaml
- name: Run coding agent
  uses: ./.github/actions/coding-agent
  with:
    api-key: ${{ secrets.DEEPSEEK_API_KEY }}
    issue-context: .agent/issue-context.json
    branch-name: ${{ github.head_ref }}
    issue-number: ${{ github.event.number }}
```

**Override to another provider:**

```yaml
- name: Run coding agent
  uses: ./.github/actions/coding-agent
  with:
    api-key: ${{ secrets.OPENAI_API_KEY }}
    issue-context: .agent/issue-context.json
    branch-name: ${{ github.head_ref }}
    issue-number: ${{ github.event.number }}
    api-base-url: "https://api.openai.com/v1"
    model: "gpt-4o"
```
