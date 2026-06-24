# Agent Rules

This document describes the agent behavior rules enforced across DSG repositories.

## Claude Agent Issue Lifecycle

All generated repositories use a single operational start label:

- `agent:ready`

The workflow must not introduce additional status labels. Status is represented in the issue title only.

### Supported title states

- `🟡 [AGENT:PLAN-REVIEW]`
- `🔵 [AGENT:IN-PROGRESS]`
- `🟢 [AGENT:NEEDS-REVIEW]`
- `🔴 [AGENT:BLOCKED]`

### Command flow

1. Add `agent:ready` to an open issue.
2. The workflow removes `agent:ready`, generates a plan comment only, and moves the issue to `PLAN-REVIEW`.
3. To request a plan revision, comment exactly:

```text
/agent revise-plan
```

4. To approve the plan and start implementation, comment exactly:

```text
/agent approve-plan
```

5. Implementation must reuse branch `claude/issue-<issue_number>` and update the existing PR when one already exists for that branch.
6. Every agent-created PR must be mechanically linked to its originating issue using a native GitHub closing keyword in the PR body so the PR appears in the issue's **Development** section.
7. The canonical PR body linkage line is:

```text
Closes #<issue_number>
```

8. Every plan comment must begin with `<!-- AGENT_PLAN -->`.
9. Every plan comment must include these exact next-step commands:
   - `/agent approve`
   - `/agent revise`

## Canonical File Contract

Every DSG repository managed by `dsg-repo-initializer` receives the same canonical file set. Both `dsg-init-repo.ps1` (initializer) and `.github/workflows/org-deploy-agent.yml` (org rollout) manage **the exact same files listed below**.

### Canonical files (always overwritten on rollout)

| File | Source in this repo |
|------|---------------------|
| `.github/workflows/code-agent.yml` | `.github/workflows/code-agent.yml` |
| `.github/workflows/ci.yml` | `.github/workflows/ci.yml` |
| `.github/workflows/lifecycle-bootstrap.yml` | `.github/workflows/lifecycle-bootstrap.yml` |
| `.github/workflows/issue-deploy-lifecycle.yml` | `.github/workflows/issue-deploy-lifecycle.yml` |
| `.github/workflows/sam-deploy.yml` | `.github/workflows/sam-deploy.yml` |
| `.github/workflows/terraform-deploy.yml` | `.github/workflows/terraform-deploy.yml` |
| `.github/workflows/validate-pr-no-ci-skip.yml` | `.github/workflows/validate-pr-no-ci-skip.yml` |
| `.github/WORKFLOW_LIFECYCLE.md` | `.github/WORKFLOW_LIFECYCLE.md` |
| `.github/deploy-lifecycle.core.json` | `.github/deploy-lifecycle.core.json` |
| `CLAUDE.md` | `canonical/CLAUDE.md` |
| `.claude/settings.json` | `canonical/claude-settings.json` |
| `.devcontainer/devcontainer.json` | `.devcontainer/devcontainer.json` |
| `.amazonq/rules/commit-message-policy.md` | `.amazonq/rules/commit-message-policy.md` |
| `docs/agent-rules.md` | `docs/agent-rules.md` |

### Preserved repo-specific file (never overwritten)

| File | Behavior |
|------|----------|
| `.github/deploy-lifecycle.overlay.json` | Seeded with a blank template if missing; **never overwritten** if present. Per-repo deploy customizations live here. |

### PR CI summary generation

- `.github/workflows/agent-pr-ci-summary.yml` is generated from the repo's actual workflow inventory.
- The org rollout auto-detects workflow names from `.github/workflows/` and filters lane-specific entries by repo compatibility.
- There is no separate supported CI-summary overlay/config file in the current canonical design.

### Key constraints

- Org rollout (`.github/workflows/org-deploy-agent.yml`) is **manual only** (`workflow_dispatch`). No push, pull_request, workflow_run, or repository_dispatch triggers.
- No repo-specific variants of governance files (CLAUDE.md, .claude/settings.json, .devcontainer/devcontainer.json, docs/agent-rules.md, workflow files, lifecycle core config).
- Canonical content for `CLAUDE.md` and `.claude/settings.json` lives in `canonical/` to keep this repo's own governance files separate from what gets deployed to target repos.

## Amazon Q Project Rules

Amazon Q Developer project rules are stored under `.amazonq/rules/` in each repository. These files guide Amazon Q's behavior when generating code, commits, and pull requests.

### Commit Message Policy

**File:** `.amazonq/rules/commit-message-policy.md`

Amazon Q must never include CI-skip markers in commit messages. The following strings are forbidden:

- `[skip ci]`
- `[ci skip]`
- `[no ci]`
- `[skip actions]`
- `skip-checks: true`

**Why this rule exists:** CI-skip markers suppress GitHub Actions workflows on push to main. In deployment-managed repositories, this silently breaks the automated deploy chain — an issue moves to DEPLOYING but no deploy workflows ever run, leaving it stranded indefinitely.

This rule applies to all commit creation scenarios: issue implementation commits, pull request follow-up commits, commits generated from `/q` instructions, and automated PR updates.

## Temporary Operating Rule: Do Not Use Amazon Q for PR Follow-Up Commits

**Status:** In effect until further notice.

Amazon Q can still emit CI-skip markers in commit headers even after repository-level `.amazonq` rules are in place. Until this is resolved, Amazon Q must not be used for PR follow-up commits.

**For PR revisions, use the issue-first workflow:**

1. Update the linked GitHub issue with the requested changes.
2. If the plan needs to change, comment `/agent revise-plan`.
3. If the plan is approved and implementation should proceed, comment `/agent approve-plan`.
4. Let Claude agent update the existing branch/PR.

The GitHub issue is the source of truth for requested changes. Do not use Amazon Q's `/q` command or GitHub Copilot-style suggestions to push follow-up commits to open PRs.

## Defense in Depth

The CI-skip prohibition is enforced at multiple levels:

1. **Amazon Q project rule** (`.amazonq/rules/commit-message-policy.md`) — prevents Q from generating forbidden commit messages.
2. **Claude agent workflow** — agent-side validation blocks CI-skip markers before push.
3. **PR validation workflow** (`.github/workflows/validate-pr-no-ci-skip.yml`) — fails any pull request containing CI-skip markers in the title, body, or commit history.
4. **CLAUDE.md guidance** — CI-skip markers are explicitly forbidden in the agent's operating instructions.

All three layers are propagated to downstream repositories by `dsg-repo-initializer`.
