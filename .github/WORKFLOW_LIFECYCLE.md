# DSG Workflow Lifecycle

This document describes the canonical DSG deploy lifecycle system used across all repos in the org.

---

## Canonical Engine Files

The lifecycle system consists of **canonical files** (pushed to every repo) and **repo-specific config files** (unique per repo).

### Canonical files — pushed to every repo, do not edit per-repo

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Canonical CI workflow (managed centrally) |
| `.github/workflows/code-agent.yml` | Canonical Claude agent workflow (managed centrally) |
| `.github/workflows/lifecycle-bootstrap.yml` | Stable event bootstrap — triggers only, no logic (managed centrally) |
| `.github/workflows/issue-deploy-lifecycle.yml` | Deterministic reconciler — all lifecycle decision logic (managed centrally) |
| `.github/workflows/sam-deploy.yml` | Canonical SAM deploy workflow (managed centrally) |
| `.github/workflows/terraform-deploy.yml` | Canonical Terraform deploy workflow (managed centrally) |
| `.github/deploy-lifecycle.core.json` | Canonical set definitions — always overwritten by org rollout (managed centrally) |
| `.github/WORKFLOW_LIFECYCLE.md` | This documentation (managed centrally) |

These files are **managed centrally** from `dsg-repo-initializer` and pushed to all repos via the `Deploy Agent + CI to Org` workflow. **Do not edit them in individual repos** — edits will be overwritten on the next org deploy.

### Repo-specific config — edit these files per repo

| File | Purpose |
|------|---------|
| `.github/deploy-lifecycle.core.json` | Canonical set definitions (overwritten by org rollout — do not edit) |
| `.github/deploy-lifecycle.overlay.json` | Repo-specific path → set overrides (preserved during rollout) |

`deploy-lifecycle.overlay.json` is the **only** lifecycle config file that maintainers should edit. It is seeded blank on first rollout and **never overwritten** by subsequent org deploys — any repo-specific customizations added here are preserved.

`deploy-lifecycle.core.json` is **always overwritten** by the org rollout. It defines the canonical set map shared across all repos.

---

## PR CI Summary / PR Ready Commenting

PR CI summary and PR-ready signaling are handled by:

- `.github/workflows/agent-pr-ci-summary.yml`

This workflow is generated per repo by `dsg-repo-initializer` from the repo's actual workflow inventory.

The rollout logic:
- lists workflow files in `.github/workflows`
- extracts each workflow `name:`
- filters lane-specific workflow names by repo compatibility (for example SAM vs Terraform)
- rewrites `workflow_run.workflows` in `agent-pr-ci-summary.yml`

The generated workflow posts:
- `<!-- AGENT_CI_SUMMARY -->` — summary of current PR checks
- `<!-- AGENT_PR_READY -->` — machine trigger comment when tracked checks for the PR head SHA are terminal

There is no separate supported CI-summary overlay/config file in the current canonical design.

---

## Architecture: Bootstrap + Reconciler

The lifecycle engine is split into two files to eliminate the self-modifying workflow problem:

### `lifecycle-bootstrap.yml` — stable bootstrap

- Contains **only event triggers** (pull_request, push, repository_dispatch, workflow_run, workflow_dispatch)
- Contains **zero lifecycle logic**
- Invokes `issue-deploy-lifecycle.yml` via `workflow_call` on every event
- Almost never needs to change; changes to lifecycle logic do not require updating this file

### `issue-deploy-lifecycle.yml` — deterministic reconciler

- Triggered **only** via `workflow_call` from the bootstrap
- Recomputes correct issue state from **repository facts** on every invocation
- Safe to re-run at any time — idempotent and self-healing
- Does not depend on comment ordering or prior title state

This split ensures that when `issue-deploy-lifecycle.yml` is updated in a PR, the bootstrap that fires on `pull_request.closed` remains the same stable version — eliminating the race condition where a self-modifying workflow file caused missed lifecycle transitions.

---

## Lifecycle Status Flow

Issues progress through the following states as a PR moves through the pipeline:

```
🔵 [AGENT:IN-PROGRESS]   — Agent is implementing the issue
🟢 [AGENT:NEEDS-REVIEW]  — PR opened; awaiting human review
🟣 [AGENT:DEPLOYING]     — PR merged; deploy workflow(s) in progress
⚫ [DEPLOYMENT COMPLETE] — All required deploy sets succeeded
🔴 [DEPLOYMENT BLOCKED]  — A deploy workflow failed
🔴 [AGENT:BLOCKED]       — Agent could not produce a PR
```

**Important:** Agent-created PRs are mechanically linked to their originating issues using GitHub's native closing-keyword behavior in the PR body. This causes the PR to appear in the issue's **Development** section and may auto-close the issue when the PR merges.

**Recovery:** If a deploy workflow succeeds for the same merge SHA after a previous failure, the issue is automatically recovered from 🔴 [DEPLOYMENT BLOCKED] to ⚫ [DEPLOYMENT COMPLETE].

Titles contain **exactly one** prefix at all times. Stacked prefixes are never written. Before applying a new agent status prefix, the workflow strips all existing leading agent status prefixes from the issue title — so re-running an issue cannot leave stale stacked prefixes such as `🔵 [AGENT:IN-PROGRESS] 🟢 [AGENT:NEEDS-REVIEW] Original title`.

---

## Set-Driven Deploy Selection (V3)

Deploy requirements are determined by **canonical sets**, not raw workflow names. The set map is defined in `.github/deploy-lifecycle.core.json` (org-managed) and optionally extended by `.github/deploy-lifecycle.overlay.json` (repo-specific).

### Canonical sets

| Set ID | Workflow(s) |
|--------|------------|
| `sam_deploy` | SAM deploy workflow(s) for this repo |
| `terraform_deploy` | Terraform deploy workflow(s) for this repo |
| `quicksight_deploy` | QuickSight deploy workflow(s) for this repo |

On PR merge, the lifecycle engine:
1. Reads all changed files for the merged PR
2. Evaluates each `path_rules` entry from the merged core + overlay config — a rule matches if **any** changed file matches **any** of its `paths` globs
3. Computes the **union** of `sets` from all matching rules (`required_sets`)
4. If the required set is **empty**: sets ⚫ [DEPLOYMENT COMPLETE] immediately
5. If the required set is **non-empty**: sets 🟣 [AGENT:DEPLOYING] and writes a `DSG-LIFECYCLE` metadata comment with the computed `required_sets`

### Deploy workflow signaling

Deploy workflows (e.g. `sam-deploy.yml`, `terraform-deploy.yml`) explicitly signal lifecycle completion via `repository_dispatch` with event type `lifecycle-deploy-signal`. The payload carries:
- `set_id` — the canonical set being reported
- `conclusion` — `success` or `failure`
- `sha` — the merge/head SHA
- `issue_numbers` — linked issue numbers

The Lifecycle Bootstrap listens for `repository_dispatch: [lifecycle-deploy-signal]` and invokes the reconciler, which updates the issue title accordingly.

### Set reconciliation

On each `lifecycle-deploy-signal` event, the reconciler:
- Marks the signaled set as succeeded or failed
- Checks overall set completion state: `completed`, `blocked`, `pending`
- Sets ⚫ [DEPLOYMENT COMPLETE] when all required sets are in the `completed` state
- Sets 🔴 [DEPLOYMENT BLOCKED] when any set is in the `blocked` state and none are still pending
- A later successful signal for the same SHA can recover a blocked issue to ⚫ [DEPLOYMENT COMPLETE]

---

## Maintainer: Repo-Specific Configuration

The **only** file maintainers should edit in individual repos is `.github/deploy-lifecycle.overlay.json`. This file controls:

- **Which file paths** trigger which canonical deploy sets (via `path_rules[].sets`)
- Additional set definitions beyond what the canonical core provides (via `sets`)

Paths not matched by any rule go straight to ⚫ [DEPLOYMENT COMPLETE].

---

## Architecture: Core + Overlay Config

At runtime, the reconciler merges both config files into a single effective config:

```
effective = {
  sets: { ...core.sets, ...overlay.sets },
  path_rules: [ ...core.path_rules, ...overlay.path_rules ]
}
```

- Overlay `sets` entries take precedence over core `sets` entries with the same key
- `path_rules` from both files are evaluated together (union)
- If `deploy-lifecycle.core.json` is absent, the reconciler falls back to legacy `.github/deploy-lifecycle.json` for backward compatibility

---

## Lifecycle Metadata

On every PR merge where deploy sets are required, the lifecycle engine writes a machine-readable metadata comment to each linked issue:

```
DSG-LIFECYCLE:
repo: <owner>/<repo>
pr: <number>
merge_sha: <sha>
head_sha: <sha>
deploy_mode: changed_paths
required_sets: ["sam_deploy", "terraform_deploy"]
timestamp: <utc>
```

This comment is the source of truth for which sets must complete for that specific PR.

Comments are **audit-only**. Normal user comments never affect lifecycle state. Only the `DSG-LIFECYCLE` metadata comment is read by the reconciler.

---

## Self-Healing

If a lifecycle event is missed (e.g., `pull_request.closed` fires during a workflow file replacement), the system self-heals via:

1. **Push backstop** — every push to main re-reconciles any linked issues not yet handled
2. **workflow_dispatch** — manually trigger `Lifecycle Bootstrap` to scan all open `NEEDS-REVIEW` issues and reconcile each one against actual PR and workflow run facts
3. **repository_dispatch lifecycle-deploy-signal** — each completed deploy workflow re-drives the issue to its correct final state

Missing one event **never** permanently breaks correctness.

---

## Configuration Schema (V3)

### `deploy-lifecycle.core.json` (canonical — do not edit per repo)

```json
{
  "version": 3,
  "sets": {
    "sam_deploy": {
      "workflows": ["Deploy (SAM)"]
    },
    "terraform_deploy": {
      "workflows": ["Terraform Deploy"]
    }
  },
  "path_rules": []
}
```

### `deploy-lifecycle.overlay.json` (repo-specific — edit this file)

```json
{
  "version": 3,
  "sets": {},
  "path_rules": [
    {
      "paths": ["sam/**"],
      "sets": ["sam_deploy"]
    }
  ]
}
```

- `version`: must be `3` for set-driven behavior
- `sets`: map of set ID → `{ workflows: [...] }` — defines which workflow names belong to each set
- `path_rules`: list of rules; each rule matches if any changed file matches any of its `paths` globs
- `path_rules[].paths`: glob patterns (supports `*` within a segment, `**` across segments)
- `path_rules[].sets`: canonical set IDs to require when this rule matches
- Overlay `sets` can override or extend the canonical core sets
- Empty `path_rules` in both files means no deploy sets are required — PRs set DEPLOYMENT COMPLETE immediately

---

## Configuration Examples

### SAM compute repo

`deploy-lifecycle.overlay.json`:
```json
{
  "version": 3,
  "sets": {},
  "path_rules": [
    {
      "paths": ["sam/**"],
      "sets": ["sam_deploy"]
    }
  ]
}
```

### Terraform (infra-only) repo

`deploy-lifecycle.overlay.json`:
```json
{
  "version": 3,
  "sets": {},
  "path_rules": [
    {
      "paths": ["terraform/**", "**/*.tf", "**/*.tfvars"],
      "sets": ["terraform_deploy"]
    }
  ]
}
```

### Multi-lane repo (independent deploy pathways)

Each path rule triggers only its own set. A PR touching only frontend files waits only for the frontend deploy; a PR touching only infra waits only for Terraform. A PR touching both waits for both.

`deploy-lifecycle.overlay.json`:
```json
{
  "version": 3,
  "sets": {
    "frontend_deploy": {
      "workflows": ["Deploy Analytics Frontend (FPC)"]
    }
  },
  "path_rules": [
    {
      "paths": ["frontend/**", "src/analytics/**"],
      "sets": ["frontend_deploy"]
    },
    {
      "paths": ["terraform/**", "**/*.tf"],
      "sets": ["terraform_deploy"]
    },
    {
      "paths": ["sam/**"],
      "sets": ["sam_deploy"]
    }
  ]
}
```

### Library / no-deploy repo

`deploy-lifecycle.overlay.json`:
```json
{
  "version": 3,
  "sets": {},
  "path_rules": []
}
```

All PRs set DEPLOYMENT COMPLETE immediately after merge. No deploy sets are ever required.

## Issue ↔ PR Linking

Agent-created pull requests must be mechanically linked to their originating GitHub issues using a native GitHub closing keyword in the PR body:

```text
Closes #<issue_number>
```

This is the canonical linkage mechanism for the DSG agent workflow:

- It causes the PR to appear in the issue's **Development** section.
- It gives a clear system-level relationship between the issue and the PR.
- The linked issue may auto-close when the PR merges, which is acceptable.
