# CLAUDE.md — DSG Claude Code Executor Rules

## Operating mode
- Make deterministic, minimal-diff edits.
- Prefer small, surgical changes over refactors.
- Do not deploy directly to AWS. Only edit files, commit, and push.

## Output format
- When asked to generate code changes, output patch-style changes or clearly scoped file edits.
- Avoid rewriting files unless necessary.

## Git workflow
- After completing requested repo changes:
  - `git status`
  - `git add -A`
  - `git commit -m "<clear message>"`
  - `git push origin main`

## Issue and PR linkage
- When work originates from a GitHub issue, the resulting PR must be mechanically linked to that issue using a native GitHub closing keyword in the PR body.
- Use this canonical format:

```text
Closes #<issue_number>
```

## Safety
- Never run destructive cloud actions (delete/terminate/destroy) unless explicitly instructed.
- Prefer CI/CD via GitHub Actions for validation and deployment.

## CI-skip markers — NEVER USE (non-negotiable)

**NEVER** include `[skip ci]`, `[ci skip]`, `[no ci]`, `[skip actions]`, or `skip-checks: true` in:
- Commit messages
- PR titles or bodies
- Any text that could end up in a merge commit message

These markers suppress GitHub Actions workflows on push to main, **silently breaking the automated deploy chain** for deployment-managed repos. An issue moves to DEPLOYING but no deploy workflows ever run, leaving it stranded indefinitely.

This rule is enforced at two levels: (1) agent-side validation in the code-agent workflow blocks CI-skip markers before push, and (2) repo-side PR validation workflow fails any pull request containing these markers in the title, body, or commit history.
