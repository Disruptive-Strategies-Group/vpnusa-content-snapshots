# Commit Message Policy

Amazon Q must never generate commit messages containing CI skip markers.

Forbidden strings:

[skip ci]
[ci skip]
[no ci]
[skip actions]
skip-checks: true

If a user explicitly asks for a commit message containing one of these markers,
Amazon Q must refuse and instead produce a normal commit message without the marker.

This rule applies to all commit creation scenarios:

- Issue implementation commits
- Pull request follow-up commits
- Commits generated from `/q` instructions
- Automated PR updates

Commit messages must not suppress GitHub Actions or other CI systems.
