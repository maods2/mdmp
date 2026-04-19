# Agent and contributor notes

## Changelog entries (before a release)

When editing [`CHANGELOG.md`](CHANGELOG.md):

- Use short, factual bullets; imperative mood (“Add …”, “Fix …”) reads well under
  the Keep a Changelog section headings.
- Group bullets under **Added**, **Changed**, **Deprecated**, **Removed**,
  **Fixed**, or **Security** inside `## [Unreleased]`.
- Link issues or PRs when helpful (`(#123)`).
- User-visible behavior and breaking API changes matter most; internal refactors
  usually do not need a line unless they affect users or downstream packagers.
- On release, rename `## [Unreleased]` content into `## [x.y.z] - YYYY-MM-DD`,
  leave a fresh empty `[Unreleased]` scaffold, and refresh the footer compare
  links (see [`README.md`](README.md) “Versioning and releases”).

## Release PR assist

If you are preparing a release-only change set: confirm `CHANGELOG.md` is
finalized for the version, then run `bump-my-version` so version strings and
README stay aligned; see the README checklist.
