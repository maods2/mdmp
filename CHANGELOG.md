# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add Monte Carlo global edge-coefficient pooling in
  `aggregate_individual_structures` via `global_beta_mc`, with support for
  multi-time indices, configurable pooling, and optional posterior quantiles.
- Add `pool_filt_for_plotting` support to build plot-ready aggregated `Filt`
  structures from per-subject filtered outputs.
- Add support for passing fitted MDM-like objects directly to
  `aggregate_individual_structures`, reusing model adjacency/filter outputs and
  deriving mean `plot_data` when needed.
- Add shared plotting input validators in `mdmp.plotting._input_checks`,
  plus test coverage for IS global-beta Monte Carlo and plotting integration.
- Add IS aggregation workflow diagrams in
  `mdmp/group_analysis/is/aggregation_diagrams.md`.

### Changed

- Re-export IS aggregation symbols from `mdmp.group_analysis` and top-level
  `mdmp` for easier imports without referencing the `is` keyworded submodule.
- Improve README and notebook examples for IS aggregation, including pooled
  plotting/filter workflows and updated notebook references.
- Update plotting modules to use centralized input checks and clearer error
  messages when required `data`, `Filt`, or `Smoo` components are missing.

### Fixed

## [0.6.2] - 2026-04-19

### Added

- This changelog and a single-source package version in `mdmp/_version.py`, with
  dynamic metadata in `pyproject.toml` and release automation via `bump-my-version`.

[Unreleased]: https://github.com/maods2/mdmp/compare/v0.6.2...HEAD
[0.6.2]: https://github.com/maods2/mdmp/releases/tag/v0.6.2
