# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `ISAggregateOptions` and `aggregate_with_options` as a grouped alternative
  to the many keyword-only arguments of `aggregate_individual_structures`.
- Add `threshold_mode` (`"strict"` / `"inclusive"`) to edge voting in
  `aggregate_individual_structures` and expose it in aggregation `metadata`.
- Add Monte Carlo options: `mc_refit_global_structure`, `mc_posterior` (`filtered` /
  `smoothed`), `mc_contributors` (`individual_edge` / `all_subjects`), optional
  `data_per_subject` / `mc_refit_n_jobs`; `GlobalBetaMCResult` now includes
  `beta_mean` and `beta_var` (draw-axis, nan-aware).
- Add `mdmp.model.refit_mdm_on_structure` for MDM-style discount selection,
  filtering, and smoothing on a fixed binary DAG (exported from `mdmp` and
  `mdmp.model`).
- Add `pool_filt_for_plotting` support to build plot-ready aggregated `Filt`
  structures from per-subject filtered outputs.
- Add support for passing fitted MDM-like objects directly to
  `aggregate_individual_structures`, reusing model adjacency/filter outputs and
  deriving mean `plot_data` when needed.
- Add shared plotting input validators in `mdmp.plotting._input_checks`,
  plus test coverage for IS global-beta Monte Carlo and plotting integration.
- Add IS aggregation workflow diagrams in
  `mdmp/group_analysis/is/aggregation_diagrams.md`.
- Move edge voting and greedy cycle repair helpers to
  `mdmp/group_analysis/is/voting.py`.

### Removed

- Remove unused `BaseLearningAlgorithm.compute_score` and `_has_cycle`, and
  `mdmp.scoring.compute_structure_score` (only used by the removed method).

### Changed

- Replace structure-learning algorithm registry with a static `METHODS` map in
  `mdmp.structure.learner`; remove `register_algorithm`, `get_algorithm`, and
  `list_algorithms` from the public API.
- `plot_arcs` now sizes its subplot grid to the number of matching parameters
  (up to four columns) instead of a fixed 2×2 cap at four panels; default
  `figsize` scales with the grid.
- `plot_dag` graph mode uses a layered layout (Graphviz ``dot`` via
  ``pygraphviz`` when available, otherwise topological generations), with
  ``spring_layout`` for cyclic graphs; optional ``hierarchical``,
  ``level_gap``, and styling parameters were added.
- Refactor IS aggregation internals into submodules (`results`, `adj_coercion`,
  `mc_global_beta`, `plot_filt_pool`) for maintainability; public API unchanged.
- Re-export IS aggregation symbols from `mdmp.group_analysis` and top-level
  `mdmp` for easier imports without referencing the `is` keyworded submodule.
- Improve README and notebook examples for IS aggregation, including pooled
  plotting/filter workflows and updated notebook references.
- Update plotting modules to use centralized input checks and clearer error
  messages when required `data`, `Filt`, or `Smoo` components are missing.

### Fixed

- Fix `build_design_matrix` when `adj_mat` uses floating dtypes (parameter counts
  must be integers).

## [0.6.2] - 2026-04-19

### Added

- This changelog and a single-source package version in `mdmp/_version.py`, with
  dynamic metadata in `pyproject.toml` and release automation via `bump-my-version`.

[Unreleased]: https://github.com/maods2/mdmp/compare/v0.6.2...HEAD
[0.6.2]: https://github.com/maods2/mdmp/releases/tag/v0.6.2
