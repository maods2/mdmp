# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add notebook `09-gs-clusters-then-vts-is.ipynb`: fit MDM per subject, cluster with
  group-structure distance, then compute VTS and IS separately per cluster on
  retail demo data.
- `plot_dag(..., style="graphviz")`: Graphviz ``dot`` rendering with circular
  filled nodes and curved edge routing (requires optional `pydot` + Graphviz
  ``dot`` binary; `pip install mdmp[graphviz]`).
- `aggregate_individual_structures`: `mc_n_jobs` to parallelize Monte Carlo over filter
  time steps (`None` or `1` = serial, `-1` = all cores).
- Add `mdmp.group_analysis.inds` subpackage (Individual Structure aggregation).
- Add split entry points: `vote_individual_structures`, `refit_on_consensus`,
  `run_inds_global_beta_mc`, `pool_conditional_filtered_states`, `as_inds_mdm_view`.
- Add `ISVoteOptions`, `ISMonteCarloOptions`, `ISMDMViewOptions`, and
  `merge_aggregate_options` for grouped configuration.
- Add `inds.pipeline` orchestration module and split `vote_edge_frequencies` /
  `repair_dag_to_acyclic` in `inds.voting`.
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
  deriving mean `time_series` when needed.
- Add shared plotting input validators in `mdmp.plotting._input_checks`,
  plus test coverage for IS global-beta Monte Carlo and plotting integration.
- Add IS aggregation workflow diagrams in
  `mdmp/group_analysis/inds/aggregation_diagrams.md`.
- Move edge voting and greedy cycle repair helpers to
  `mdmp/group_analysis/inds/voting.py`.

### Removed

- Remove the `simulation/` comparison tree and `notebooks/03-mdmr.ipynb` (R `mdmr` demo).
- Remove split IS entry points: `vote_individual_structures`,
  `run_inds_global_beta_mc`, `pool_conditional_filtered_states`, `as_inds_mdm_view`,
  `aggregate_with_options`, and `refit_on_consensus` from the public API.
  Use `aggregate_individual_structures` only.
- Remove unused `BaseLearningAlgorithm.compute_score` and `_has_cycle`, and
  `mdmp.scoring.compute_structure_score` (only used by the removed method).

### Changed

- Set package version to `0.1.0` (intentional reset from `0.6.2` for the
  public/docs refresh).
- Move Jupyter notebooks, `retail_helpers.py`, and retail CSV under
  `examples/notebooks/`; refresh `examples/*.py` demos (including new IS/GS
  scripts) and update README / examples README links.
- Document MDMP as a new Python MDM implementation (not an R `mdmr` port) in
  the README and example notebooks; restyle notebook intros to a topic-first
  pattern; lead README examples with the retail case study.
- Expand README Features with IS, GS, and anomaly detection; move development
  setup, tests, linting, and release checklist to [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Rewrite `04-is-vs-vts-multi-individual.ipynb` to use retail demo data.
- Optional `title=` on `plot_dag` / `plot_stream` / `plot_marginal` (pass
  `None` to omit the axes title; default text unchanged for existing callers).
- When `MDM(..., verbose=False)` (and other structure learners with
  `verbose=False`), raise the `pgmpy` logger to WARNING during structure
  learning and default `show_progress=False` so INFO messages such as
  datatype inference are not printed.
- `aggregate_individual_structures`: `mc_refit_global_structure=None` (default) refits
  on the consensus DAG when inputs are MDM-like; pass `False` for the previous
  individual-DAG filtered posterior path.
- Global-beta Monte Carlo always runs at every filter time step ``0 … T-1``;
  removed `time_index` / `time_indices` from the public aggregate API
  (`beta_samples` shape ``(mc_n_samples, n_edges, T)``).
- Rename ``GlobalBetaMCResult.beta_draws`` to ``beta_samples`` (aligned with
  ``mc_n_samples`` and internal ``_sample_*`` helpers).
- Monte Carlo uses only population-mean pooling
  (:math:`\\bar\\theta_t^{(b)} = \\frac{1}{S}\\sum_i \\theta_{it}^{(b)}` per
  replicate); removed ``pooling``, ``mc_contributors``, and conditional
  ``mean_with_edge`` / ``sum_with_edge`` modes.
- Vote stage builds `ISAggregatedMDMView` directly; the aggregate path no longer
  converts consensus → view mid-pipeline.
- Rename internal ``filtered_per_subject`` to ``posterior_per_subject`` (and
  ``resolved_filtered_*`` to ``resolved_posterior_*``) across IS coercion, MC, and
  plot-pooling helpers.
- Speed up global-beta Monte Carlo: vectorized replicate sampling, edge coefficient
  index precomputation, and optional parallelism over time.
- Simplify `aggregate_individual_structures`: inputs are adjacency matrices/DataFrames
  or fitted MDMs only; strict edge voting is fixed; MDM inputs auto-run Monte Carlo
  (`mc_n_samples` default 500) and auto-build pooled `Filt` for `plot_arcs`; removed
  `threshold_mode`, `time_series`, `plot_filt` /
  `plot_smoo` / `plot_df`, `pool_filt_for_plotting`, `posterior_per_subject`, and
  `data_per_subject` from the public aggregate API (split helpers retain MC/plot knobs).
- Individual Structure implementation package renamed from `mdmp.group_analysis.is`
  to `mdmp.group_analysis.inds` (the `is` import path is removed).
- Public IS API lives in `inds.pipeline` (`aggregation.py` removed).
- Rename Monte Carlo keyword `n_draws` to `mc_n_samples` on aggregation and
  `ISAggregateOptions` / `ISMonteCarloOptions` (number of posterior samples per edge).
- Rename aggregation keyword `plot_data` to `time_series` (multivariate ``(T, N)``
  series on the consensus view, stored as ``ISAggregatedMDMView.data``).
- Clarify internal naming: post-coercion values use ``resolved_*`` (e.g.
  ``resolved_posterior_per_subject``, ``resolved_time_series``) instead of ``*_eff``.
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

- Fix `plot_idag` so the animation heatmap only colors real edges from `row_names`
  and `adj_mat` (avoids intercept-on-diagonal and index-based mis-mapping).
- Fix `build_design_matrix` when `adj_mat` uses floating dtypes (parameter counts
  must be integers).

## [0.7.0] - 2026-07-11

### Added

- Add `mdmp.group_analysis.distance` subpackage for group-structure (GS) pairwise
  subject dissimilarity.
- Add `fit_individual_structures` to estimate one MDM per subject (workflow stage 1).
- Add `compute_mdm_distance` for the pairwise separation matrix (stages 2–3),
  accepting raw time-series arrays or pre-fitted MDM objects.
- Add `MDMDistanceResult` with dense, labelled, sparse, linkage, and cluster-cutting
  helpers (`to_frame`, `to_sparse`, `to_linkage`, `cluster_labels`, `to_similarity`).
- Add pluggable pairwise metrics via `METRIC_REGISTRY`: `lpl_separation` (default),
  `structural_hamming`, and `strength_frobenius`.
- Add sparse neighbourhood graphs via `MDMDistanceResult.to_sparse` for t-SNE,
  Isomap, and UMAP projectors.
- Add proximity-analysis helpers: `nearest_neighbours`, `silhouette`,
  `suggest_clusters`, and `bayes_factor_cut`.
- Add `mdmp.plotting.projection` with `project_distance`, `plot_projection`,
  `plot_dendrogram`, and `plot_group_embedding` (MDS, non-metric MDS, t-SNE,
  Isomap, UMAP).
- Add optional `[umap]` extra (`umap-learn`) for UMAP projection.
- Add `notebooks/08-gs-distance-projection.ipynb` demonstrating the GS distance
  and projection workflow.
- Re-export distance and projection entry points from `mdmp.group_analysis` and
  top-level `mdmp`.

### Changed

- Add `scikit-learn` as a core dependency (multidimensional projection and
  silhouette scoring).

### Fixed

- Reuse `subject_id` (or `subject` / `id`) stored on pre-fitted MDM objects when
  `subject_ids` is omitted in the distance workflow.

## [0.6.2] - 2026-04-19

### Added

- This changelog and a single-source package version in `mdmp/_version.py`, with
  dynamic metadata in `pyproject.toml` and release automation via `bump-my-version`.

[Unreleased]: https://github.com/maods2/mdmp/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/maods2/mdmp/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/maods2/mdmp/releases/tag/v0.6.2
