# Individual Structure (IS) — review and refinement prompt

Refactor, simplify, and harden the **Individual Structure** submodule at
`mdmp/group_analysis/inds/`. The package already went through an initial modular
split; this task is to **review what exists**, close remaining gaps, and push
the design toward the clarity of mature scientific Python libraries (compare with
`mdmp/group_analysis/vts/`).

---

## Context and purpose

**Goal:** estimate a **global structural representation** (one DAG) from multiple
individual DAGs — e.g. EEG-derived connectivity graphs per subject — by collapsing
subject-specific structures into a shared graph.

**Public entry point:** `aggregate_individual_structures` (re-exported from
`mdmp` and `mdmp.group_analysis`). Options bundle:
`ISAggregateOptions` / `aggregate_with_options`.

**Import path:** `mdmp.group_analysis.inds` (or top-level `mdmp` re-exports).
The former `mdmp.group_analysis.is` package was removed because `is` is a Python keyword.

---

## Current module layout (baseline)

| Module | Responsibility |
|--------|----------------|
| `aggregation.py` | Orchestration: validate → coerce → vote → optional refit → optional MC |
| `voting.py` | Edge-frequency thresholding + greedy acyclic repair |
| `monte_carlo.py` | DLM posterior sampling + cross-subject pooling onto global edges |
| `coercion.py` | Input normalization, binary adjacency validation, MDM duck-typing |
| `pooled_filtering.py` | `build_plot_filt_from_subjects` — pooled `Filt` dict for plotting |
| `results.py` | `ISAggregationResult`, `ISAggregatedMDMView`, `GlobalBetaMCResult`, `ISAggregateOptions`, type aliases |
| `__init__.py` | Explicit public re-exports |
| `aggregation_diagrams.md` | Mermaid flowcharts (keep in sync with code) |

**Tests:** `tests/test_is_aggregation.py`, `tests/test_is_global_beta.py`
**Notebooks:** `05-is-aggregation.ipynb`, `06-is-aggregation-cycle-demo.ipynb`,
`04-is-vs-vts-multi-individual.ipynb`

---

## Algorithm (as implemented — preserve correctness)

### Step 1 — Global DAG structure (edge voting)

For each directed edge \(i \to j\):

1. Count subjects that include the edge.
2. Compute empirical frequency \(f_{ij} = \text{count} / S\).
3. Keep edge if frequency passes threshold `tau` (default `0.5`):
   - `threshold_mode='strict'`: keep if \(f_{ij} > \tau\) (historical default).
   - `threshold_mode='inclusive'`: keep if \(f_{ij} \ge \tau\).

Output: candidate global adjacency + metadata (`edge_counts`, `edge_frequencies`,
`threshold_mode`).

### Step 2 — Enforce DAG constraints

If the thresholded graph has directed cycles:

- **Greedy repair** (in `voting.py`): find one directed cycle, remove the
  cycle edge with **lowest empirical vote frequency**, repeat until acyclic.
- This is **not** a minimum feedback arc set (global FAS); document and preserve
  that behavior unless deliberately changed.
- Record removals in `metadata['edges_removed_for_acyclicity']`.

### Step 3 — Optional global-structure refit (not always run)

When `mc_refit_global_structure=True` (requires per-subject `(T, N)` data):

- Refit each subject with `mdmp.model.refit_mdm_on_structure` on the **fixed
  aggregated DAG** (same discount/filter/smooth pipeline as `MDM`).
- Store per-subject refit outputs on `ISAggregatedMDMView.refit_filt_per_subject`
  / `refit_smoo_per_subject`.

**Default MC path** does **not** refit: it draws from each subject's **individual**
DAG filtered (or smoothed) posterior, aligning coefficients to global edges.

### Step 4 — Monte Carlo global edge coefficients (optional)

Enabled when `n_draws > 0` (default `0`; **not** 2000 — callers opt in).
Requires `rng`.

For each replicate \(b = 1, \ldots, B\) and time index \(t\):

1. **Sample** each subject's DLM regression state at \(t\) via Gamma–Normal
   mixture (`_sample_dlm_state_posterior`), using:
   - `mc_posterior='filtered'`: filter `(mt, Ct, nt, dt)`.
   - `mc_posterior='smoothed'`: smoother `(smt, sCt)` with filter `(nt, dt)`
     at the same \(t\) (pragmatic approximation — document, do not oversell).
2. **Align** local parent coefficients to each **global** edge \((p, c)\).
3. **Pool** across contributing subjects:
   - `mc_contributors='individual_edge'` (default): only subjects whose
     **individual** DAG had edge \(p \to c\) contribute; divisor \(A \le S\).
   - `mc_contributors='all_subjects'`: all subjects at each global edge;
     **requires** `mc_refit_global_structure=True`.
   - `pooling='mean_with_edge'`: \(\bar\theta^{(b)} = \frac{1}{A}\sum_{i\in\mathcal{A}} \theta_i^{(b)}\)
     — **conditional** mean among contributors; see
     [Statistical interpretation](#statistical-interpretation--required-corrections-code--docs).
   - `pooling='sum_with_edge'`: sum instead of mean.

Summaries: `GlobalBetaMCResult.beta_draws`, `beta_mean`, `beta_var`, optional
`beta_quantiles`. Multi-time via `time_indices` → shape `(B, n_edges, T)`.

See **Statistical interpretation — required corrections** below for inferential
meaning, interval interpretation, and optional population-averaging mode.

### Step 5 — Plotting adapter (optional, separate from inference)

- `pool_filt_for_plotting=True` or explicit `plot_filt` / `plot_data` /
  `plot_smoo` / `plot_df` populate `ISAggregatedMDMView` so `mdmp.plotting`
  routines accept the result like a fitted `MDM`.
- Pooled filter construction lives in `pooled_filtering.py`; it is a
  visualization summary, not a joint Bayesian posterior.

---

## Statistical interpretation — required corrections (code + docs)

The Monte Carlo aggregation is **mathematically coherent** for pooling
subject-specific DLM/MDM posteriors onto a fixed consensus DAG. The
implementation must **not** overstate the inferential meaning of
`GlobalBetaMCResult` coefficients or intervals. Apply the corrections below
across code, docstrings, metadata, README, notebooks, and
`aggregation_diagrams.md`.

### Canonical disclaimer (reuse verbatim where appropriate)

> This procedure performs Monte Carlo aggregation of independent subject-level
> posterior distributions under a fixed consensus structure. It is **not** a
> joint hierarchical Bayesian population model.

Place this (or an equivalent one-liner) in:

- `monte_carlo.py` module docstring
- `GlobalBetaMCResult` class docstring (`results.py`)
- `aggregate_individual_structures` Notes section (`aggregation.py`)
- README IS / group-analysis section
- Optional: `GlobalBetaMCResult.metadata['interpretation']` at construction time

---

### 1. Clarify the inferential target of the default pooling strategy

**Current default:** `mc_contributors='individual_edge'` +
`pooling='mean_with_edge'` computes

\[
\bar{\theta}^{(b)} = \frac{1}{A}\sum_{i \in \mathcal{A}} \theta_i^{(b)},
\qquad
\mathcal{A} = \{ i : \text{edge}_{pc} \text{ present on subject } i \text{'s individual DAG} \}.
\]

**This does NOT estimate a population-average effect across all \(S\) subjects.**
It estimates a **conditional** quantity:

\[
E[\theta_{pc,t} \mid \text{edge}_{pc} = 1 \text{ on subject's individual DAG}]
\]

—that is, the effect **among subjects who express the edge** on their individual
structure, pooled via Monte Carlo over independent subject posteriors.

**Implementation tasks:**

| Location | Change |
|----------|--------|
| `results.py` — `GlobalBetaMCResult` | State explicitly that default draws summarize a **conditional posterior mean among contributors**, not a global population effect. |
| `aggregation.py` — `pooling`, `mc_contributors` param docs | Replace vague “group mean” wording with “conditional mean over contributors \(\mathcal{A}\)”. |
| `monte_carlo.py` — `_monte_carlo_beta_draws_at_time` docstring | Add the \(E[\theta \mid \text{edge}=1]\) interpretation for the default path. |
| `README.md`, notebooks | Never call `beta_mean` a “population effect” or “global population coefficient” under default settings. Prefer **consensus-edge conditional effect** or **contributor-conditional pooled coefficient**. |
| `GlobalBetaMCResult.metadata` | Add keys e.g. `inferential_target='conditional_on_individual_edge'`, `pooling_divisor='contributors_A'` when defaults apply. |

**Do not rename** `mean_with_edge` silently; if a clearer public alias is added
(e.g. `conditional_mean_contributors`), document equivalence to the current mode.

---

### 2. Distinguish conditional vs population effects in the pipeline

The pipeline combines **hard structural selection** (voting, threshold, acyclic
repair) with **conditional coefficient aggregation** without modeling edge
absence probabilistically.

**Document explicitly:**

- Subjects without the edge on their individual DAG are **excluded** from
  aggregation at that edge (not averaged in as zero unless a new mode adds that).
- Edge absence is **not** modeled probabilistically.
- No spike-and-slab or integration over structural uncertainty is performed.
- The resulting posterior summaries are **conditional on edge presence** in the
  contributor set (and, separately, conditional on the fixed consensus graph
  \(G^*\)—see §4).

**Implementation tasks:**

- Add a short **“Statistical model assumptions”** subsection to
  `aggregate_individual_structures` Notes listing the four bullets above.
- In `aggregation_diagrams.md`, annotate the MC pooling box:
  “conditional on individual edge presence; non-contributors omitted”.
- Consider `ISAggregationResult.metadata['structural_selection']` =
  `'hard_threshold_and_repair'` to make the selection mechanism machine-readable.

---

### 3. Clarify this is NOT a hierarchical Bayesian population model

The method propagates **independent** subject-specific posteriors through Monte
Carlo averaging. It does **not** model

\[
\theta_{it} \sim \mathcal{N}(\mu_t, \tau_t^2)
\]

with explicit population-level random effects.

**State explicitly that the method does NOT:**

- perform hierarchical pooling across subjects;
- estimate population random effects \(\mu_t, \tau_t\);
- apply Bayesian shrinkage across subjects;
- treat subjects as exchangeable draws from a hyperprior.

**It DOES assume independence** between subjects during posterior sampling
(replicates draw independently per subject; no cross-subject coupling in the
sampler).

**Implementation tasks:**

| Location | Change |
|----------|--------|
| `monte_carlo.py` module docstring | Expand the existing “not hierarchical” note with the four “does NOT” bullets and the canonical disclaimer. |
| `results.py` — `GlobalBetaMCResult` | Same; reference `metadata['independence_assumption']='per_subject_dlm_posteriors'`. |
| `group_analysis/__init__.py` submodule blurb | One sentence: MC output is not a hierarchical population model. |
| Tests | Optional: assert `global_beta_mc.metadata` contains interpretation keys when `n_draws > 0`. |

---

### 4. Clarify structural uncertainty handling

Pipeline order:

1. edge voting → 2. thresholding → 3. hard DAG repair → 4. posterior inference
   **conditional on** the resulting graph \(G^*\).

Inference targets \(p(\theta \mid G^*)\), **not**

\[
p(\theta) = \sum_G p(\theta \mid G)\, p(G).
\]

Structural uncertainty (which edges belong in \(G\), acyclicity choices during
repair) is **not** propagated into `beta_draws` or quantile intervals.

**Implementation tasks:**

- Document in `voting.py` and `aggregation.py` that `metadata['edges_removed_for_acyclicity']` is diagnostic only—not uncertainty propagated into MC.
- Add to `GlobalBetaMCResult.metadata`: `graph_uncertainty='fixed_consensus_dag'`, `consensus_adj_mat` reference or hash optional.
- README / notebooks: intervals are **parameter uncertainty given \(G^*\)** and contributor set, not **structure uncertainty**.

---

### 5. Clarify what Monte Carlo uncertainty intervals capture

**Intervals from `beta_draws` / `beta_quantiles` DO capture:**

- uncertainty from each subject-level DLM posterior (filter or smoothed approximation);
- propagation through the aggregation map (align + pool per replicate);
- empirical between-subject variability **among contributors** at that edge.

**They DO NOT capture:**

- uncertainty in graph structure \(G\);
- hierarchical population uncertainty (hyperpriors, random effects);
- joint dependence between subjects beyond independent draw composition.

**Implementation tasks:**

- Add a **“Uncertainty decomposition”** bullet list to `GlobalBetaMCResult` docstring (included / excluded as above).
- If plotting helpers add credible bands, label them e.g.
  `"MC aggregator uncertainty (fixed DAG, conditional contributors)"` not
  `"population credible interval"`.
- Document that `beta_var` is **variance across MC replicates** of the pooled
  statistic, not a full marginal posterior variance over population and structure.

---

### 6. Clarify the smoothed posterior approximation (wording + code comments)

When `mc_posterior='smoothed'`, the sampler combines:

- smoothed state moments `(smt, sCt)`;
- filtered variance parameters `(nt, dt)` at the **same** time index;

in the existing Gamma–Normal / Student-\(t\) draw step. This is a **practical
approximation**, not an exact draw from the full joint smoothed DLM posterior.

**Required wording** (adapt consistently):

> When `mc_posterior='smoothed'`, the implementation uses smoothed state moments
> together with filtered variance parameters at the same time index as a
> **practical approximation** to the full smoothed Student-\(t\) posterior.

**Implementation tasks:**

| Location | Change |
|----------|--------|
| `monte_carlo.py` — `_sample_dlm_state_posterior` | Inline comment when called from smoothed branch. |
| `aggregation.py` — `mc_posterior` param doc | Use exact wording above. |
| `results.py` — `GlobalBetaMCResult` step (1) | Same. |
| `GlobalBetaMCResult.metadata` | Set `smoothed_sampling='approximate_moments_with_filter_nt_dt'` when `mc_posterior='smoothed'`. |

Do **not** describe smoothed MC output as “exact smoothing posterior” anywhere.

---

### 7. Optional: expose true population averaging mode (code change)

If users need an **unconditional population-average** effect across all \(S\)
subjects (not only contributors), add an **opt-in** pooling mode—do not change
default behavior.

**Proposed API extension:**

```python
PoolingMode = Literal[
    "mean_with_edge",           # current default: 1/A over contributors (conditional)
    "sum_with_edge",            # current
    "mean_all_subjects",        # NEW: (1/S) * sum_i tilde_theta_i
]
```

**Semantics for `mean_all_subjects`:**

\[
\bar{\theta}^{(b)} = \frac{1}{S}\sum_{i=1}^{S} \tilde{\theta}_i^{(b)},
\qquad
\tilde{\theta}_i^{(b)} =
\begin{cases}
\theta_i^{(b)} & \text{if subject } i \text{ has edge on individual DAG} \\
0 & \text{otherwise}
\end{cases}
\]

(or document spike-at-zero explicitly; alternative: `NaN` excluded from sum with
divisor \(S\) fixed—**pick one** and test; zero-fill is the user’s suggested
formulation).

**Implementation tasks if added:**

1. Extend `PoolingMode` in `results.py` and validation in `aggregation.py`.
2. Implement branch in `_monte_carlo_beta_draws_at_time` (`monte_carlo.py`).
3. Document contrast:
   - `mean_with_edge` → \(E[\theta \mid \text{edge}=1]\) among contributors;
   - `mean_all_subjects` → unconditional population average with zero imputation
     for non-expressers (still **not** hierarchical; still **fixed** \(G^*\)).
4. Add tests in `tests/test_is_global_beta.py` comparing divisors \(A\) vs \(S\).
5. CHANGELOG entry under **Added**; note this is **not** spike-and-slab.

**If not implemented:** document in README that unconditional population
averaging is **out of scope** and point users to the conditional interpretation
or external hierarchical tooling.

---

### Interpretation correction checklist (deliverable)

When executing this prompt, verify:

- [ ] No docstring, README, or notebook calls default `beta_mean` a population effect.
- [ ] Conditional vs population distinction is explicit for `individual_edge` + `mean_with_edge`.
- [ ] Hard structural selection vs probabilistic edge modeling is documented.
- [ ] Canonical non-hierarchical disclaimer appears in all primary entry points.
- [ ] Fixed-\(G^*\) vs summed-over-\(G\) distinction is documented.
- [ ] Interval interpretation lists included vs excluded uncertainty sources.
- [ ] Smoothed MC uses the approved approximation wording everywhere.
- [ ] `GlobalBetaMCResult.metadata` exposes machine-readable interpretation flags.
- [ ] Optional `mean_all_subjects` either implemented with tests or explicitly deferred.

---

## Input flexibility (preserve)

- Plain binary adjacency matrices / DataFrames, or a **homogeneous** sequence of
  fitted `MDM`-like objects (duck-typed: `adj_mat`, `Filt`, `node_names`).
- Single MDM or single 2D adjacency auto-wrapped to a one-element sequence.
- MDM path auto-fills `filtered_per_subject` and can derive mean `plot_data`.

---

## What to review and improve

Analyze the **current** codebase (not a greenfield rewrite). Prioritize
readability, mathematical traceability, modularity, debuggability, and simplicity
**without** reducing scientific correctness.

### Likely remaining issues

1. **`aggregation.py` still mixes orchestration with refit wiring** — helpers
   like `_build_mc_inputs`, `_refit_each_subject_on_global_adj`, and validation
   could move to dedicated modules (e.g. `refit.py`, `validation.py`).
2. **Stale references** — docstrings mention `mc_global_beta` module that no
   longer exists (logic is in `monte_carlo.py`); align docs and diagrams.
3. **Monte Carlo hot path** — nested Python loops in
   `_monte_carlo_beta_draws_at_time`; consider vectorization **only if** pooling
   definition stays identical (note in `monte_carlo.py` docstring).
4. **Public vs private API** — many `_`-prefixed helpers; clarify what should
   become public (if anything) vs stay internal.
5. **Plotting vs inference boundary** — `ISAggregatedMDMView` intentionally
   mirrors `MDM` for plotting; confirm plotting-only code never leaks into MC
   or voting paths.
6. **Naming consistency** — `IS*` prefix vs descriptive names; compare export
   style with `vts` (`VTSResult`, `compute_vts`, strategy classes).
7. **Default MC draw count** — original spec suggested default `n_draws=2000`;
   current default is `0`. Recommend whether to change (truncated or keep
   opt-in and document rationale.
8. **`aggregation_diagrams.md`** — partly Portuguese; consider English for repo
   consistency, and verify diagrams match post-refactor module names.
9. **Overstated inferential language** — docstrings/README may still imply
   population or hierarchical effects; apply
   [Statistical interpretation](#statistical-interpretation--required-corrections-code--docs)
   corrections and populate `GlobalBetaMCResult.metadata` interpretation keys.

### Separation of concerns (target)

| Concern | Where it should live |
|---------|---------------------|
| Structural voting + acyclic repair | `voting.py` |
| Input coercion / validation | `coercion.py` (+ optional `validation.py`) |
| Global-structure refit | dedicated module or thin wrapper over `refit_mdm_on_structure` |
| Monte Carlo sampling + pooling | `monte_carlo.py` |
| Result / options dataclasses | `results.py` |
| Plot-ready filter pooling | `pooled_filtering.py` |
| Single orchestrator | `aggregation.py` (thin) |

### Anti-patterns to flag

- Duplicated validation logic between `aggregation.py` and submodules.
- Oversized keyword surface on `aggregate_individual_structures` without
  `ISAggregateOptions` (already partially addressed — assess if further grouping helps).
- Hidden behavior (e.g. smoothed MC using filter `nt`/`dt`) without docstring visibility.
- Mixing nan-handling semantics across pooling and quantiles.

---

## Deliverables

Produce:

1. **Structural review** of `mdmp/group_analysis/inds/` against the layout above.
2. **Gap analysis:** what is already done vs what still needs work.
3. **Refactoring plan** with ordered, small PR-sized steps (avoid big-bang).
4. **Module/package organization** proposal (if changes warranted).
5. **Naming recommendations** (types, functions, optional directory rename).
6. **Separation-of-concern** checklist with file-level moves.
7. **Execution-flow simplification** — ideally one readable top-down path in
   `aggregate_individual_structures`.
8. **Maintainability / readability** recommendations (docstrings, types, tests).
9. **Anti-patterns** currently present (with file/line references).
10. **Scientific code quality** — explicit assumptions (greedy repair, independent
    MC, smoothed sampling shortcut, contributor divisor \(A\)), plus complete
    [statistical interpretation corrections](#statistical-interpretation--required-corrections-code--docs).
11. **Concrete simplification examples** where a 10–30 line rewrite clarifies intent.
12. **Interpretation correction checklist** — all items in that section verified.

### Constraints

- Do **not** change default pooling math or voting semantics silently; call out any
  proposed behavioral change explicitly (new opt-in modes such as
  `mean_all_subjects` are allowed with tests and CHANGELOG).
- Keep backward compatibility for public API unless a breaking change is
  justified and documented in `CHANGELOG.md`.
- Run / extend `tests/test_is_aggregation.py` and `tests/test_is_global_beta.py`
  for any refactor.
- Match patterns used in `mdmp/group_analysis/vts/` where appropriate.

**Success criteria:** the package feels easy to navigate, debug, and validate
mathematically — explicit rather than overly abstract — while preserving the
scientific behavior already encoded in tests and notebooks.
