"""Refit orchestration: run MDM filtering on the fixed consensus DAG per subject.

Statistical note — structural conditioning
------------------------------------------
When ``mc_refit_global_structure=True``, each subject's DLM is refit using the
consensus DAG G* as a fixed design. All subsequent Monte Carlo samples are then
conditional on that single fixed structure:

    p(θ | G*)

This is **not** the Bayesian model average over graph uncertainty:

    p(θ) = Σ_G p(θ | G) p(G)

Structural uncertainty is never propagated; the consensus DAG is treated as
known. Monte Carlo intervals from pooled samples reflect uncertainty in
individual DLM posteriors given G*, not uncertainty about G* itself.
"""

from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ..._node_dispatch import smooth_all_nodes
from ...model.refit import refit_mdm_on_structure
from .results import MCPosteriorSource


class _MCInputs(NamedTuple):
    """Bundled Monte Carlo inputs once the consensus DAG is fixed."""

    filt_per_subject: List[Mapping[str, Any]]
    design_adjs: List[np.ndarray]
    smoothed_per_subject: Optional[List[Dict[str, Any]]]
    refit_filt_per_subject: Optional[List[Dict[str, Any]]]
    refit_smoo_per_subject: Optional[List[Dict[str, Any]]]


class IndsRefitResult(NamedTuple):
    """Per-subject refit outputs on the consensus DAG G*."""

    filt_per_subject: List[Dict[str, Any]]
    smoo_per_subject: List[Dict[str, Any]]
    filt_for_mc: List[Mapping[str, Any]]


def _per_subject_data_for_refit(
    n_subjects: int,
    n_nodes: int,
    data_per_subject: Optional[Sequence[np.ndarray]],
    mdm_data_per_subject: Optional[List[np.ndarray]],
) -> List[np.ndarray]:
    if data_per_subject is not None:
        if len(data_per_subject) != n_subjects:
            raise ValueError(
                f"data_per_subject length {len(data_per_subject)} must match "
                f"number of subjects {n_subjects}"
            )
        return [np.asarray(x, dtype=float) for x in data_per_subject]
    if mdm_data_per_subject is not None:
        return mdm_data_per_subject
    raise ValueError(
        "mc_refit_global_structure=True requires per-subject data "
        "(fitted MDM inputs with .data, or data_per_subject=...)"
    )


def _refit_each_subject_on_global_adj(
    datas: List[np.ndarray],
    global_adj: np.ndarray,
    names: List[str],
    n_nodes: int,
    mc_refit_n_jobs: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Mapping[str, Any]]]:
    """Refit MDM filtering on ``global_adj`` for each subject's data array.

    Returns ``(refit_filt, refit_smoo, filt_mc)`` where ``filt_mc`` is used for
    Monte Carlo samples (equal to ``refit_filt``; returned separately for clarity).

    All posteriors are conditional on the fixed consensus DAG G*.
    """
    refit_filt: List[Dict[str, Any]] = []
    refit_smoo: List[Dict[str, Any]] = []
    filt_mc: List[Mapping[str, Any]] = []
    for di in datas:
        if di.ndim != 2 or di.shape[1] != n_nodes:
            raise ValueError(
                f"each per-subject data array must have shape (T, {n_nodes}); "
                f"got {di.shape}"
            )
        rfit = refit_mdm_on_structure(
            di,
            global_adj,
            node_names=names,
            verbose=False,
            n_jobs=mc_refit_n_jobs,
        )
        refit_filt.append(dict(rfit.Filt))
        smoo_raw = getattr(rfit, "Smoo", None)
        if smoo_raw is None:
            smoo_raw = smooth_all_nodes(
                mt=rfit.Filt["mt"],
                Ct=rfit.Filt["Ct"],
                Rt=rfit.Filt["Rt"],
                nt=rfit.Filt["nt"],
                dt=rfit.Filt["dt"],
                n_jobs=mc_refit_n_jobs,
            )
        refit_smoo.append(dict(smoo_raw))
        filt_mc.append(rfit.Filt)
    return refit_filt, refit_smoo, filt_mc


def refit_on_consensus(
    global_adj: np.ndarray,
    node_names: List[str],
    *,
    data_per_subject: Sequence[np.ndarray],
    mc_refit_n_jobs: Optional[int] = None,
) -> IndsRefitResult:
    """
    Refit each subject's DLM on the fixed consensus DAG G*.

    Recommended before Monte Carlo when per-subject ``(T, N)`` data are available.
    All posteriors are conditional on G*: ``p(θ | G*)``.
    """
    n_nodes = int(np.asarray(global_adj).shape[0])
    datas = [np.asarray(x, dtype=float) for x in data_per_subject]
    refit_filt, refit_smoo, filt_mc = _refit_each_subject_on_global_adj(
        datas,
        np.asarray(global_adj, dtype=int),
        node_names,
        n_nodes,
        mc_refit_n_jobs,
    )
    return IndsRefitResult(refit_filt, refit_smoo, filt_mc)


def build_mc_inputs(
    *,
    mc_refit_global_structure: bool,
    mc_posterior: MCPosteriorSource,
    arrays: List[np.ndarray],
    names: List[str],
    n_subjects: int,
    n_nodes: int,
    out_adj: np.ndarray,
    resolved_posterior_per_subject: Optional[Sequence[Mapping[str, Any]]],
    mdm_data_per_subject: Optional[List[np.ndarray]],
    data_per_subject: Optional[Sequence[np.ndarray]],
    mc_refit_n_jobs: Optional[int],
) -> _MCInputs:
    """Assemble MC inputs after the consensus DAG is fixed.

    When ``mc_refit_global_structure=True`` subjects are refit on the fixed
    consensus DAG G* before samples are taken.  In either path the downstream
    Monte Carlo is conditioned on a single fixed DAG (G* or the original
    individual DAGs).

    Returns an :class:`_MCInputs` bundle consumed by
    :func:`~mdmp.group_analysis.inds.monte_carlo._monte_carlo_global_edge_beta`.
    """
    from .monte_carlo import _smooth_filtered_sequence

    global_adj = np.asarray(out_adj, dtype=int)

    if mc_refit_global_structure:
        datas = _per_subject_data_for_refit(
            n_subjects, n_nodes, data_per_subject, mdm_data_per_subject
        )
        refit_filt, refit_smoo, filt_mc = _refit_each_subject_on_global_adj(
            datas, global_adj, names, n_nodes, mc_refit_n_jobs
        )
        design = [global_adj.copy() for _ in range(n_subjects)]
        smoo_seq = refit_smoo if mc_posterior == "smoothed" else None
        return _MCInputs(filt_mc, design, smoo_seq, refit_filt, refit_smoo)

    assert resolved_posterior_per_subject is not None
    filt_mc = list(resolved_posterior_per_subject)
    smoo_seq = (
        _smooth_filtered_sequence(filt_mc, mc_refit_n_jobs)
        if mc_posterior == "smoothed"
        else None
    )
    return _MCInputs(filt_mc, arrays, smoo_seq, None, None)
