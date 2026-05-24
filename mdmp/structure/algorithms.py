"""
Structure learning algorithms for MDM Bayesian networks.

This module implements the base class and concrete algorithm implementations
using the Strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd

from .scoring import MdmStructureScore
from .utils import extract_adjacency_from_model


def _build_pgmpy_score(df, node_names, nbf):
    """Wrap MdmStructureScore for pgmpy compatibility.

    Returns (df, columns, score_wrapper) ready to pass to a pgmpy estimator.
    """
    try:
        from pgmpy.estimators import StructureScore
    except ImportError as exc:
        raise ImportError(
            "pgmpy is required for structure learning. Install with `pip install pgmpy`."
        ) from exc
    except AttributeError as exc:
        raise _pgmpy_import_error_hint(exc) from exc

    N = df.shape[1]
    columns = list(node_names) if node_names is not None else [f"V{i+1}" for i in range(N)]
    mdm_score_obj = MdmStructureScore(df, nbf_value=nbf)

    class _Wrapper(StructureScore):
        def __init__(self, df_input, mdm_score):
            super().__init__(df_input)
            self._mdm_score = mdm_score

        def local_score(self, variable, parents):
            return self._mdm_score.local_score(variable, parents)

    return columns, _Wrapper(df, mdm_score_obj)


def _preload_torch_for_pgmpy() -> None:
    """
    Import PyTorch before pgmpy when available.

    pgmpy pulls in torch transitively; on some setups that nested import leaves
    ``torch`` only partially initialized (e.g. ``torch.types`` / ``memory_format``
    missing). Loading torch first avoids that ordering issue.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    except AttributeError as exc:
        raise _pgmpy_import_error_hint(exc) from exc


def _pgmpy_import_error_hint(exc: BaseException) -> ImportError:
    msg = str(exc).lower()
    if "torch" in msg and ("partially initialized" in msg or "has no attribute" in msg):
        return ImportError(
            "PyTorch failed to finish loading while importing pgmpy. Typical causes: "
            "a local file named torch.py (or a folder torch/) shadowing the real "
            "package on sys.path, or a broken PyTorch install. Restart the kernel, "
            "run from a directory without such a file, and reinstall torch if needed."
        )
    return ImportError(f"Could not import pgmpy estimators: {exc}")


class BaseLearningAlgorithm(ABC):
    """
    Base class for structure learning algorithms.

    All learning algorithms should inherit from this class and implement
    the `learn` method.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the learning algorithm.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        """
        self.verbose = verbose

    @abstractmethod
    def learn(
        self,
        data: np.ndarray,
        nbf: int,
        delta: np.ndarray,
        node_names: Optional[List[str]],
        **kwargs
    ) -> np.ndarray:
        """
        Learn structure and return adjacency matrix.

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N).
        nbf : int
            Burn-in time point.
        delta : np.ndarray
            Sequence of discount factors.
        node_names : list of str, optional
            Node/variable names.
        **kwargs
            Additional algorithm-specific arguments.

        Returns
        -------
        np.ndarray
            Adjacency matrix (N x N).
        """
        pass


class PgmpyAlgorithmMixin:
    """
    Mixin class for common pgmpy algorithm functionality.

    Provides shared methods for algorithms that use pgmpy.
    """

    def _prepare_dataframe(
        self,
        data: np.ndarray,
        node_names: Optional[List[str]]
    ) -> tuple:
        """
        Prepare DataFrame and column names for pgmpy.

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N).
        node_names : list of str, optional
            Node/variable names.

        Returns
        -------
        tuple
            Tuple of (DataFrame, columns).
        """
        N = data.shape[1]
        if node_names is not None:
            if len(node_names) != N:
                raise ValueError(
                    f"node_names length ({len(node_names)}) must match number of columns "
                    f"in data ({N})"
                )
            columns = list(node_names)
        else:
            columns = [f"V{i+1}" for i in range(N)]
        df = pd.DataFrame(data, columns=columns)
        return df, columns

    def _clean_kwargs(self, kwargs: dict) -> dict:
        """
        Remove algorithm-specific kwargs that shouldn't be passed to pgmpy.

        Parameters
        ----------
        kwargs : dict
            Original kwargs.

        Returns
        -------
        dict
            Cleaned kwargs.
        """
        cleaned = dict(kwargs)
        cleaned.pop("methodtype", None)
        cleaned.pop("scoretype", None)
        cleaned.pop("scoring_method", None)
        return cleaned


class HillClimbingAlgorithm(BaseLearningAlgorithm, PgmpyAlgorithmMixin):
    """
    Hill-climbing structure learning algorithm using pgmpy.

    This algorithm uses pgmpy's HillClimbSearch with a custom MDM scoring
    function that optimizes the log predictive likelihood per node.
    """

    def learn(
        self,
        data: np.ndarray,
        nbf: int,
        delta: np.ndarray,
        node_names: Optional[List[str]],
        **kwargs
    ) -> np.ndarray:
        """
        Learn structure using pgmpy's hill-climbing with a custom MDM score.

        This mirrors the R implementation using bnlearn::hc with a custom score
        function by optimizing the MDM log predictive likelihood per node.
        """
        _preload_torch_for_pgmpy()
        try:
            from pgmpy.estimators import HillClimbSearch
        except ImportError as exc:
            raise ImportError(
                "pgmpy is required for hill-climbing algorithm. "
                "Install with `pip install pgmpy`."
            ) from exc
        except AttributeError as exc:
            raise _pgmpy_import_error_hint(exc) from exc

        df, columns = self._prepare_dataframe(data, node_names)
        columns, mdm_score = _build_pgmpy_score(df, node_names, nbf)
        model = HillClimbSearch(df).estimate(
            scoring_method=mdm_score, **self._clean_kwargs(kwargs)
        )
        return extract_adjacency_from_model(model, columns)


class TabuAlgorithm(BaseLearningAlgorithm, PgmpyAlgorithmMixin):
    """
    Tabu search structure learning algorithm using pgmpy.

    This algorithm uses pgmpy's HillClimbSearch with tabu search enabled
    (via tabu_length parameter) and a custom MDM scoring function.
    """

    def learn(
        self,
        data: np.ndarray,
        nbf: int,
        delta: np.ndarray,
        node_names: Optional[List[str]],
        **kwargs
    ) -> np.ndarray:
        """
        Learn structure using pgmpy's hill-climbing with tabu search and custom MDM score.

        This uses the same implementation as HillClimbingAlgorithm but enables
        tabu search by setting tabu_length in the kwargs.

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N).
        nbf : int
            Burn-in time point.
        delta : np.ndarray
            Sequence of discount factors.
        node_names : list of str, optional
            Node/variable names.
        **kwargs
            Additional arguments passed to HillClimbSearch.estimate().
            Common parameters:
            - tabu_length: int, default 100. Length of tabu list.
            - max_iter: int, default 1000000. Maximum iterations.
            - epsilon: float, default 0.0001. Convergence threshold.
            - show_progress: bool, default True. Show progress.

        Returns
        -------
        np.ndarray
            Adjacency matrix (N x N).
        """
        _preload_torch_for_pgmpy()
        try:
            from pgmpy.estimators import HillClimbSearch
        except ImportError as exc:
            raise ImportError(
                "pgmpy is required for tabu search algorithm. "
                "Install with `pip install pgmpy`."
            ) from exc
        except AttributeError as exc:
            raise _pgmpy_import_error_hint(exc) from exc

        df, columns = self._prepare_dataframe(data, node_names)
        columns, mdm_score = _build_pgmpy_score(df, node_names, nbf)
        tabu_kwargs = self._clean_kwargs(kwargs)
        if "tabu_length" not in tabu_kwargs:
            tabu_kwargs["tabu_length"] = 100
        model = HillClimbSearch(df).estimate(scoring_method=mdm_score, **tabu_kwargs)
        return extract_adjacency_from_model(model, columns)


class MMHCAlgorithm(BaseLearningAlgorithm, PgmpyAlgorithmMixin):
    """
    Max-Min Hill-Climbing (MMHC) structure learning algorithm using pgmpy.

    This algorithm uses pgmpy's MmhcEstimator with a custom MDM scoring
    function. MMHC first learns an undirected skeleton via MMPC (Max-Min
    Parents and Children), then orients edges using hill-climbing with
    the custom MDM score.
    """

    def learn(
        self,
        data: np.ndarray,
        nbf: int,
        delta: np.ndarray,
        node_names: Optional[List[str]],
        **kwargs
    ) -> np.ndarray:
        """
        Learn structure using pgmpy's MMHC with a custom MDM score.

        MMHC first learns a skeleton via conditional independence tests (MMPC),
        then orients edges using hill-climbing with the custom MDM scoring
        function that optimizes the log predictive likelihood per node.
        """
        _preload_torch_for_pgmpy()
        try:
            from pgmpy.estimators import MmhcEstimator
        except ImportError as exc:
            raise ImportError(
                "pgmpy is required for MMHC algorithm. "
                "Install with `pip install pgmpy`."
            ) from exc
        except AttributeError as exc:
            raise _pgmpy_import_error_hint(exc) from exc

        df, columns = self._prepare_dataframe(data, node_names)
        columns, mdm_score = _build_pgmpy_score(df, node_names, nbf)
        model = MmhcEstimator(df).estimate(
            scoring_method=mdm_score, **self._clean_kwargs(kwargs)
        )
        return extract_adjacency_from_model(model, columns)


class NotearsAlgorithm(BaseLearningAlgorithm):
    """
    NOTEARS (Non-combinatorial Optimization via Trace Exponential and
    Augmented lagrangian for Structure learning) algorithm.

    Uses continuous optimization to learn DAG structure from data.
    NOTEARS assumes i.i.d. data; when given time series, each row is
    treated as an independent sample. Does not use MDM scoring.

    Reference: Zheng et al. (2018) "DAGs with NO TEARS"
    Install: pip install -e ../notears from repo root.
    """

    def learn(
        self,
        data: np.ndarray,
        nbf: int,
        delta: np.ndarray,
        node_names: Optional[List[str]],
        **kwargs
    ) -> np.ndarray:
        """
        Learn structure using NOTEARS linear model.

        Parameters
        ----------
        data : np.ndarray
            Time series data (T x N). Treated as i.i.d. samples for NOTEARS.
        nbf : int
            Burn-in (ignored by NOTEARS, kept for API compatibility).
        delta : np.ndarray
            Discount factors (ignored by NOTEARS, kept for API compatibility).
        node_names : list of str, optional
            Node names (ignored, kept for API compatibility).
        **kwargs
            Passed to notears_linear:
            - lambda1 : float, L1 penalty (default 0.1)
            - loss_type : str, 'l2', 'logistic', or 'poisson' (default 'l2')
            - w_threshold : float, edge weight threshold (default 0.3)
            - max_iter : int, max dual ascent steps (default 100)
            - h_tol : float, acyclicity tolerance (default 1e-8)

        Returns
        -------
        np.ndarray
            Adjacency matrix (N x N).
        """
        try:
            from notears.linear import notears_linear
        except ImportError as exc:
            raise ImportError(
                "notears is required for NOTEARS algorithm. "
                "Install with `pip install -e ../notears` from repo root, "
                "or `pip install mdmp[notears]`."
            ) from exc

        lambda1 = kwargs.pop("lambda1", 0.1)
        loss_type = kwargs.pop("loss_type", "l2")
        w_threshold = kwargs.pop("w_threshold", 0.3)
        max_iter = kwargs.pop("max_iter", 100)
        h_tol = kwargs.pop("h_tol", 1e-8)

        # NOTEARS expects (n, d) - each row is a sample
        X = np.asarray(data, dtype=np.float64)

        if self.verbose:
            print("Running NOTEARS linear (treating time series as i.i.d. samples)...")

        W_est = notears_linear(
            X,
            lambda1=lambda1,
            loss_type=loss_type,
            max_iter=max_iter,
            h_tol=h_tol,
            w_threshold=w_threshold,
        )

        # W_est[j,i] != 0 means edge j -> i (j is parent of i)
        # adj[i,j] = 1 means i is parent of j, so adj[j,i] = 1 when j->i
        adj = (np.abs(W_est) > w_threshold).astype(int)

        if self.verbose:
            n_edges = int(adj.sum())
            print(f"NOTEARS found {n_edges} edges.")

        return adj


class IpaAlgorithm(BaseLearningAlgorithm):
    """
    Integer Programming Approach (IPA) structure learning algorithm.

    This algorithm uses GOBNILP for exact structure learning via integer
    programming. Currently not implemented.
    """

    def learn(
        self,
        data: np.ndarray,
        nbf: int,
        delta: np.ndarray,
        node_names: Optional[List[str]],
        **kwargs
    ) -> np.ndarray:
        """
        Learn structure using Integer Programming Approach (IPA) via GOBNILP.

        Note: This requires GOBNILP to be installed and configured.
        """
        raise NotImplementedError(
            "IPA method with GOBNILP is not yet implemented in Python. "
            "Please use one of: hc, tabu"
        )
