"""
VTS computation strategies: concatenation-based, mean-based, and median-based approaches.
"""

from abc import ABC, abstractmethod
from typing import List, Literal

import numpy as np

from .data import align_subjects
from .estimators import get_estimator
from .types import VTSResult


class BaseVTSStrategy(ABC):
    """Abstract base class for VTS computation strategies."""

    @abstractmethod
    def compute(
        self,
        data: List[np.ndarray],
        metadata: dict,
        **kwargs,
    ) -> VTSResult:
        """
        Compute Virtual Typical Subject from multi-subject data.

        Parameters
        ----------
        data : list of np.ndarray
            List of (T_s x N) arrays.
        metadata : dict
            Metadata from prepare_multi_subject_data.
        **kwargs
            Strategy-specific options.

        Returns
        -------
        VTSResult
            The computed VTS result.
        """
        pass


class ConcatenationStrategy(BaseVTSStrategy):
    """
    Concatenation-based VTS: concatenate along time, then apply estimator.

    Produces either (T_total x N) concatenated series for MDM, or (N,) summary
    via estimator over time, depending on return_series.
    """

    def __init__(
        self,
        estimator: str = "mean",
        return_series: bool = True,
    ):
        """
        Initialize concatenation strategy.

        Parameters
        ----------
        estimator : str, optional
            Estimator for global stats: "mean", "median". Used when
            return_series=False. Default "mean".
        return_series : bool, optional
            If True, return concatenated (T_total x N) for MDM fitting.
            If False, apply estimator over time to get (N,) summary.
            Default True.
        """
        self.estimator = estimator
        self.return_series = return_series

    def compute(
        self,
        data: List[np.ndarray],
        metadata: dict,
        **kwargs,
    ) -> VTSResult:
        """Compute VTS via concatenation."""
        concat = np.concatenate(data, axis=0)
        if self.return_series:
            vts_data = concat
        else:
            est_fn = get_estimator(self.estimator)
            vts_data = est_fn(concat, axis=0)
        return VTSResult(
            vts_data=vts_data,
            method="concatenation",
            n_subjects=metadata["n_subjects"],
            metadata={
                **metadata,
                "estimator": self.estimator,
                "return_series": self.return_series,
            },
        )


class MeanBasedStrategy(BaseVTSStrategy):
    """
    Mean-based VTS: mean per subject, then mean across subjects.

    Requires aligned time lengths. Uses align_subjects when lengths differ.
    """

    def __init__(
        self,
        align_method: Literal["truncate", "pad", "interpolate"] = "truncate",
    ):
        """
        Initialize mean-based strategy.

        Parameters
        ----------
        align_method : {"truncate", "pad", "interpolate"}, optional
            How to align subjects with different T. Default "truncate".
        """
        self.align_method = align_method

    def compute(
        self,
        data: List[np.ndarray],
        metadata: dict,
        **kwargs,
    ) -> VTSResult:
        """Compute VTS via subject-level then group-level mean."""
        aligned = align_subjects(data, method=self.align_method)
        subject_means = [arr for arr in aligned]
        vts_data = np.mean(subject_means, axis=0)
        return VTSResult(
            vts_data=vts_data,
            method="mean",
            n_subjects=metadata["n_subjects"],
            metadata={
                **metadata,
                "align_method": self.align_method,
            },
        )


class MedianBasedStrategy(BaseVTSStrategy):
    """
    Median-based VTS: align subjects, then pointwise median across subjects.

    Requires aligned time lengths. Uses align_subjects when lengths differ.
    """

    def __init__(
        self,
        align_method: Literal["truncate", "pad", "interpolate"] = "truncate",
    ):
        """
        Initialize median-based strategy.

        Parameters
        ----------
        align_method : {"truncate", "pad", "interpolate"}, optional
            How to align subjects with different T. Default "truncate".
        """
        self.align_method = align_method

    def compute(
        self,
        data: List[np.ndarray],
        metadata: dict,
        **kwargs,
    ) -> VTSResult:
        """Compute VTS via aligned stack, then median across subjects."""
        aligned = align_subjects(data, method=self.align_method)
        stacked = [arr for arr in aligned]
        vts_data = np.median(stacked, axis=0)
        return VTSResult(
            vts_data=vts_data,
            method="median",
            n_subjects=metadata["n_subjects"],
            metadata={
                **metadata,
                "align_method": self.align_method,
            },
        )
