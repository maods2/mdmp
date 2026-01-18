"""
Type definitions for scoring results.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DiscountFactorResult:
    """
    Results from discount factor selection.

    Attributes
    ----------
    lpldet : np.ndarray
        Log predictive likelihoods for each delta and node (nd, N).
    DF_hat : np.ndarray
        Selected discount factors for each node (N,).
    """
    lpldet: np.ndarray
    DF_hat: np.ndarray

    def to_dict(self) -> dict:
        """
        Convert to dictionary format for backward compatibility.

        Returns
        -------
        dict
            Dictionary with keys: lpldet, DF_hat
        """
        return {
            'lpldet': self.lpldet,
            'DF_hat': self.DF_hat
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DiscountFactorResult":
        """
        Create from dictionary format.

        Parameters
        ----------
        data : dict
            Dictionary with keys: lpldet, DF_hat

        Returns
        -------
        DiscountFactorResult
            DiscountFactorResult instance.
        """
        return cls(
            lpldet=data['lpldet'],
            DF_hat=data['DF_hat']
        )


@dataclass
class ScoreResult:
    """
    Structure score result.

    Attributes
    ----------
    total_score : float
        Total structure score (sum of maximum log predictive likelihoods).
    node_scores : np.ndarray, optional
        Individual node scores (N,). Default is None.
    """
    total_score: float
    node_scores: np.ndarray = None
