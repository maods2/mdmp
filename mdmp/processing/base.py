"""
Base classes for node processing abstractions.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from ..progress import get_progress_bar


class NodeProcessor(ABC):
    """
    Abstract base class for processing nodes in serial or parallel.

    This class provides a unified interface for processing operations
    that can be executed either serially or in parallel, eliminating
    code duplication.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize processor.

        Parameters
        ----------
        verbose : bool, optional
            Whether to show progress bars. Default is False.
        """
        self.verbose = verbose

    @abstractmethod
    def process(
        self,
        items: List[Any],
        desc: Optional[str] = None,
        unit: str = "items"
    ) -> List[Any]:
        """
        Process a list of items.

        Parameters
        ----------
        items : list
            List of items to process.
        desc : str, optional
            Description for progress bar.
        unit : str, optional
            Unit name for progress bar. Default is "items".

        Returns
        -------
        list
            List of processed results.
        """
        pass

    def _create_progress_bar(self, total: int, desc: Optional[str] = None, unit: str = "items"):
        """Create a progress bar for tracking progress."""
        return get_progress_bar(
            total=total,
            desc=desc,
            disable=not self.verbose,
            unit=unit
        )
