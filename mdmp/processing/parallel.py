"""
Parallel implementation of node processing.
"""

from typing import Any, Callable, List, Optional

from ..progress import process_map_with_progress
from .base import NodeProcessor


class ParallelProcessor(NodeProcessor):
    """
    Parallel processor for node operations.

    Processes items in parallel using multiprocessing.
    """

    def __init__(self, n_jobs: int, verbose: bool = False):
        """
        Initialize parallel processor.

        Parameters
        ----------
        n_jobs : int
            Number of parallel workers.
        verbose : bool, optional
            Whether to show progress bars. Default is False.
        """
        super().__init__(verbose=verbose)
        self.n_jobs = n_jobs

    def process(
        self,
        items: List[Any],
        process_func: Callable[[Any], Any],
        desc: Optional[str] = None,
        unit: str = "items"
    ) -> List[Any]:
        """
        Process items in parallel.

        Parameters
        ----------
        items : list
            List of items to process.
        process_func : callable
            Function to apply to each item.
        desc : str, optional
            Description for progress bar.
        unit : str, optional
            Unit name for progress bar. Default is "items".

        Returns
        -------
        list
            List of processed results.
        """
        return process_map_with_progress(
            process_func,
            items,
            max_workers=self.n_jobs,
            desc=desc,
            disable=not self.verbose,
            unit=unit
        )
