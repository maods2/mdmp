"""
Serial implementation of node processing.
"""

from typing import Any, Callable, List, Optional

from .base import NodeProcessor


class SerialProcessor(NodeProcessor):
    """
    Serial processor for node operations.

    Processes items one at a time in sequence.
    """

    def process(
        self,
        items: List[Any],
        process_func: Callable[[Any], Any],
        desc: Optional[str] = None,
        unit: str = "items"
    ) -> List[Any]:
        """
        Process items serially.

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
        pbar = self._create_progress_bar(len(items), desc=desc, unit=unit)
        results = []

        try:
            for item in items:
                result = process_func(item)
                results.append(result)
                if hasattr(pbar, 'update'):
                    pbar.update(1)
        finally:
            if hasattr(pbar, 'close'):
                pbar.close()

        return results
