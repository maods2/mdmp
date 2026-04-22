"""
Progress bar utilities for MDM operations.

This module provides progress bar functionality using tqdm for visualizing
MDM processing progress, especially useful for large time series.
"""

from typing import Any, Iterable, Optional

try:
    from tqdm import tqdm
    from tqdm.contrib.concurrent import process_map
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Create dummy functions if tqdm is not available
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is None:
            return None
        return iterable

    def process_map(*args, **kwargs):
        from concurrent.futures import ProcessPoolExecutor
        func = args[0]
        iterable = args[1]
        max_workers = kwargs.get('max_workers', 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(func, iterable))


def get_progress_bar(
    iterable: Optional[Iterable] = None,
    desc: Optional[str] = None,
    total: Optional[int] = None,
    disable: bool = False,
    **kwargs
) -> Any:
    """
    Get a progress bar wrapper that respects verbose settings.

    Parameters
    ----------
    iterable : iterable, optional
        Iterable to wrap with progress bar. If None, returns a progress bar
        that can be updated manually.
    desc : str, optional
        Description to display on the progress bar.
    total : int, optional
        Total number of iterations. If None, will be inferred from iterable.
    disable : bool, optional
        If True, disable the progress bar. Default is False.
    **kwargs
        Additional arguments passed to tqdm.

    Returns
    -------
    tqdm progress bar or iterable
        Progress bar if tqdm is available and not disabled, otherwise
        returns the iterable unchanged.
    """
    if not TQDM_AVAILABLE or disable:
        return iterable

    tqdm_kwargs = {
        'desc': desc,
        'total': total,
        'disable': disable,
        **kwargs
    }

    if iterable is None:
        # Return a progress bar that can be updated manually
        return tqdm(**tqdm_kwargs)
    else:
        return tqdm(iterable, **tqdm_kwargs)


def process_map_with_progress(
    func,
    iterable: Iterable,
    max_workers: int = 1,
    desc: Optional[str] = None,
    disable: bool = False,
    **kwargs
) -> list:
    """
    Process items in parallel with progress bar.

    Parameters
    ----------
    func : callable
        Function to apply to each item.
    iterable : iterable
        Iterable of items to process.
    max_workers : int, optional
        Number of parallel workers. Default is 1.
    desc : str, optional
        Description for the progress bar.
    disable : bool, optional
        If True, disable the progress bar. Default is False.
    **kwargs
        Additional arguments passed to process_map.

    Returns
    -------
    list
        List of results from applying func to each item.
    """
    if not TQDM_AVAILABLE or disable:
        # Fallback to regular processing if no progress needed
        if max_workers == 1:
            return [func(item) for item in iterable]
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                return list(executor.map(func, iterable))

    # Use tqdm's process_map for progress tracking
    # For single worker, still use tqdm for progress display
    if max_workers == 1:
        # Use tqdm with regular map for single worker
        from tqdm import tqdm
        return [func(item) for item in tqdm(iterable, desc=desc, disable=disable, **kwargs)]
    else:
        return process_map(
            func,
            iterable,
            max_workers=max_workers,
            desc=desc,
            disable=disable,
            **kwargs
        )


class ProgressContext:
    """
    Context manager for managing nested progress bars.

    This class helps manage multiple progress bars that may be nested
    or sequential, ensuring proper display and cleanup.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize progress context.

        Parameters
        ----------
        verbose : bool, optional
            Whether to show progress bars. Default is True.
        """
        self.verbose = verbose
        self.bars = []

    def create_bar(
        self,
        iterable: Optional[Iterable] = None,
        desc: Optional[str] = None,
        total: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Create a progress bar within this context.

        Parameters
        ----------
        iterable : iterable, optional
            Iterable to wrap.
        desc : str, optional
            Description for the bar.
        total : int, optional
            Total number of items.
        **kwargs
            Additional arguments for tqdm.

        Returns
        -------
        Progress bar or iterable
        """
        bar = get_progress_bar(
            iterable=iterable,
            desc=desc,
            total=total,
            disable=not self.verbose,
            **kwargs
        )
        if hasattr(bar, '__enter__'):
            self.bars.append(bar)
        return bar

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close all progress bars
        for bar in self.bars:
            if hasattr(bar, 'close'):
                bar.close()
        return False
