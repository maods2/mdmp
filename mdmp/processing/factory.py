"""
Factory for creating appropriate processor instances.
"""

from typing import Optional

from ..parallel import _get_n_jobs
from .base import NodeProcessor
from .parallel import ParallelProcessor
from .serial import SerialProcessor


def create_processor(
    n_jobs: Optional[int] = None,
    verbose: bool = False,
    default_n_jobs: int = 1
) -> NodeProcessor:
    """
    Create appropriate processor based on n_jobs parameter.

    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel jobs. If None or 1, creates serial processor.
        If > 1, creates parallel processor.
    verbose : bool, optional
        Whether to show progress bars. Default is False.
    default_n_jobs : int, optional
        Default number of jobs if n_jobs is None. Default is 1.

    Returns
    -------
    NodeProcessor
        Serial or parallel processor instance.
    """
    n_jobs_actual = _get_n_jobs(n_jobs, default=default_n_jobs)

    if n_jobs_actual == 1:
        return SerialProcessor(verbose=verbose)
    else:
        return ParallelProcessor(n_jobs=n_jobs_actual, verbose=verbose)
