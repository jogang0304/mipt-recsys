"""Thread settings scoped to a fit or evaluation call; arrays remain shared in RAM."""

from contextlib import contextmanager
import os
from threading import RLock

import numba
from threadpoolctl import threadpool_limits


_THREAD_SETTINGS_LOCK = RLock()


def cpu_count():
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    requested = int(os.environ.get("RECSYS_NUM_THREADS", available))
    if requested < 1:
        raise ValueError("RECSYS_NUM_THREADS must be positive")
    return min(requested, available)


@contextmanager
def training_threads(*, parallel=False, torch_model=False):
    """Avoid nested BLAS pools when Numba already distributes rows across cores."""
    # Notebook grid searches can call fit/evaluation from multiple Python threads.
    # BLAS/PyTorch settings are process-wide, so their scoped changes must not overlap.
    with _THREAD_SETTINGS_LOCK:
        workers = cpu_count()
        old_numba = numba.get_num_threads()
        if torch_model:
            import torch

            old_torch = torch.get_num_threads()
            # Small embedding batches are slower with all SMT threads on this workload.
            torch_workers = int(os.environ.get("RECSYS_TORCH_THREADS", min(workers, 8)))
            if torch_workers < 1:
                raise ValueError("RECSYS_TORCH_THREADS must be positive")
        try:
            numba.set_num_threads(min(workers, numba.config.NUMBA_NUM_THREADS))
            if torch_model:
                torch.set_num_threads(min(workers, torch_workers))
            with threadpool_limits(limits=1 if parallel else workers, user_api="blas"):
                yield
        finally:
            numba.set_num_threads(old_numba)
            if torch_model:
                torch.set_num_threads(old_torch)
