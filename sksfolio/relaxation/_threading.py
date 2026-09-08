"""Cached BLAS thread-limit control for the first-order solvers.

``threadpoolctl.threadpool_limits`` and ``threadpoolctl.threadpool_info``
rediscover every loaded native library on each call. On Windows that walk is
an avoidable constant per-solve cost, which matters most at small dimensions
and inside branch-and-bound where the relaxation is solved repeatedly.

Using a process-wide :class:`threadpoolctl.ThreadpoolController` pays that
discovery cost once and reuses it for subsequent limit and info queries.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, ContextManager, List


_CONTROLLER: Any = None
_CONTROLLER_FAILED = False


def _controller() -> Any:
    """Return the process-wide controller, or ``None`` if unavailable."""
    global _CONTROLLER, _CONTROLLER_FAILED
    if _CONTROLLER is not None or _CONTROLLER_FAILED:
        return _CONTROLLER
    try:
        from threadpoolctl import ThreadpoolController
    except ImportError:
        _CONTROLLER_FAILED = True
        return None
    _CONTROLLER = ThreadpoolController()
    return _CONTROLLER


def available() -> bool:
    """Return whether thread limits can be enforced in this process."""
    return _controller() is not None


def limit_blas_threads(threads: int) -> ContextManager[Any]:
    """Limit BLAS threads for the duration of the returned context."""
    threads = int(threads)
    if threads <= 0:
        return nullcontext()
    controller = _controller()
    if controller is None:
        raise RuntimeError(
            "threadpoolctl is required to enforce solver thread limits"
        )
    return controller.limit(limits=threads, user_api="blas")


def threadpool_info() -> List[dict]:
    """Return the current thread-pool description without rediscovery."""
    controller = _controller()
    if controller is None:
        return []
    return controller.info()


__all__ = ["available", "limit_blas_threads", "threadpool_info"]
