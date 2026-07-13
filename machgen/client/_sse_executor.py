"""Process-global thread pool for ``TaskHandle`` SSE consumers.

All handles across all ``MachGenClient`` instances submit their stream
consumption work to one ``ThreadPoolExecutor`` so the per-process worker
count stays bounded — the alternative (one ad-hoc daemon thread per
handle) blows up when callers spin up many clients with many handles.

Cap: ``MACHGEN_CLIENT_MAX_SSE_WORKERS`` env var (default 256). The pool
grows lazily up to that cap.

Atexit handling: at program shutdown we signal stop on every live stream
state before the pool's own atexit hook fires, so workers blocked on
``iter_sse()`` are unstuck and the pool's join doesn't hang. Atexit is
LIFO and the pool's hook is registered the first time a
``ThreadPoolExecutor`` is created — we register ours after that, so ours
runs first.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from machgen.client.task_handle import _StreamState

logger = logging.getLogger(__name__)

_DEFAULT_MAX_WORKERS = int(os.environ.get("MACHGEN_CLIENT_MAX_SSE_WORKERS", "256"))

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()
_active_states: "weakref.WeakSet[_StreamState]" = weakref.WeakSet()
_active_lock = threading.Lock()


def _ensure_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is not None:
        return _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=_DEFAULT_MAX_WORKERS,
                thread_name_prefix="machgen-sse",
            )
            atexit.register(_atexit_shutdown)
    return _executor


def submit(fn: Callable[..., None], *args, **kwargs) -> Future:
    """Submit an SSE consumer function to the shared pool."""
    return _ensure_executor().submit(fn, *args, **kwargs)


def register_state(state: "_StreamState") -> None:
    """Track *state* so atexit can stop its worker before pool shutdown."""
    with _active_lock:
        _active_states.add(state)


def _atexit_shutdown() -> None:
    """Signal stop on every live state, then shut down the pool.

    Runs before ``concurrent.futures.thread._python_exit`` (LIFO atexit
    order). ``signal_stop`` wakes any worker blocked on ``iter_sse()``;
    the pool's subsequent join then proceeds promptly.
    """
    with _active_lock:
        states = list(_active_states)
    for state in states:
        try:
            state.signal_stop(terminal_message="process exiting")
        except Exception:
            logger.warning("error stopping SSE worker at exit", exc_info=True)
    if _executor is not None:
        _executor.shutdown(wait=False)
