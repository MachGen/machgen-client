"""
Handle returned by ``MachGenClient.submit_task``.
"""

from __future__ import annotations

import logging
import threading
import weakref
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING

import httpx
from httpx_sse import SSEError, connect_sse

from machgen.client._models import TaskStatusResponse
from machgen.client.api import TaskStatus

if TYPE_CHECKING:
    from machgen.client.client import MachGenClient, SseRetryConfig

logger = logging.getLogger(__name__)

# HTTP statuses that should propagate to the caller without retrying — they
# indicate a problem the client cannot recover from by reconnecting.
_NON_RETRYABLE_STATUSES = frozenset({401, 403, 404})


UpdateCallback = Callable[[TaskStatusResponse], None]


class _StreamState:
    """Mutable state shared between a ``TaskHandle`` and its background worker.

    Lives independently of the user-facing handle so that dropping the handle
    can run a finalizer that signals the worker to stop, instead of being
    pinned alive by a worker bound-method reference.
    """

    def __init__(
        self,
        task_id: str,
        sse_retry: SseRetryConfig,
        callback: UpdateCallback | None,
    ) -> None:
        self.task_id = task_id
        self.sse_retry = sse_retry

        self.lock = threading.Lock()
        self.update_callback: UpdateCallback | None = callback
        self.latest: TaskStatusResponse | None = None
        self.active_response: httpx.Response | None = None

        self.terminal_event = threading.Event()
        self.terminal_resp: TaskStatusResponse | None = None
        self.terminal_exc: BaseException | None = None

        self.stop_event = threading.Event()

        # The worker future, attached by ``TaskHandle`` after submitting to
        # the executor pool. Used by ``MachGenClient.close()`` to wait briefly
        # for the worker to wind down before tearing the http transport.
        self.future: Future | None = None

    def signal_stop(self, *, terminal_message: str | None = None) -> None:
        if not self.terminal_event.is_set():
            with self.lock:
                if self.terminal_exc is None and self.terminal_resp is None:
                    self.terminal_exc = RuntimeError(
                        terminal_message
                        or f"TaskHandle for {self.task_id} closed before terminal status"
                    )
            self.terminal_event.set()
        self.stop_event.set()
        with self.lock:
            resp = self.active_response
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass

    def dispatch(self, resp: TaskStatusResponse) -> None:
        """Record the new payload and fan out to registered callbacks."""
        with self.lock:
            self.latest = resp
        if self.update_callback:
            _safe_invoke(self.update_callback, resp, self.task_id)

    def set_terminal(
        self,
        resp: TaskStatusResponse | None,
        *,
        exc: BaseException | None = None,
    ) -> None:
        with self.lock:
            if resp is not None:
                self.terminal_resp = resp
            self.terminal_exc = exc
        self.terminal_event.set()


def _safe_invoke(cb: UpdateCallback, resp: TaskStatusResponse, task_id: str) -> None:
    try:
        cb(resp)
    except Exception:
        logger.exception(f"on_update callback raised for task {task_id}")


class TaskHandle:
    def __init__(
        self,
        task_id: str,
        client: MachGenClient,
        *,
        on_update: UpdateCallback | None = None,
    ) -> None:
        self._task_id = task_id
        self._state = _StreamState(
            task_id=task_id,
            sse_retry=client.sse_retry,
            callback=on_update,
        )
        if on_update is not None:
            client._ensure_worker(self)
        weakref.finalize(self, _StreamState.signal_stop, self._state)

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def state(self) -> TaskStatusResponse | None:
        """
        Fetches the latest state of the handle.

        Returns:
            the current task state, or None if there's no state from server yet
        """
        return self._state.terminal_resp


# ── Worker (free functions; do not capture a TaskHandle reference) ────────


def _run_worker(http: httpx.Client, state: _StreamState) -> None:
    """Pool worker entry point: open SSE stream, consume, reconnect on drop."""
    try:
        _run_inner(http, state)
    finally:
        # Make sure terminal_event is set so wait() never hangs even on
        # an unexpected exception path.
        if not state.terminal_event.is_set():
            state.set_terminal(
                None,
                exc=RuntimeError(
                    f"SSE worker for task {state.task_id} exited unexpectedly"
                ),
            )


def _run_inner(http: httpx.Client, state: _StreamState) -> None:
    retry = state.sse_retry
    attempt = 0
    backoff = retry.initial_backoff_secs
    while not state.stop_event.is_set():
        try:
            _consume_one_stream(http, state)
            return
        except _StreamDropped as e:
            if state.stop_event.is_set():
                return
            if attempt >= retry.max_attempts:
                state.set_terminal(
                    None,
                    exc=RuntimeError(
                        f"SSE stream for task {state.task_id} dropped and "
                        f"reconnect attempts exhausted "
                        f"({retry.max_attempts}): {e.reason}"
                    ),
                )
                return
            logger.warning(
                "SSE stream for task %s dropped (%s); reconnect %d/%d in %.1fs",
                state.task_id,
                e.reason,
                attempt + 1,
                retry.max_attempts,
                backoff,
            )
            # Sleep on stop_event so close() interrupts immediately.
            if state.stop_event.wait(timeout=backoff):
                return
            attempt += 1
            backoff = min(backoff * retry.multiplier, retry.max_backoff_secs)
        except httpx.HTTPStatusError as e:
            state.set_terminal(None, exc=e)
            return
        except Exception as e:
            # A close() racing with an in-flight iter_raw() raises
            # httpx.StreamClosed here; treat it like the _StreamDropped
            # branch above and keep the terminal state signal_stop() set.
            if state.stop_event.is_set():
                return
            logger.exception("Unexpected error in SSE consumer thread")
            state.set_terminal(None, exc=e)
            return


def _consume_one_stream(http: httpx.Client, state: _StreamState) -> None:
    """Open one SSE stream and consume until terminal or drop.

    Sets ``terminal_*`` and returns when terminal is reached. Raises
    :class:`_StreamDropped` if the connection died mid-stream (recoverable).
    Raises ``httpx.HTTPStatusError`` on non-retryable HTTP errors
    (401/403/404) — these propagate to ``_run_inner`` which surfaces them
    to ``client.wait(handle)``.
    """
    try:
        with connect_sse(
            http,
            "GET",
            f"/api/v0/tasks/{state.task_id}/updates",
        ) as event_source:
            response = event_source.response
            if response.status_code in _NON_RETRYABLE_STATUSES:
                response.raise_for_status()
            if response.status_code >= 400:
                raise _StreamDropped(reason=f"HTTP {response.status_code}", cause=None)

            with state.lock:
                state.active_response = response
            # Tight race: ``signal_stop`` runs ``stop_event.set()`` before
            # peeking at ``active_response`` under the lock. If it fired
            # between us getting the response and registering it, it saw
            # ``None`` and skipped the close. Re-check the flag so we bail
            # before blocking in ``iter_sse`` waiting for data the server
            # may never send (heartbeat-only streams in particular).
            if state.stop_event.is_set():
                return
            try:
                for sse in event_source.iter_sse():
                    if state.stop_event.is_set():
                        return
                    if sse.event == "idle_timeout":
                        state.set_terminal(
                            None,
                            exc=TimeoutError(
                                f"Task {state.task_id} idle for too long; "
                                "server closed stream"
                            ),
                        )
                        return
                    resp = TaskStatusResponse.model_validate_json(sse.data)
                    state.dispatch(resp)
                    if resp.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                        state.set_terminal(resp)
                        return
            finally:
                with state.lock:
                    state.active_response = None
    except (httpx.TransportError, httpx.RemoteProtocolError, SSEError) as e:
        raise _StreamDropped(reason=type(e).__name__, cause=e) from e

    # Iter exhausted without a terminal payload — server likely restarted.
    raise _StreamDropped(reason="stream ended without terminal status", cause=None)


class _StreamDropped(Exception):
    """Internal signal that the SSE stream dropped and a reconnect may help."""

    def __init__(self, *, reason: str, cause: BaseException | None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.cause = cause
