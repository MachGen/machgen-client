"""HTTP client that calls the public ``/api/v0/*`` endpoints.

Authenticates via the ``MGA_<key_id>:<secret>`` API key — supplied directly or
read from the ``MACHGEN_API_KEY`` environment variable.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
import threading
import weakref
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Self

import httpx

from machgen.client import _sse_executor
from machgen.client._models import (
    GenerateResponse,
    TaskStatusResponse,
    UploadResponse,
)
from machgen.client.api import TaskInput
from machgen.client.task_handle import (
    TaskHandle,
    UpdateCallback,
    _run_worker,
    _StreamState,
)

_DEFAULT_BASE_URL = "https://api.machgen.ai"
_DEFAULT_TIMEOUT_SECS = 60.0

# A source ref is either a public http(s):// URL - forwarded untouched - or a
# local filesystem path, which is uploaded to the input bucket on submit.
_HTTP_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _is_http_url(ref: str) -> bool:
    return bool(_HTTP_URL_RE.match(ref))


class SseRetryConfig:
    """
    Reconnect policy applied when the SSE update stream drops.
    """

    __slots__ = (
        "max_attempts",
        "initial_backoff_secs",
        "max_backoff_secs",
        "multiplier",
    )

    def __init__(
        self,
        *,
        max_attempts: int = 5,
        initial_backoff_secs: float = 1.0,
        max_backoff_secs: float = 30.0,
        multiplier: float = 2.0,
    ) -> None:
        if max_attempts < 0:
            raise ValueError("max_attempts must be non-negative")
        if initial_backoff_secs <= 0 or max_backoff_secs <= 0 or multiplier <= 0:
            raise ValueError("backoff parameters must be positive")
        self.max_attempts = max_attempts
        self.initial_backoff_secs = initial_backoff_secs
        self.max_backoff_secs = max_backoff_secs
        self.multiplier = multiplier


class MachGenClient:
    """
    Public client to interact with MachGen API service.

    The client currently supports:
        - Image generation
            - T2I (Text to Image)
            - I2I (Image Editing)
        - Video generation
            - T2V (Text to Video)
            - I2V (Image to Video)
            - R2V (Reference to Video)

    The client uses a polling model. The caller can submit a video or image task.

    The submission call does not block waiting for the task to complete.
    It returns a handle to be used for state polling.

    A caller is expected to call one of
        - `client.get_task_state(handle)`
        - `handle.state`
        - `client.wait(handle)` (blocking)
    to get the state/wait for completion of the task.

    ```
    task = TaskInput(
        prompt="A quick brown fox jumps over the lazy dog.",
        model="Wan2.2-A14B",
        task_type="T2V",
        video_config=VideoConfig(
            fps=16,
            height=480,
            aspect_ratio="16:9",
            duration_secs=5,
        ),
    )

    with MachGenClient() as client:
        handle = client.submit_task(task)

        # optionally, add a callback to suscribe to updates
        # handle = client.submit_task(task, on_update=lambda status: ...)

        result = client.get_task_state(handle)
        while result.status != TaskStatus.COMPLETED:
            time.sleep(1)
            result = client.get_task_state(handle)

        # alternatively use a blocking wait:
        # result = client.wait(handle)
        assert result.status == TaskStatus.COMPLETED

        with open(output_path, "wb") as f:
            f.write(client.download_asset(handle.task_id))
    ```
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECS,
        sse_retry: SseRetryConfig | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        resolved_base_url = (
            base_url or os.environ.get("MACHGEN_API_URL") or _DEFAULT_BASE_URL
        )
        resolved_api_key = api_key or os.environ.get("MACHGEN_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "MachGenClient requires an api_key — pass api_key=... or set "
                "MACHGEN_API_KEY in the environment."
            )

        self._http = httpx.Client(
            base_url=resolved_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {resolved_api_key}"},
            timeout=timeout,
            transport=transport,
        )
        self._sse_retry = sse_retry or SseRetryConfig()
        self._closed = False
        # Track every state whose worker is still using ``self._http`` so
        # ``close()`` can stop them before the transport goes away. WeakSet
        # so finished handles drop out automatically once their state is GC'd.
        self._active_states: weakref.WeakSet[_StreamState] = weakref.WeakSet()
        self._states_lock = threading.Lock()

    @property
    def sse_retry(self) -> SseRetryConfig:
        return self._sse_retry

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN001
        self.close()

    def close(self) -> None:
        """Stop every in-flight handle, then tear down the connection pool.

        Any handle still streaming has its worker signaled to stop and its
        terminal state forced to a ``RuntimeError`` so a concurrent
        ``client.wait(handle)`` unblocks. Subsequent ``submit_task`` /
        ``get_task_status`` / ``download_asset`` raise.
        """
        with self._states_lock:
            if self._closed:
                return
            self._closed = True
            states = list(self._active_states)

        for state in states:
            state.signal_stop(
                terminal_message=(
                    f"MachGenClient closed before task {state.task_id} reached "
                    "terminal status"
                )
            )
        for state in states:
            fut = state.future
            if fut is None or fut.done():
                continue
            try:
                fut.result(timeout=2.0)
            except FutureTimeoutError:
                pass
            except Exception:
                # Worker exit exceptions are already captured in terminal_exc.
                pass

        self._http.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(
                "MachGenClient is closed; create a new client to issue requests"
            )

    def _ensure_worker(self, handle: TaskHandle) -> None:
        """Start *handle*'s SSE consumer if not already running. Idempotent.

        Streaming is lazy — submitted on the first of: an ``on_update``
        callback registered at construction, or a ``client.wait(handle)``
        call. A handle that is only ever polled via ``get_task_status``
        never opens an SSE stream.

        Registration into ``_active_states`` and the executor submit happen
        atomically under ``_states_lock`` so a concurrent ``close()`` either
        sees the new state in its snapshot or rejects this call via
        ``_check_open``; the worker can't start on a torn-down transport.
        """
        state = handle._state
        if state.future is not None:
            return  # fast path: worker already running or finished
        with state.lock:
            if state.future is not None:
                return
            with self._states_lock:
                self._check_open()
                self._active_states.add(state)
            _sse_executor.register_state(state)
            state.future = _sse_executor.submit(_run_worker, self._http, state)

    # ── Public API ───────────────────────────────────────────────────

    def submit_task(
        self,
        task: TaskInput,
        *,
        on_update: UpdateCallback | None = None,
    ) -> TaskHandle:
        """
        Submit a task and get a handle.

        If the task input has source references,
        each source ref is either a public http(s):// URL,
        or a local file path, which gets uploaded on submission.

        Args:
            task: the task input to submit
            on_update: optional callback to receive task status updates

        Returns:
            a handle to the submitted task, which can be used to poll for status
        """
        self._check_open()
        task = self._upload_local_sources(task)
        resp = self._http.post(
            "/api/v0/generate",
            json=task.model_dump(mode="json", exclude_none=True),
        )
        resp.raise_for_status()
        body = GenerateResponse.model_validate(resp.json())
        return TaskHandle(body.task_id, self, on_update=on_update)

    def _upload_local_sources(self, task: TaskInput) -> TaskInput:
        """Replace every local-file source ref with an uploaded ``@input/...``
        ref, leaving http(s):// URLs untouched. Returns the task unchanged when
        it carries no local sources."""
        updates: dict[str, object] = {}
        if task.src_image_urls is not None:
            updates["src_image_urls"] = [
                self._resolve_source_ref(u) for u in task.src_image_urls
            ]
        return task.model_copy(update=updates) if updates else task

    def _resolve_source_ref(self, ref: str) -> str:
        if _is_http_url(ref):
            return ref
        path = Path(ref)
        if not path.is_file():
            raise ValueError(
                f"Source path does not exist: {str(path.absolute())}. Provide a path to a "
                "local file or a public http(s):// URL."
            )
        return self._upload_local_file(path)

    def _upload_local_file(self, path: Path) -> str:
        logging.info(f"Uploading input {path}")
        content_type, _ = mimetypes.guess_type(path.name)
        resp = self._http.post(
            "/api/v0/upload",
            files={
                "file": (
                    path.name,
                    path.read_bytes(),
                    content_type or "application/octet-stream",
                )
            },
        )
        resp.raise_for_status()
        artifact_path = UploadResponse.model_validate(resp.json()).artifact_path
        return f"@input/{artifact_path}"

    def wait(self, handle: TaskHandle, timeout: float = 300.0) -> TaskStatusResponse:
        """
        Blocks the caller until the handle reaches a terminal status.

        Returns:
            the final task status response
        """
        self._ensure_worker(handle)
        state = handle._state
        if not state.terminal_event.wait(timeout=timeout):
            raise TimeoutError(
                f"Task {handle.task_id} did not complete within {timeout}s"
            )
        if state.terminal_exc is not None:
            raise state.terminal_exc
        assert state.terminal_resp is not None
        return state.terminal_resp

    def get_task_state(self, handle: TaskHandle) -> TaskStatusResponse:
        """
        Get the task's current state.

        This is similar to `handle.state` except that it eagerly fetches the state from server,
        and raises exception if the state is not available yet,
        as opposed to waiting for server polling.

        Returns:
            the current task state
        """
        self._check_open()
        resp = self._http.get(f"/api/v0/tasks/{handle.task_id}")
        resp.raise_for_status()
        return TaskStatusResponse.model_validate(resp.json())

    def download_asset(self, task_id: str) -> bytes:
        self._check_open()
        resp = self._http.get(f"/api/v0/assets/{task_id}")
        resp.raise_for_status()
        return resp.content
