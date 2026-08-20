"""HTTP client that calls the public ``/api/v0/*`` endpoints.

Authenticates via the ``MGA_<key_id>:<secret>`` API key — supplied directly or
read from the ``MACHGEN_API_KEY`` environment variable.
"""

from __future__ import annotations

import base64
import logging
import math
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
_LEGACY_UPLOAD_MAX_BYTES = 32 * 1024 * 1024

# A source ref is a public http(s):// URL or an inline data: URL - both
# forwarded untouched - or a local filesystem path, which is uploaded to the
# input bucket on submit.
_HTTP_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

# Raw-byte guard for inline_image_source, matching the server's per-image
# cap for inline data: sources (1 MB raw, ~1.37 MB encoded - base64 inflates
# by ~4/3).
_MAX_INLINE_SOURCE_BYTES = 1 * 1024 * 1024


def _is_http_url(ref: str) -> bool:
    return bool(_HTTP_URL_RE.match(ref))


def inline_image_source(path: str | Path) -> str:
    """Encode a local image file as an inline ``data:`` source ref.

    For the latency-sensitive ``POST /v0/generate/sync`` endpoint: an inline
    source rides in the request body, skipping the separate upload round trip.
    The server verifies the bytes, stores a durable copy, and the task echoes a
    normal ``@input/...`` ref afterward. Only the sync endpoint accepts inline
    sources; ``/v0/generate`` rejects them with a 400.

    Raises ValueError for a missing file, an oversize file (1 MB raw - the
    server's per-image cap for inline sources), or a file whose type cannot
    be inferred as an image.
    """
    file = Path(path)
    if not file.is_file():
        raise ValueError(f"Source path does not exist: {str(file.absolute())}")
    if file.stat().st_size > _MAX_INLINE_SOURCE_BYTES:
        raise ValueError(
            f"Inline source exceeds {_MAX_INLINE_SOURCE_BYTES // (1024 * 1024)} MB: "
            f"{str(file.absolute())}. Upload it instead (submit the local path)."
        )
    content_type, _ = mimetypes.guess_type(file.name)
    if not content_type or not content_type.startswith("image/"):
        raise ValueError(
            f"Inline sources must be images; could not infer an image type "
            f"from {file.name!r}."
        )
    encoded = base64.b64encode(file.read_bytes()).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


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
        - Free clip extraction from a completed owned video

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

    def extract_video_clip(
        self,
        source_task_id: str,
        start_secs: float,
        end_secs: float,
        *,
        on_update: UpdateCallback | None = None,
    ) -> TaskHandle:
        """Save part of an owned generated video as a new video asset.

        The operation preserves the source audio and uses the free
        ``EXTRACT_VIDEO_CLIP`` post-processing task rather than paid generation.
        ``start_secs`` is inclusive and ``end_secs`` is the exclusive endpoint.
        The server validates the range against the owned source's duration.
        """
        self._check_open()
        task_id = source_task_id.strip()
        if (
            not task_id
            or len(task_id) > 200
            or any(separator in task_id for separator in ("/", "\\", "?", "#"))
        ):
            raise ValueError("source_task_id must be a valid generated task id")
        if not math.isfinite(start_secs) or start_secs < 0:
            raise ValueError("start_secs must be a non-negative number")
        if not math.isfinite(end_secs) or end_secs <= start_secs:
            raise ValueError("end_secs must be greater than start_secs")

        resp = self._http.post(
            "/api/v0/generate",
            json={
                "prompt": f"Clip {start_secs:g}s to {end_secs:g}s",
                "model": "NO_MODEL",
                "task_type": "EXTRACT_VIDEO_CLIP",
                "src_video_urls": [f"/api/v0/assets/{task_id}"],
                "clip_start_secs": start_secs,
                "clip_end_secs": end_secs,
            },
        )
        resp.raise_for_status()
        body = GenerateResponse.model_validate(resp.json())
        return TaskHandle(body.task_id, self, on_update=on_update)

    def _upload_local_sources(self, task: TaskInput) -> TaskInput:
        """Replace every local-file source ref with an uploaded ``@input/...``
        ref, leaving http(s):// URLs untouched. Returns the task unchanged when
        it carries no local sources."""
        updates: dict[str, object] = {}
        for field in (
            "src_image_urls",
            "src_video_urls",
            "src_audio_urls",
            "src_file_urls",
        ):
            refs = getattr(task, field)
            if refs is not None:
                allow_direct_video = (
                    task.task_type in {"UPSCALE", "R2V"} and field == "src_video_urls"
                )
                updates[field] = [
                    self._resolve_source_ref(ref, allow_direct_video=allow_direct_video)
                    for ref in refs
                ]
        return task.model_copy(update=updates) if updates else task

    def _resolve_source_ref(self, ref: str, *, allow_direct_video: bool) -> str:
        if _is_http_url(ref):
            return ref
        if ref.startswith("data:"):
            # Inline sources (see inline_image_source) are forwarded verbatim;
            # the server decides where they are accepted (sync endpoint only).
            return ref
        path = Path(ref)
        if not path.is_file():
            raise ValueError(
                f"Source path does not exist: {str(path.absolute())}. Provide a path to a "
                "local file, a public http(s):// URL, or an inline data: URL."
            )
        return self._upload_local_file(path, allow_direct_video=allow_direct_video)

    def _upload_local_file(self, path: Path, *, allow_direct_video: bool) -> str:
        logging.info(f"Uploading input {path}")
        content_type, _ = mimetypes.guess_type(path.name)
        if path.stat().st_size > _LEGACY_UPLOAD_MAX_BYTES:
            if not allow_direct_video or not (content_type or "").startswith("video/"):
                raise ValueError(
                    "This local file is above the 32 MiB request-body limit and "
                    "the selected input does not support direct video upload"
                )
            return self._upload_direct_video(path, content_type or "video/mp4")
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

    def _send_storage_request(
        self, method: str, url: str, *, content: bytes, headers: dict[str, str]
    ) -> httpx.Response:
        request = self._http.build_request(
            method, url, content=content, headers=headers
        )
        request.headers.pop("Authorization", None)
        return self._http.send(request)

    @staticmethod
    def _resumable_offset(response: httpx.Response, fallback: int) -> int:
        match = re.search(r"bytes=\d+-(\d+)$", response.headers.get("Range", ""), re.I)
        return int(match.group(1)) + 1 if match else fallback

    def _query_resumable_offset(self, upload_url: str, total_bytes: int) -> int:
        response = self._send_storage_request(
            "PUT",
            upload_url,
            content=b"",
            headers={"Content-Range": f"bytes */{total_bytes}"},
        )
        if response.status_code in (200, 201):
            return total_bytes
        if response.status_code == 308:
            return self._resumable_offset(response, 0)
        response.raise_for_status()
        raise RuntimeError("Unexpected resumable upload status")

    def _upload_direct_video(self, path: Path, content_type: str) -> str:
        size_bytes = path.stat().st_size
        initiated = self._http.post(
            "/api/v0/uploads/direct/initiate",
            json={
                "file_name": path.name,
                "content_type": content_type,
                "size_bytes": size_bytes,
            },
        )
        initiated.raise_for_status()
        session = initiated.json()
        upload_url = str(session["upload_url"])
        chunk_size = int(session["chunk_size"])
        offset = 0
        with path.open("rb") as source:
            while offset < size_bytes:
                source.seek(offset)
                chunk = source.read(min(chunk_size, size_bytes - offset))
                end = offset + len(chunk)
                last_error: Exception | None = None
                for _ in range(3):
                    try:
                        response = self._send_storage_request(
                            "PUT",
                            upload_url,
                            content=chunk,
                            headers={
                                "Content-Type": content_type,
                                "Content-Range": f"bytes {offset}-{end - 1}/{size_bytes}",
                            },
                        )
                        if response.status_code in (200, 201):
                            offset = size_bytes
                        elif response.status_code == 308:
                            offset = self._resumable_offset(response, end)
                        else:
                            response.raise_for_status()
                        last_error = None
                        break
                    except (httpx.HTTPError, OSError) as error:
                        last_error = error
                        try:
                            offset = self._query_resumable_offset(
                                upload_url, size_bytes
                            )
                        except (httpx.HTTPError, OSError) as query_error:
                            last_error = query_error
                            continue
                        if offset >= size_bytes:
                            last_error = None
                            break
                        source.seek(offset)
                        chunk = source.read(min(chunk_size, size_bytes - offset))
                        end = offset + len(chunk)
                if last_error is not None:
                    raise last_error
        completed = self._http.post(
            "/api/v0/uploads/direct/complete",
            json={"upload_token": session["upload_token"]},
        )
        completed.raise_for_status()
        artifact_path = UploadResponse.model_validate(completed.json()).artifact_path
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
