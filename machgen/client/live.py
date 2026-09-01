"""Client helpers for the public real-time LiveSession API."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Literal, Self

import httpx
from pydantic import BaseModel
from websockets.sync.client import ClientConnection, connect


class LiveAvatar(BaseModel):
    image_url: str
    persona: str
    name: str | None = None
    voice: str | None = None


class LiveRtcCredentials(BaseModel):
    app_id: str
    channel_id: str
    user_id: str
    token: str
    token_expire_at: int


class LiveSession(BaseModel):
    id: str
    model: str
    status: str
    call_mode: Literal["audio", "video"]
    max_session_seconds: int
    provider_max_session_seconds: int | None = None
    unit_price_micros: int
    authorized_cost_micros: int
    billed_seconds: int | None = None
    settled_cost_micros: int | None = None
    hangup_reason: str | None = None
    error_message: str | None = None
    created_at: str
    started_at: str | None = None
    ended_at: str | None = None


class CreatedLiveSession(LiveSession):
    rtc: LiveRtcCredentials
    control_url: str
    control_token: str
    control_token_expires_at: int


class LiveControlEvent(BaseModel):
    type: str
    status: str | None = None
    reason: str | None = None
    code: str | None = None
    retry_in_seconds: int | None = None
    max_session_seconds: int | None = None


class LiveControl(AbstractContextManager["LiveControl"]):
    def __init__(self, control_url: str, control_token: str) -> None:
        self._url = control_url
        self._token = control_token
        self._socket: ClientConnection | None = None
        self._ended = False

    def __enter__(self) -> Self:
        self._socket = connect(self._url, ping_interval=5, ping_timeout=15)
        self._socket.send(json.dumps({"type": "authenticate", "token": self._token}))
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN001
        if self._socket is not None:
            if not self._ended:
                self.end()
            self._socket.close()
            self._socket = None

    def receive(self, timeout: float | None = None) -> LiveControlEvent:
        socket = self._require_socket()
        raw = socket.recv(timeout=timeout)
        if not isinstance(raw, str):
            raise RuntimeError("MachGen live control returned a binary message")
        return LiveControlEvent.model_validate_json(raw)

    def wait_until_live(self, timeout: float = 30.0) -> LiveControlEvent:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Live session did not become ready in time")
            event = self.receive(timeout=remaining)
            if event.type == "error":
                raise RuntimeError(event.code or "Live control failed")
            if event.type == "session_state" and event.status == "live":
                return event
            if event.type == "session_state" and event.status == "ended":
                raise RuntimeError(
                    f"Live session ended before it became ready: {event.reason or 'unknown'}"
                )

    def end(self) -> None:
        if self._ended:
            return
        self._require_socket().send(json.dumps({"type": "end"}))
        self._ended = True

    def _require_socket(self) -> ClientConnection:
        if self._socket is None:
            raise RuntimeError("Use LiveControl as a context manager")
        return self._socket


class LiveClient:
    def __init__(self, http: httpx.Client, check_open: Callable[[], None]) -> None:
        self._http = http
        self._check_open = check_open

    def create(
        self,
        *,
        avatar: LiveAvatar | dict,
        model: Literal["Vidu-S1"] = "Vidu-S1",
        call_mode: Literal["audio", "video"] = "video",
        max_session_seconds: int = 60,
    ) -> CreatedLiveSession:
        self._check_open()
        avatar_model = (
            avatar
            if isinstance(avatar, LiveAvatar)
            else LiveAvatar.model_validate(avatar)
        )
        response = self._http.post(
            "/api/v0/live/sessions",
            json={
                "model": model,
                "call_mode": call_mode,
                "max_session_seconds": max_session_seconds,
                "avatar": avatar_model.model_dump(exclude_none=True),
            },
        )
        response.raise_for_status()
        return CreatedLiveSession.model_validate(response.json())

    def get(self, session_id: str) -> LiveSession:
        self._check_open()
        response = self._http.get(f"/api/v0/live/sessions/{session_id}")
        response.raise_for_status()
        return LiveSession.model_validate(response.json())

    def list(self, *, limit: int = 50, offset: int = 0) -> list[LiveSession]:
        self._check_open()
        response = self._http.get(
            "/api/v0/live/sessions", params={"limit": limit, "offset": offset}
        )
        response.raise_for_status()
        return [LiveSession.model_validate(row) for row in response.json()["sessions"]]

    def control(self, session: CreatedLiveSession) -> LiveControl:
        self._check_open()
        return LiveControl(session.control_url, session.control_token)

    def rotate_control_token(self, session_id: str) -> tuple[str, int]:
        self._check_open()
        response = self._http.post(f"/api/v0/live/sessions/{session_id}/control-token")
        response.raise_for_status()
        body = response.json()
        return str(body["control_token"]), int(body["control_token_expires_at"])

    def end(self, session_id: str) -> LiveSession:
        self._check_open()
        response = self._http.post(f"/api/v0/live/sessions/{session_id}/end")
        response.raise_for_status()
        return LiveSession.model_validate(response.json())
