"""Vidu S1: create and control a real-time avatar session.

Join ``session.rtc`` with AliRTC before opening the control channel. The control
acknowledgement starts provider billing, so this example waits for an explicit
confirmation before connecting.
"""

from __future__ import annotations

import os

from machgen.client import CreatedLiveSession, MachGenClient

DEFAULT_AVATAR_IMAGE_URL = "https://www.machgen.ai/static/vidu-s1-photoreal-avatar.webp"


def run(client: MachGenClient) -> CreatedLiveSession:
    avatar_image_url = os.environ.get(
        "MACHGEN_AVATAR_IMAGE_URL", DEFAULT_AVATAR_IMAGE_URL
    )
    session = client.live.create(
        model="Vidu-S1",
        call_mode="video",
        max_session_seconds=60,
        avatar={
            "image_url": avatar_image_url,
            "persona": "A friendly product specialist",
            "name": "MachGen Guide",
        },
    )
    return session


def control_session(client: MachGenClient, session: CreatedLiveSession) -> None:
    print("Join this AliRTC channel before continuing:")
    print(session.rtc.model_dump_json(indent=2))
    input("Press Enter after the RTC client has joined. Billing starts next. ")
    with client.live.control(session) as control:
        control.wait_until_live()
        input("The avatar is live. Press Enter to end the session. ")
        control.end()


if __name__ == "__main__":
    with MachGenClient() as machgen:
        created = run(machgen)
        control_session(machgen, created)
