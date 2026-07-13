"""Quickstart: generate a video from a text prompt, then download it.

Run it with your API key in the environment::

    export MACHGEN_API_KEY="MGA_<key_id>:<secret>"
    python -m examples.quickstart
"""

from __future__ import annotations

import time

from machgen.client import MachGenClient, TaskInput, TaskStatus, VideoConfig


def run(client: MachGenClient) -> bytes:
    task = TaskInput(
        prompt="A red panda exploring a misty forest at dawn",
        model="Wan2.2-T2V-A14B",
        task_type="T2V",
        video_config=VideoConfig(
            duration_secs=5,
            height=480,
            aspect_ratio="16:9",
            fps=16,
        ),
    )

    handle = client.submit_task(task)

    # Poll until the task reaches a terminal status.
    result = client.get_task_state(handle)
    while result.status != TaskStatus.COMPLETED:
        time.sleep(2)
        result = client.get_task_state(handle)

    return client.download_asset(handle.task_id)


if __name__ == "__main__":
    with MachGenClient() as client:
        video = run(client)
    with open("output.mp4", "wb") as f:
        f.write(video)
    print(f"Saved {len(video)} bytes to output.mp4")
