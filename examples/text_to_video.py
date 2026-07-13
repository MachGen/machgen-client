"""Text-to-video (T2V): generate a short clip from a text prompt."""

from __future__ import annotations

import time

from machgen.client import (
    MachGenClient,
    TaskInput,
    TaskOutputType,
    TaskStatus,
    VideoConfig,
)


def run(client: MachGenClient) -> str:
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

    result = client.get_task_state(handle)
    while result.status != TaskStatus.COMPLETED:
        time.sleep(2)
        result = client.get_task_state(handle)

    assert result.task_output is not None
    return result.task_output[TaskOutputType.VIDEO]


if __name__ == "__main__":
    with MachGenClient() as client:
        print(run(client))
