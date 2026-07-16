"""Image-to-video (I2V): animate a still image into a short clip."""

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
        prompt="slow dolly-in as the leaves drift in the wind",
        model="Vidu-Q3-Turbo",
        task_type="I2V",
        src_image_urls=["https://example.com/first-frame.png"],
        video_config=VideoConfig(duration_secs=5, height=720),
    )

    handle = client.submit_task(task)

    result = client.get_task_state(handle)
    while result.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
        time.sleep(2)
        result = client.get_task_state(handle)
    if result.status == TaskStatus.FAILED:
        raise RuntimeError(f"Generation failed: {result.error_msg}")

    assert result.task_output is not None
    return result.task_output[TaskOutputType.VIDEO]


if __name__ == "__main__":
    with MachGenClient() as client:
        print(run(client))
