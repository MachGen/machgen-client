"""Video upscaling (UPSCALE): upscale a completed video task to 4K."""

from __future__ import annotations

import time

from machgen.client import (
    MachGenClient,
    TaskInput,
    TaskOutputType,
    TaskStatus,
    UpscaleConfig,
    VideoConfig,
)


def run(client: MachGenClient) -> str:
    task = TaskInput(
        prompt="",
        model="Topaz-Video-Precision",
        task_type="UPSCALE",
        src_task_ids=["<completed-video-task-id>"],
        upscale_config=UpscaleConfig(engine="proteus"),
        video_config=VideoConfig(height=2160, duration_secs=-1),
    )

    handle = client.submit_task(task)

    result = client.get_task_state(handle)
    while result.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
        time.sleep(2)
        result = client.get_task_state(handle)
    if result.status == TaskStatus.FAILED:
        raise RuntimeError(f"Upscaling failed: {result.error_msg}")

    assert result.task_output is not None
    return result.task_output[TaskOutputType.VIDEO]


if __name__ == "__main__":
    with MachGenClient() as client:
        print(run(client))
