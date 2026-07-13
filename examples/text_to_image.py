"""Text-to-image (T2I): generate a still image from a prompt."""

from __future__ import annotations

import time

from machgen.client import (
    ImageConfig,
    MachGenClient,
    TaskInput,
    TaskOutputType,
    TaskStatus,
)


def run(client: MachGenClient) -> str:
    task = TaskInput(
        prompt="An isometric illustration of a cozy reading nook, soft lighting",
        model="Nano-Banana-Pro",
        task_type="T2I",
        image_config=ImageConfig(aspect_ratio="1:1", height=1024),
    )

    handle = client.submit_task(task)

    result = client.get_task_state(handle)
    while result.status != TaskStatus.COMPLETED:
        time.sleep(2)
        result = client.get_task_state(handle)

    # task_output maps each output kind to its download locator; a T2I task
    # yields a single image. Use client.download_asset(handle.task_id) for bytes.
    assert result.task_output is not None
    return result.task_output[TaskOutputType.IMAGE]


if __name__ == "__main__":
    with MachGenClient() as client:
        print(run(client))
