"""Image editing (I2I): edit an existing image with a text instruction.

Source images may be public ``http(s)://`` URLs (forwarded as-is) or local file
paths (uploaded automatically on submit). This example uses a URL.
"""

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
        prompt="make the sky a dramatic sunset",
        model="Nano-Banana-Pro",
        task_type="I2I",
        src_image_urls=["https://example.com/landscape.png"],
        # or local file path:
        # src_image_urls=["/path/to/local/image.png"],
        image_config=ImageConfig(aspect_ratio="1:1", height=1024),
    )

    handle = client.submit_task(task)

    result = client.get_task_state(handle)
    while result.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
        time.sleep(2)
        result = client.get_task_state(handle)
    if result.status == TaskStatus.FAILED:
        raise RuntimeError(f"Generation failed: {result.error_msg}")

    assert result.task_output is not None
    return result.task_output[TaskOutputType.IMAGE]


if __name__ == "__main__":
    with MachGenClient() as client:
        print(run(client))
