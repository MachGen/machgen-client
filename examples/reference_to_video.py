"""Reference-to-video (R2V): drive a video from named subject references.

``subject_to_image_ids`` maps a subject name to the indices of its reference
images in ``src_image_urls``. The prompt can then address a subject via
``@name`` (honored by vendors with named subjects, e.g. Vidu reference2video).
"""

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
        prompt="@alice waves to @bob across a busy street",
        model="Vidu-Q3",
        task_type="R2V",
        src_image_urls=[
            "https://example.com/alice-1.png",
            "https://example.com/alice-2.png",
            "https://example.com/bob.png",
        ],
        subject_to_image_ids={"alice": [0, 1], "bob": [2]},
        video_config=VideoConfig(duration_secs=5, aspect_ratio="16:9"),
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
