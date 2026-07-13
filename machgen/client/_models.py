"""Wire-level response models shared between client and server.

The server ([frontend/api_server.py](frontend/api_server.py)) and the external
client both import these so the JSON schema stays in lock-step.

**Schema evolution policy.** Every model here sets ``extra="ignore"`` so an
older peer (client *or* server) silently drops fields it doesn't know about.
That means new fields can be added without coordinating a release across both
sides — but it requires:

- New fields are always **optional** (carry a default).
- Existing required fields are **never removed or renamed**.
- Enum members are **never removed** (additions are tolerated by the receiver
  only if the receiver doesn't dispatch on the enum value).

These rules are enforced by reviewer discipline, not by code; the test
[tests/api/test_wire_compat.py](tests/api/test_wire_compat.py) pins the
``extra="ignore"`` behavior so a regression flips a red light.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from machgen.client.api import TaskStatus

_WIRE_MODEL_CONFIG = ConfigDict(extra="ignore")


class TaskMetadata(BaseModel):
    model_config = _WIRE_MODEL_CONFIG

    prompt: str = Field(description="Prompt the task was submitted with.")
    model: str | None = Field(default=None, description="Resolved model id.")
    task_type: str | None = Field(default=None, description="Resolved task type.")
    seed: int | None = Field(default=None, description="Seed used for generation.")
    fps: int | None = Field(default=None, description="Frames per second (video).")
    height: int | None = Field(default=None, description="Output height in pixels.")
    width: int | None = Field(default=None, description="Output width in pixels.")
    aspect_ratio: str | None = Field(default=None, description="Output aspect ratio.")
    duration_secs: int | None = Field(
        default=None, description="Clip duration in seconds (video)."
    )
    src_image_urls: list[str] | None = Field(
        default=None, description="Source / reference image refs."
    )
    src_video_url: str | None = Field(
        default=None, description="Source video ref (video-to-video)."
    )
    acceleration: str | None = Field(
        default=None, description="Acceleration profile applied by the backend."
    )


class TaskOutputType(StrEnum):
    """A task may have 1 or more types of output."""

    TEXT = "text"
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"


class ModerationStage(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class ModerationResult(BaseModel):
    model_config = _WIRE_MODEL_CONFIG

    flagged: bool = Field(
        default=False, description="Whether moderation matched a block category."
    )
    stage: ModerationStage | None = Field(
        default=None, description="Which stage flagged (input or output)."
    )
    surface: str | None = Field(
        default=None, description='Surface that tripped: "text" or "image".'
    )
    categories: list[str] = Field(
        default_factory=list, description="Matched block categories."
    )
    source: str | None = Field(
        default=None,
        description='Who produced the verdict: "machgen" (OpenAI) or "vendor".',
    )
    moderation_time_secs: float | None = Field(
        default=None, description="Wall-clock time spent in the moderation call(s)."
    )


class GenerateResponse(BaseModel):
    model_config = _WIRE_MODEL_CONFIG

    task_id: str = Field(
        description="Server-assigned id; poll status and download the asset with it."
    )
    metadata: TaskMetadata = Field(description="Echo of the resolved task parameters.")
    task_output: dict[TaskOutputType, str] | None = Field(
        default=None,
        description="Per output-type asset locators, as they become available.",
    )
    moderation: ModerationResult | None = Field(
        default=None,
        description="Set when moderation rejected the request at admission. The task was admitted then immediately failed (refunded); no generation work was done.",
    )


class TaskStatusResponse(GenerateResponse):
    status: TaskStatus = Field(description="Current task status.")
    error_msg: str | None = Field(
        default=None, description="Failure reason, set when status is FAILED."
    )
    generation_time_secs: float | None = Field(
        default=None, description="Time spent on actual generation."
    )
    upload_time_secs: float | None = Field(
        default=None, description="Time spent uploading the asset after generation."
    )
    queue_time_secs: float | None = Field(
        default=None,
        description="Time spent waiting in the queue for generation to begin.",
    )
    moderation: ModerationResult | None = Field(
        default=None,
        description="Structured moderation verdict when the task failed or was blocked by moderation.",
    )


class UploadResponse(BaseModel):
    model_config = _WIRE_MODEL_CONFIG

    artifact_path: str = Field(
        description="Storage path of the uploaded input; reference it as @input/<artifact_path>."
    )
