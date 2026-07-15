from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

_WIRE_MODEL_CONFIG = ConfigDict(extra="ignore", frozen=True)


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    # Generation + upload both done; task_output is populated. Terminal.
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class VideoConfig(BaseModel):
    model_config = _WIRE_MODEL_CONFIG

    fps: int | None = Field(
        default=None,
        description="Video frames per second. If omitted the default FPS would be used based on the model.",
    )
    duration_secs: int = Field(
        description="Video duration in seconds.",
    )
    height: int | None = Field(
        default=None,
        description="Output height in pixels",
    )
    width: int | None = Field(
        default=None,
        description="Output width in pixels.",
    )
    aspect_ratio: str | None = Field(
        default=None,
        description=(
            "Output aspect ratio. "
            "The default is 16:9 if omitted. "
            "The width of the output will be updated to match the height based on the aspect ratio, "
            "rounded up to the nearest integer. "
        ),
    )
    infer_steps: int | None = Field(
        default=None,
        description=(
            "Number of denoising / inference steps. Higher values trade more "
            "compute for potentially finer detail. This is a best-effort match: "
            "if the model does not support it the model default "
            "is used. "
        ),
    )
    audio: bool | None = Field(
        default=None,
        description=(
            "Whether the video should include audio. "
            "**Note:** some models do no support audio, "
            "or the audio is always on (e.g. Veo 3.1). "
            "In those cases this field has no effect."
        ),
    )
    guidance_scale: list[float] | None = Field(
        default=None,
        description=(
            "Classifier-free guidance scale(s). Controls how strongly the output "
            "adheres to the prompt: higher values follow the prompt more closely "
            "at the cost of diversity. If the model supports multiple guidance "
            "scales, these will be applied in a sequence (e.g. per stage or per "
            "denoising phase). This is a best-effort match: if the model does "
            "not support it, or does not support the number of scales provided, "
            "the model default is used. "
        ),
    )

    # Accept the legacy scalar shape from tasks stored before guidance_scale
    # became a list, so old rows keep loading after the schema change.
    @field_validator("guidance_scale", mode="before")
    @classmethod
    def _coerce_scalar_guidance_scale(cls, v: object) -> object:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return [float(v)]
        return v


class ImageConfig(BaseModel):
    model_config = _WIRE_MODEL_CONFIG

    height: int | None = Field(
        default=None,
        description="Output height in pixels",
    )
    width: int | None = Field(
        default=None,
        description="Output width in pixels.",
    )
    aspect_ratio: str | None = Field(
        default=None,
        description=(
            "Output aspect ratio. "
            "The default is 1:1 if omitted. "
            "The width of the output will be updated to match the height based on the aspect ratio, "
            "rounded up to the nearest integer. "
        ),
    )
    infer_steps: int | None = Field(
        default=None,
        description=(
            "Number of denoising / inference steps. Higher values trade more "
            "compute for potentially finer detail. This is a best-effort match: "
            "if the model does not support it the model default "
            "is used. "
        ),
    )
    guidance_scale: list[float] | None = Field(
        default=None,
        description=(
            "Classifier-free guidance scale(s). Controls how strongly the output "
            "adheres to the prompt: higher values follow the prompt more closely "
            "at the cost of diversity. If the model supports multiple guidance "
            "scales, these will be applied in a sequence (e.g. per stage or per "
            "denoising phase). This is a best-effort match: if the model does "
            "not support it, or does not support the number of scales provided, "
            "the model default is used. "
        ),
    )

    # Accept the legacy scalar shape from tasks stored before guidance_scale
    # became a list, so old rows keep loading after the schema change.
    @field_validator("guidance_scale", mode="before")
    @classmethod
    def _coerce_scalar_guidance_scale(cls, v: object) -> object:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return [float(v)]
        return v


class TaskUpdate(BaseModel):
    model_config = _WIRE_MODEL_CONFIG

    status: TaskStatus
    progress: float | None = None


class TaskInput(BaseModel):
    """
    Public input model for :meth:`machgen.client.MachGenClient.submit_task`.

    Model/task_type need to match the supported model list.
    """

    model_config = _WIRE_MODEL_CONFIG

    prompt: str = Field(description="Text prompt driving generation.")
    enhance_prompt: bool = Field(
        default=False,
        description=(
            "Whether prompt enhancement should be enabled. "
            "Enabling this would slow down generation. "
        ),
    )
    multi_prompt: list[str] | None = Field(
        default=None,
        description=(
            "Per-shot prompts for multi-shot passthrough vendors; length must "
            "match src_image_urls when used as reference frames. "
            "**This is only supported/needed for Kling-v3 R2V.**"
        ),
    )
    model: str = Field(description="Model id, e.g. 'Wan2.2-T2V-A14B', 'Kling-v3'.")
    task_type: str = Field(description="one of T2I, I2I, T2V, I2V, R2V")
    moderate: bool = Field(
        default=True,
        description="Whether this request is screened by content moderation.",
    )
    video_config: VideoConfig | None = Field(
        default=None, description="Required for video task types."
    )
    image_config: ImageConfig | None = Field(
        default=None, description="Required for image task types."
    )
    seed: int | None = Field(
        default=None,
        description="Seed for reproducible generation. If not specified, a random seed will be used.",
    )
    src_image_urls: list[str] | None = Field(
        default=None,
        description=(
            "Source / reference image URLs. "
            "Only needed for tasks that require input images like I2I, I2V, R2V "
            "Refer to the API docs for concrete examples how to use this and what inputs are allowed. "
            "For image to video tasks, this can optionally specify 1 (first frame) or 2 (first & last frames) input images. "
        ),
    )
    subject_to_image_ids: dict[str, list[int]] | None = Field(
        default=None,
        description=(
            "R2V only: maps a subject name to the indices of its reference "
            "images in src_image_urls, e.g. {'alice': [0, 1], 'bob': [2]}. The "
            "prompt may address a subject via '@name'. Honored by vendors with "
            "named subjects (Vidu reference2video)."
        ),
    )
