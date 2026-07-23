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
        description="Output height in pixels. Required for generation (a missing height is rejected at submit).",
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
    bitrate_mode: str | None = Field(
        default=None,
        description=(
            "Encode quality for the delivered video: 'standard' or 'high'. "
            "**Only supported for Seedance-2.0.** Omitted -> the vendor default."
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
            "**Note:** some models do not support audio, "
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
        description="Output height in pixels. Required for generation (a missing height is rejected at submit).",
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

    # What to generate
    model: str = Field(description="Model id, e.g. 'Wan2.2-A14B', 'Kling-v3'.")
    task_type: str = Field(description="one of T2I, I2I, T2V, I2V, R2V")

    # Prompt
    prompt: str = Field(description="Text prompt driving generation.")
    enhance_prompt: bool | None = Field(
        default=None,
        description=(
            "Whether prompt enhancement should be enabled. "
            "Enabling this would slow down generation but would improve quality. "
            "By default, if this is not explicitly set we will let the model determine the default behavior. "
            "Users can still explicitly force it to enable/disable by setting this field based on the requirement."
        ),
    )
    multi_prompt: list[str] | None = Field(
        default=None,
        description=(
            "Per-shot text prompts for Kling-v3 multi-shot video (`shot_type` "
            "'customize'). 1-6 shots, paired 1:1 with `shot_durations`. The "
            "top-level `prompt` is ignored when this is set."
        ),
    )
    shot_type: str | None = Field(
        default=None,
        description=(
            "Enables Kling-v3 multi-shot (T2V / I2V). 'customize' splits the "
            "video into the shots given by `multi_prompt` + `shot_durations`; "
            "'intelligence' derives the shots from the single `prompt`. Omitted "
            "-> single-shot."
        ),
    )
    shot_durations: list[int] | None = Field(
        default=None,
        description=(
            "Per-shot durations in seconds for `shot_type` 'customize'; one per "
            "`multi_prompt` entry, each >= 1, summing to "
            "`video_config.duration_secs`."
        ),
    )
    negative_prompt: str | None = Field(
        default=None,
        description=(
            "What the video should avoid. **Kling-v3 only.** Omitted -> the "
            "vendor default ('blur, distort, and low quality')."
        ),
    )
    cfg_scale: float | None = Field(
        default=None,
        description=(
            "How strongly the output adheres to the prompt, in [0, 1]. "
            "**Kling-v3 only.** Omitted -> the vendor default (0.5)."
        ),
    )

    # Output configuration
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

    # Source media (I2I, I2V, R2V)
    src_image_urls: list[str] | None = Field(
        default=None,
        description=(
            "Source / reference image URLs. "
            "Only needed for tasks that require input images like I2I, I2V, R2V. "
            "Refer to the API docs for concrete examples of how to use this and what inputs are allowed. "
            "For I2V, entry 0 is the start frame and an optional entry 1 is the "
            "end frame on models that support it (Seedance-2.0, Kling-v3, "
            "LTX-2.3-Pro, Vidu-Q3-Turbo, Vidu-Q3-Pro); a second image returns "
            "400 elsewhere. "
        ),
    )
    src_video_urls: list[str] | None = Field(
        default=None,
        description=(
            "Reference video URLs. **Only supported for Seedance-2.0 R2V**, "
            "which accepts up to 3 clips of 2-15s each (15s combined)."
        ),
    )
    src_audio_urls: list[str] | None = Field(
        default=None,
        description=(
            "Reference audio URLs. **Only supported for Seedance-2.0 R2V**, "
            "which accepts up to 3 clips totalling 15s. Audio may not be the "
            "only reference - at least one image or video is required."
        ),
    )

    # Named references the prompt addresses via @name (R2V)
    subject_to_image_ids: dict[str, list[int]] | None = Field(
        default=None,
        description=(
            "R2V only: maps a subject name to the indices of its reference "
            "images in src_image_urls, e.g. {'alice': [0, 1], 'bob': [2]}. The "
            "prompt may address a subject via '@name'. Honored by vendors with "
            "named subjects (Vidu reference2video)."
        ),
    )
    subject_to_video_ids: dict[str, list[int]] | None = Field(
        default=None,
        description=(
            "R2V only: subject_to_image_ids for src_video_urls. A name may "
            "appear in only one of the three subject maps. Seedance-2.0 only."
        ),
    )
    subject_to_audio_ids: dict[str, list[int]] | None = Field(
        default=None,
        description=(
            "R2V only: subject_to_image_ids for src_audio_urls. A name may "
            "appear in only one of the three subject maps. Seedance-2.0 only."
        ),
    )
    element_ids: list[int] | None = Field(
        default=None,
        description=(
            "Kling-v3 only: ordered Kling element library ids (<=3) to include. "
            "The prompt references them via `@handle` (see `element_handles`), "
            "rewritten to Kling's positional `<<<element_N>>>` at submit."
        ),
    )
    element_handles: list[str] | None = Field(
        default=None,
        description=(
            "Kling-v3 only: the `@handle` for each `element_ids` entry (same "
            "order). Each `@handle` in the prompt is rewritten to "
            "`<<<element_N>>>` for the vendor while the stored prompt keeps it."
        ),
    )

    # Policy
    moderate: bool = Field(
        default=True,
        description="Whether this request is screened by content moderation.",
    )
