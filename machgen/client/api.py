from __future__ import annotations

from enum import StrEnum
from typing import Literal

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
        description=(
            "Video duration in seconds. Where a surface allows it, -1 asks the "
            "model to determine the length from the request, such as matching a "
            "reference clip or selecting a smart output duration. Allowed values "
            "are enforced by the synced capability contract."
        ),
    )
    height: int | None = Field(
        default=None,
        description=(
            "Output height in pixels. Required for generation (a missing height "
            "is rejected at submit). For UPSCALE on a video upscale model this "
            "is the target resolution tier - the output's shorter side - and "
            "both output dimensions are derived from the source, preserving "
            "its aspect ratio."
        ),
    )
    width: int | None = Field(
        default=None,
        description="Output width in pixels.",
    )
    aspect_ratio: str | None = Field(
        default=None,
        description=(
            "Output aspect ratio. The model capability contract declares the "
            "omission default; most models use 16:9, while source-aware models "
            "may use adaptive. "
            "The width of the output will be updated to match the height based on the aspect ratio, "
            "rounded up to the nearest integer. "
        ),
    )
    bitrate_mode: str | None = Field(
        default=None,
        description=(
            "Encode quality for the delivered video: 'standard' or 'high'. "
            "Supported for Seedance-2.0 and Seedance-2.0-Fast. "
            "Omitted -> the vendor default."
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
    sharpen: bool = Field(
        default=False, description="Sharpens MiniMax-H3 480p/768p/2K videos"
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
    multi_prompt: list[str] | None = Field(
        default=None,
        description=(
            "Per-shot text prompts for Kling Video 3.0 and Omni multi-shot video (`shot_type` "
            "'customize'). 1-6 shots, paired 1:1 with `shot_durations`. The "
            "top-level `prompt` is ignored when this is set."
        ),
    )
    shot_type: str | None = Field(
        default=None,
        description=(
            "Enables Kling Video 3.0 or Omni multi-shot. 'customize' splits the "
            "video into the shots given by `multi_prompt` + `shot_durations`; "
            "'intelligence' derives the shots from the single `prompt`. Omitted "
            "-> single-shot."
        ),
    )
    shot_durations: list[int] | None = Field(
        default=None,
        description=(
            "Per-shot durations in seconds for `shot_type` 'customize'; one per "
            "`multi_prompt` entry, each >= 1, summing to `duration_secs`."
        ),
    )
    negative_prompt: str | None = Field(
        default=None,
        description=(
            "What the video should avoid. **Kling-v3 only.** Omitted -> the "
            "vendor default ('blur, distort, and low quality')."
        ),
    )
    element_ids: list[int] | None = Field(
        default=None,
        description=(
            "Kling Video 3.0 and Omni only: ordered Kling element library ids (<=3) to include. "
            "The prompt references them via `@handle` (see `element_handles`), "
            "rewritten to Kling's positional `<<<element_N>>>` at submit."
        ),
    )
    element_handles: list[str] | None = Field(
        default=None,
        description=(
            "Kling Video 3.0 and Omni only: the `@handle` for each `element_ids` entry (same "
            "order). Each `@handle` in the prompt is rewritten to "
            "`<<<element_N>>>` for the vendor while the stored prompt keeps it."
        ),
    )

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


class DialogueTurn(BaseModel):
    """One speaker turn in a T2D (text to dialogue) task."""

    model_config = _WIRE_MODEL_CONFIG

    voice_id: str = Field(description="Voice speaking this turn.")
    text: str = Field(description="What this speaker says.")


class MusicChunk(BaseModel):
    """One segment of a T2M composition plan (ElevenLabs music_v2)."""

    model_config = _WIRE_MODEL_CONFIG

    text: str = Field(
        default="",
        description="Sung or spoken text for this segment. Empty for instrumental.",
    )
    duration_ms: int = Field(description="Segment length in milliseconds.")
    positive_styles: list[str] | None = Field(
        default=None, description="Styles this segment should have."
    )
    negative_styles: list[str] | None = Field(
        default=None, description="Styles this segment should avoid."
    )
    context_adherence: str | None = Field(
        default=None,
        description=(
            "How closely this segment follows the surrounding ones: 'low', "
            "'medium', or 'high'. Omitted uses the model default."
        ),
    )


class CompositionPlan(BaseModel):
    """Segment-by-segment structure for a T2M task, in place of a prompt."""

    model_config = _WIRE_MODEL_CONFIG

    chunks: list[MusicChunk] = Field(description="Ordered segments of the track.")


class UpscaleConfig(BaseModel):
    """Knobs for the UPSCALE task type.

    Kept off ``ImageConfig`` / ``VideoConfig`` so the configs every other surface
    sends stay free of fields that are meaningless there. Those configs carry
    only output geometry - and for image upscale models the geometry is derived
    by the server from the source, never sent by the client, because it sets the
    per-megapixel charge.

    Which knobs a model accepts is declared per surface in the capability grid;
    sending one the selected engine does not declare is rejected at admission.
    """

    model_config = _WIRE_MODEL_CONFIG

    engine: str | None = Field(
        default=None,
        description=(
            "Engine variant within the selected model, e.g. 'proteus' or "
            "'redefine'. Omit to use the model's default engine. Call "
            "list_models for the engines a model exposes."
        ),
    )
    factor: int | None = Field(
        default=None,
        description=(
            "Image upscale models only. How much to scale each edge, e.g. 2 "
            "doubles width and height. The result is clamped to the model's "
            "maximum output edge and area, and the final dimensions are "
            "returned in the task's image_config."
        ),
    )
    params: dict[str, float | bool | str] | None = Field(
        default=None,
        description=(
            "Engine tuning parameters, by name. Only send what you want to "
            "change: Topaz auto-configures anything omitted, so passing a value "
            "replaces their tuning rather than confirming it. Which names an "
            "engine accepts, and their bounds, come from list_models."
        ),
    )


class TrainConfig(BaseModel):
    """Trainer settings for the TRAINING task type.

    Kept off the media configs the way ``UpscaleConfig`` is: a training request
    carries no prompt and no output geometry. Allowed ranges come from the
    model's capability grid and are checked at admission.
    """

    model_config = _WIRE_MODEL_CONFIG

    dataset: str = Field(
        description=(
            "The dataset archive to train on, as the @input/<artifact_path> "
            "reference returned by the upload endpoint for a .zip file."
        )
    )
    steps: int = Field(description="Number of optimizer steps to train for.")
    learning_rate: float = Field(description="Optimizer learning rate.")
    rank: int | None = Field(
        default=None,
        le=64,
        description="LoRA rank. LORA mode only. Default is 16 if not set. Support up to 64.",
    )
    alpha: float | None = Field(
        default=None,
        description="LoRA alpha scaling. LORA mode only; omit to use the model default.",
    )
    trigger_word: str | None = Field(
        default=None,
        description="Token prepended to every training prompt, and later to prompts that should use the adapter.",
    )
    eval_ratio: float | None = Field(
        default=None,
        description=(
            "Fraction of the dataset (0 to 1, exclusive) held out to report a "
            "validation loss during training. Omit or 0 to train on every sample."
        ),
    )
    default_caption: str | None = Field(
        default=None,
        description="Instruction used for samples that carry no caption of their own.",
    )
    checkpoint_name: str | None = Field(
        default=None,
        description=(
            "Name for the resulting adapter, shared across your account. Omit to get "
            "`<user_id>/<task_id>`. Letters, digits, `.`, `_` and `-`, up to 64 characters."
        ),
    )


class AudioConfig(BaseModel):
    """Configuration for audio generation.

    Supported fields and defaults depend on the selected model and task type.
    Provide text or generation instructions in ``TaskInput.prompt``. For YuE2,
    use the prompt for musical style and ``lyrics`` for the words to sing.
    For T2D, provide spoken text in ``turns`` instead of the prompt.
    """

    model_config = _WIRE_MODEL_CONFIG

    lyrics: str | None = Field(
        default=None,
        description=(
            "T2M models that accept separate lyrics: the words to sing, with "
            "section labels such as [Verse] and [Chorus]. Required for YuE2; "
            "provide the musical style in TaskInput.prompt. Omit for models "
            "that do not support separate lyrics."
        ),
    )

    voice_id: str | None = Field(
        default=None,
        description="T2S only: the voice to speak the prompt.",
    )
    turns: list[DialogueTurn] | None = Field(
        default=None,
        description=(
            "T2D only: ordered speaker turns. The prompt is unused for T2D - "
            "all spoken text lives here."
        ),
    )
    duration_secs: float | None = Field(
        default=None,
        description=(
            "T2SFX and T2M models that support a requested duration: target audio "
            "length in seconds. Accepted ranges and defaults depend on the model. "
            "Omit for YuE2, which generates variable-length audio and does not "
            "accept a requested duration."
        ),
    )
    stability: float | None = Field(
        default=None,
        description=(
            "T2S and T2D: 0-1. Lower is more emotionally variable, higher is "
            "more consistent. eleven_v3 accepts only 0.0, 0.5, or 1.0. "
            "Omitted uses the model default."
        ),
    )
    speed: float | None = Field(
        default=None,
        description="T2S only: speaking rate multiplier. Omitted uses the model default.",
    )
    similarity_boost: float | None = Field(
        default=None,
        description=(
            "T2S only: 0-1. How closely the output tracks the original voice. "
            "Omitted uses the model default."
        ),
    )
    style: float | None = Field(
        default=None,
        description=(
            "T2S only: 0-1. Style exaggeration. Higher values cost more latency. "
            "Omitted uses the model default."
        ),
    )
    use_speaker_boost: bool | None = Field(
        default=None,
        description="T2S only: boost similarity to the original speaker.",
    )
    output_format: str | None = Field(
        default=None,
        description=(
            "Audio encoding; supported values and defaults depend on the model. "
            "YuE2 supports only `flac`, which is also its default, and produces "
            "a `.flac` file. ElevenLabs T2S, T2D and T2SFX support "
            "`mp3_22050_32`, `mp3_24000_48`, `mp3_44100_32`, `mp3_44100_64`, "
            "`mp3_44100_96`, `mp3_44100_128`, `mp3_44100_192`, "
            "`opus_48000_32`, `opus_48000_64`, `opus_48000_96`, "
            "`opus_48000_128` and `opus_48000_192`. ElevenLabs T2M also supports "
            "`mp3_48000_128`, `mp3_48000_192`, `mp3_48000_240` and "
            "`mp3_48000_320`. ElevenLabs defaults to `mp3_44100_128`. "
            "MP3 is stored as `.mp3` and Opus as `.ogg`."
        ),
    )
    apply_text_normalization: str | None = Field(
        default=None,
        description=(
            "T2S and T2D: 'auto' (default), 'on', or 'off'. Controls whether "
            "numbers, dates, and abbreviations are spelled out before synthesis."
        ),
    )
    prompt_influence: float | None = Field(
        default=None,
        description=(
            "T2SFX only: 0-1. How literally the sound follows the prompt. "
            "Omitted uses the model default."
        ),
    )
    loop: bool | None = Field(
        default=None,
        description="T2SFX only: generate a seamlessly looping clip.",
    )
    force_instrumental: bool | None = Field(
        default=None,
        description="T2M only: generate without vocals.",
    )
    composition_plan: CompositionPlan | None = Field(
        default=None,
        description=(
            "T2M only: build the track segment by segment instead of from a "
            "prompt. Mutually exclusive with `prompt`, and the track length is "
            "the sum of the segment durations rather than `duration_secs`."
        ),
    )


class TaskUpdate(BaseModel):
    model_config = _WIRE_MODEL_CONFIG

    status: TaskStatus
    progress: float | None = None


class ReferenceOrderItem(BaseModel):
    """One entry in the original cross-media R2V conditioning sequence.

    ``index`` addresses the matching ``src_*_urls`` list. Keeping order
    separate avoids duplicating URLs while preserving the sequence expected by
    models such as MiniMax H3.
    """

    model_config = _WIRE_MODEL_CONFIG

    type: Literal["image", "video", "audio"]
    index: int = Field(ge=0)


class TaskInput(BaseModel):
    """
    Public input model for :meth:`machgen.client.MachGenClient.submit_task`.

    Model/task_type need to match the supported model list.
    """

    model_config = _WIRE_MODEL_CONFIG

    # What to generate
    model: str = Field(description="Model id, e.g. 'MiniMax-H3', 'Kling-v3'.")
    task_type: str = Field(
        description=(
            "one of:\n"
            "**T2I**, **I2I** (image), **T2V**, **I2V**, **R2V** (video), "
            "**UPSCALE** (image or video based on input type), "
            "**SHARPEN** (video), "
            "**T2S**, **T2D**, **T2SFX**, **T2M** (audio)"
        )
    )

    # Prompt
    prompt: str = Field(
        description=(
            "Text prompt driving generation. Unused for T2D, whose text lives "
            "in audio_config.turns."
        )
    )
    enhance_prompt: bool | None = Field(
        default=None,
        description=(
            "Enhance the prompt to get expected output. "
            "Most users should leave this in default setting. "
            "For new generation of models (LTX, MiniMax-H3 and beyond), "
            "turning off prompt enhancement would severely degrade quality. "
            "One should only disable for debugging purpose or advanced use cases. "
        ),
    )
    prompt_enhancer: str | None = Field(
        default=None,
        description=(
            "What prompt enhancer to use.\n"
            "Options are: default, native. If not specified, the default will be used.\n\n"
            "**default**: use MachGen fine-tuned prompt enhancer for faster, quality output\n"
            "**native**: the original enhancer paired with the model "
            "(This is slower and should only be used if the default one shows issues.)\n\n"
            "If the model does not have one, this option will fallback to the default."
        ),
    )
    # Output configuration
    video_config: VideoConfig | None = Field(
        default=None, description="Required for video task types."
    )
    image_config: ImageConfig | None = Field(
        default=None, description="Required for image task types."
    )
    audio_config: AudioConfig | None = Field(
        default=None, description="Required for audio task types."
    )
    upscale_config: UpscaleConfig | None = Field(
        default=None,
        description=("Engine and tuning knobs for UPSCALE."),
    )
    mode: str | None = Field(
        default=None,
        description=(
            "TRAINING only: what to produce. 'LORA' trains an adapter over the "
            "frozen base model; 'FULL_MODEL' tunes every weight."
        ),
    )
    train_config: TrainConfig | None = Field(
        default=None, description="Required for the TRAINING task type."
    )
    seed: int | None = Field(
        default=None,
        description="Seed for reproducible generation. If not specified, a random seed will be used.",
    )
    optimization_level: str | None = Field(
        default=None,
        description=(
            "Speed/quality trade-off tier: 'STANDARD/FAST/EXPRESS'. "
            "Higher optimization level means more aggressive optimizations, "
            "while lower level means more quality details. When not set, "
            "default levels are automatically applied."
        ),
    )

    adapter: str | None = Field(
        default=None,
        description=(
            "LoRA selector for the model, if supported. "
            "Omit to use the model's default.\n\n"
            "Currently supported on `MiniMax-H3-Turbo` (T2V, I2V, R2V):\n"
            "`larryvrh_v4_step600_ema` - 6 steps.\n"
            "`lightx2v_fl2v_8step_v10_native` - 8 steps.\n"
            "`silveroxides_dareties_fro099_v2` - 6 steps.\n"
            "`plaguekind_parasyte_turbo` - 6 steps.\n"
            "`silveroxides_4to8_dareties_v2` - 6 steps.\n"
            "`hyperflow` - 8 steps."
        ),
    )

    # Source media (I2I, I2V, R2V)
    src_image_urls: list[str] | None = Field(
        default=None,
        description=(
            "Source / reference image URLs. "
            "For I2V, entry 0 is the start frame and an optional entry 1 is the end frame. "
            "For R2V, these are used as references, and `subject_to_image_ids` must be used "
            "if any @ handles are used in the prompt as references to the subjects. "
            "Public image URLs or base64 encoded data URLs can be used. "
        ),
    )
    keyframe_indices: list[Literal[0, -1]] | None = Field(
        default=None,
        description=(
            "I2V only: where each src_image_urls entry sits in the output, "
            "aligned 1:1 with that list. 0 is the first frame and -1 the last "
            "frame, so [0], [0, -1] and [-1] are the accepted values. Omitted "
            "keeps the positional reading (entry 0 first, entry 1 last). "
            "[-1] alone animates towards a single end frame and is honored only "
            "by models that declare it (MiniMax-H3)."
        ),
    )
    src_video_urls: list[str] | None = Field(
        default=None, description=("Reference video URLs. Public http(s) URLs only. ")
    )
    src_audio_urls: list[str] | None = Field(
        default=None,
        description=(
            "Reference audio URLs for R2V surfaces that declare audio-reference "
            "support. Limits and whether audio may be sent alone are "
            "model-specific and enforced by the synced capability contract."
        ),
    )
    src_task_ids: list[str] | None = Field(
        default=None,
        description=(
            "The output artifact (video or image) generated from the upstream task(s) "
            "will be used as the input together with its input unless overridden in this task. "
            "For most tasks (UPSCALE etc.), this should be a singleton list only. "
            "Invalid input will be rejected. "
        ),
    )
    src_file_urls: list[str] | None = Field(
        default=None,
        description=(
            "Public HTTPS document URLs or local paths for R2V surfaces "
            "that declare file input support. This mode may be mutually "
            "exclusive with ordinary media references on a given model. Local "
            "paths are uploaded by the Python client before submission."
        ),
    )
    src_webpage_urls: list[str] | None = Field(
        default=None,
        description=(
            "Publicly reachable HTTPS webpage URLs for R2V surfaces that can use a linked page as input. "
            "This mode may be mutually exclusive with files and ordinary media "
            "references on a given model."
        ),
    )
    reference_video_operation: Literal["reference", "edit", "extend"] | None = Field(
        default=None,
        description=(
            "How the primary reference video is used on R2V surfaces that "
            "declare these operations. 'reference' conditions a new video, "
            "'edit' modifies the primary clip while matching its framing and "
            "length, and 'extend' continues it with a requested output length. "
            "The primary clip is src_video_urls[0]. Omitted preserves legacy "
            "provider-classified behavior."
        ),
    )
    reference_video_start_secs: list[float] | None = Field(
        default=None,
        description=(
            "R2V only: where each reference video starts being read, in seconds "
            "from the beginning of the clip, aligned 1:1 with src_video_urls as "
            "it stands after src_task_ids routing. Each value is >= 0 and must "
            "fall inside the clip; the remaining length is what the model sees "
            "and what is billed. Omitted reads every clip from 0. A non-zero "
            "start is honored only by models that declare it (MiniMax-H3)."
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
            "appear in only one of the three subject maps. Seedance 2.0 family only."
        ),
    )
    subject_to_audio_ids: dict[str, list[int]] | None = Field(
        default=None,
        description=(
            "R2V only: subject_to_image_ids for src_audio_urls. A name may "
            "appear in only one of the three subject maps. Seedance 2.0 family only."
        ),
    )
    reference_order: list[ReferenceOrderItem] | None = Field(
        default=None,
        description=(
            "R2V only: original order across image, video, and audio "
            "references. Each entry points to the matching src_*_urls list."
        ),
    )
    # Policy
    moderate: bool = Field(
        default=True,
        description="Whether this request is screened by content moderation.",
    )
