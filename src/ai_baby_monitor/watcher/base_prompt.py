import datetime as dt

from ai_baby_monitor.i18n import get_language

# The JSON field names (should_alert, reasoning, recommended_awareness_level)
# and the enum values (LOW/MEDIUM/HIGH) are part of the response contract and
# must stay in English in both languages.
_PROMPT_TEMPLATES = {
    "en": """
You are given the following instructions:
{instructions}

If the instructions are violated, you should alert the user.
You should also recommend the awareness level based on the image.
Please generate a structured response in raw JSON format:
- should_alert (boolean)
- reasoning (string)
- recommended_awareness_level (Enum AwarenessLevel; one of: LOW, MEDIUM, HIGH)
Always respond in English, regardless of the content in the images.

Note on the camera: it may switch to infrared night-vision mode, in which frames
are grayscale/black-and-white. A grayscale or dark image is normal camera
behavior — it is NOT a camera fault and is NOT itself a hazard; do not alert just
because the picture is monochrome, dim, or low on detail. In night-vision
(grayscale) frames, be more conservative: only set should_alert for a clear,
high-confidence violation; when unsure, prefer should_alert = false.
        """,
    "zh": """
你会收到以下监控规则：
{instructions}

如果这些规则被违反，你应当提醒用户。
你还需要根据画面内容给出关注等级（awareness level）。
请严格按以下 JSON 结构返回原始 JSON（字段名保持英文）：
- should_alert (布尔值)
- reasoning (字符串)
- recommended_awareness_level (枚举 AwarenessLevel，取值之一：LOW、MEDIUM、HIGH)
无论画面内容是什么，请始终使用中文回答（reasoning 字段用中文）。

关于摄像头：它可能会切换到红外夜视模式，此时画面是灰度/黑白的。黑白或偏暗
的画面是相机的正常行为——不是摄像头故障，本身也不是危险信号；不要仅仅因为画面是
黑白、偏暗或细节少就告警。在夜视（灰度）画面下请更加保守：只在有明确、高置信度的
违规时才置 should_alert = true，拿不准时宁可不告警。
        """,
}


def get_instructions_prompt(instructions: list[str], language: str | None = None) -> str:
    """Inject the instructions into the base prompt for the given language."""
    if len(instructions) == 0:
        raise ValueError("Instructions must be a non-empty list")

    language = language or get_language()
    template = _PROMPT_TEMPLATES.get(language, _PROMPT_TEMPLATES["en"])

    instructions_str = "\n".join(["* " + _ for _ in instructions])
    return template.format(instructions=instructions_str)


# Per-image / per-skip time annotations interleaved with the frames. Kept
# language-aware so the model sees the note in the same language it answers in.
_FRAME_ANNOTATION = {
    "en": "[frame {index} @ {ts}]",
    "zh": "[帧{index} @ {ts}]",
}
_NIGHT_SUFFIX = {
    "en": " · night-vision/IR",
    "zh": " · 夜视/红外模式",
}
_SKIPPED_ANNOTATION = {
    "en": "[skipped {count} frames: {start} -> {end} (no visible change)]",
    "zh": "[跳过 {count} 帧：{start} → {end}（画面无明显变化）]",
}
_MODE_TRANSITION_ANNOTATION = {
    "night": {
        "en": "[switched to night-vision (IR) mode @ {ts}]",
        "zh": "[切换到夜视（红外）模式 @ {ts}]",
    },
    "color": {
        "en": "[switched back to color mode @ {ts}]",
        "zh": "[切换到彩色模式 @ {ts}]",
    },
}
_CAMERA_HEADER = {
    "en": "[camera: {camera}]",
    "zh": "[摄像头：{camera}]",
}


def _prefix_camera(text: str, camera: str | None) -> str:
    """Fold a camera name into a bracketed annotation: ``[baby · 帧1 @ …]``."""
    if not camera or not text.startswith("["):
        return text
    return f"[{camera} · {text[1:]}"


def _format_timestamp(timestamp: dt.datetime) -> str:
    """Format to millisecond precision (no timezone math — timestamps are local)."""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def format_frame_annotation(
    index: int,
    timestamp: dt.datetime,
    night: bool = False,
    camera: str | None = None,
    language: str | None = None,
) -> str:
    """Annotation before a frame image, e.g. ``[帧1 @ 2026-10-14 12:00:00.000]``.

    When ``night`` is set, a ``夜视/红外模式`` marker is appended so the model
    knows the grayscale picture is a night-vision frame, not a fault. When
    ``camera`` is set, the camera/view name is prefixed (multi-camera rooms).
    """
    language = language or get_language()
    template = _FRAME_ANNOTATION.get(language, _FRAME_ANNOTATION["en"])
    text = template.format(index=index + 1, ts=_format_timestamp(timestamp))
    if night:
        text += _NIGHT_SUFFIX.get(language, _NIGHT_SUFFIX["en"])
    return _prefix_camera(text, camera)


def format_skipped_annotation(
    count: int,
    start_ts: dt.datetime,
    end_ts: dt.datetime,
    camera: str | None = None,
    language: str | None = None,
) -> str:
    """Annotation replacing a run of unchanged frames (no image is sent)."""
    language = language or get_language()
    template = _SKIPPED_ANNOTATION.get(language, _SKIPPED_ANNOTATION["en"])
    text = template.format(
        count=count,
        start=_format_timestamp(start_ts),
        end=_format_timestamp(end_ts),
    )
    return _prefix_camera(text, camera)


def format_mode_transition_annotation(
    to_night: bool,
    timestamp: dt.datetime,
    camera: str | None = None,
    language: str | None = None,
) -> str:
    """Annotation for a color<->night-vision switch (no image is sent)."""
    language = language or get_language()
    key = "night" if to_night else "color"
    template = _MODE_TRANSITION_ANNOTATION[key].get(
        language, _MODE_TRANSITION_ANNOTATION[key]["en"]
    )
    text = template.format(ts=_format_timestamp(timestamp))
    return _prefix_camera(text, camera)


def format_camera_header(camera: str, language: str | None = None) -> str:
    """Header text labeling which camera a following block of frames is from."""
    language = language or get_language()
    template = _CAMERA_HEADER.get(language, _CAMERA_HEADER["en"])
    return template.format(camera=camera)
