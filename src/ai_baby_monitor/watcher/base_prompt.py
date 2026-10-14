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
_SKIPPED_ANNOTATION = {
    "en": "[skipped {count} frames: {start} -> {end} (no visible change)]",
    "zh": "[跳过 {count} 帧：{start} → {end}（画面无明显变化）]",
}


def _format_timestamp(timestamp: dt.datetime) -> str:
    """Format to millisecond precision (no timezone math — timestamps are local)."""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def format_frame_annotation(
    index: int, timestamp: dt.datetime, language: str | None = None
) -> str:
    """Annotation placed before a frame image, e.g. ``[帧1 @ 2026-10-14 12:00:00.000]``."""
    language = language or get_language()
    template = _FRAME_ANNOTATION.get(language, _FRAME_ANNOTATION["en"])
    return template.format(index=index + 1, ts=_format_timestamp(timestamp))


def format_skipped_annotation(
    count: int,
    start_ts: dt.datetime,
    end_ts: dt.datetime,
    language: str | None = None,
) -> str:
    """Annotation replacing a run of unchanged frames (no image is sent)."""
    language = language or get_language()
    template = _SKIPPED_ANNOTATION.get(language, _SKIPPED_ANNOTATION["en"])
    return template.format(
        count=count,
        start=_format_timestamp(start_ts),
        end=_format_timestamp(end_ts),
    )
