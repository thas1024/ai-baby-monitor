"""Minimal i18n for user-facing strings.

Language is chosen by the ``APP_LANG`` environment variable (``zh`` or ``en``,
default ``zh``). Only text the user reads is translated — log messages, Redis
field names and code identifiers stay in English on purpose.
"""

import os

DEFAULT_LANG = "zh"
SUPPORTED_LANGS = ("zh", "en")

# Keyed by message id, then language. Use ``{name}`` placeholders for values.
_TRANSLATIONS: dict[str, dict[str, str]] = {
    # App chrome
    "app_title": {"zh": "婴儿监控 · 实时画面", "en": "Baby Monitor Stream Viewer"},
    "logs_viewer_title": {"zh": "婴儿监控 · 历史日志", "en": "Baby Monitor Logs Viewer"},
    "no_room_config": {"zh": "未提供房间配置文件。", "en": "No room config file was provided."},
    # Sidebar
    "sidebar_header": {"zh": "房间配置", "en": "Room Configuration"},
    "select_room": {"zh": "选择房间", "en": "Select Room"},
    "select_mode": {"zh": "选择模式", "en": "Select Mode"},
    "mode_realtime": {"zh": "实时画面", "en": "Real-time stream"},
    "mode_historic": {"zh": "历史日志", "en": "Historic logs"},
    "instructions_label": {"zh": "监控规则：", "en": "Instructions:"},
    "camera_details": {"zh": "摄像头信息", "en": "Camera details"},
    "camera_default": {"zh": "默认", "en": "default"},
    "camera_uri_line": {"zh": "{camera}： {uri}", "en": "{camera}: {uri}"},
    "camera_geometry": {
        "zh": "{w}×{h}，每 {rate} 帧取 1 帧",
        "en": "{w}×{h}, 1 of every {rate} frames",
    },
    "camera_photo_label": {"zh": "摄像头「{camera}」", "en": "Camera \"{camera}\""},
    "camera_uri": {"zh": "摄像头地址： {uri}", "en": "Camera URI:  {uri}"},
    "frame_width": {"zh": "画面宽度： {value}", "en": "Frame width: {value}"},
    "frame_height": {"zh": "画面高度： {value}", "en": "Frame height: {value}"},
    "subsample_rate": {
        "zh": "抽帧间隔：每 {value} 帧取 1 帧",
        "en": "Subsample rate: {value}",
    },
    "llm_model_expander": {"zh": "大模型", "en": "LLM Model"},
    "llm_model_caption": {"zh": "模型： {model}", "en": "LLM Model: {model}"},
    # Stream area
    "frame_info": {"zh": "画面信息", "en": "Frame Info"},
    "timestamp": {"zh": "时间： {ts}", "en": "Timestamp: {ts}"},
    "llm_logs": {"zh": "大模型日志", "en": "LLM Logs"},
    # Historic logs controls
    "num_logs_label": {
        "zh": "获取最近多少条日志",
        "en": "Number of latest logs to fetch",
    },
    "fetch_logs_button": {"zh": "查询日志", "en": "Fetch logs"},
    "filter_level": {"zh": "按关注等级过滤", "en": "Filter by awareness level"},
    "filter_alerts_only": {"zh": "只看告警", "en": "Alerts only"},
    "no_logs_match": {"zh": "没有符合条件的日志。", "en": "No logs match the filter."},
    # Log records
    "alert_status": {"zh": "告警状态： {status}", "en": "Alert status: {status}"},
    "awareness_level": {"zh": "关注等级： {level}", "en": "Awareness level: {level}"},
    "reasoning": {"zh": "判断理由： {reasoning}", "en": "Reasoning: {reasoning}"},
    "error": {"zh": "错误： {error}", "en": "Error: {error}"},
    "night_mode": {"zh": "夜视模式： {value}", "en": "Night vision: {value}"},
    "night_mode_short": {"zh": "夜视", "en": "night vision"},
    "yes": {"zh": "是", "en": "yes"},
    "no": {"zh": "否", "en": "no"},
    "photo": {"zh": "现场照片", "en": "Photo"},
    "no_photo": {"zh": "（此条无照片）", "en": "(no photo for this entry)"},
    "frame_time": {"zh": "画面时间： {ts}", "en": "Frame time: {ts}"},
    # Awareness level display labels (stored values stay LOW/MEDIUM/HIGH)
    "awareness_LOW": {"zh": "低", "en": "LOW"},
    "awareness_MEDIUM": {"zh": "中", "en": "MEDIUM"},
    "awareness_HIGH": {"zh": "高", "en": "HIGH"},
    "awareness_UNKNOWN": {"zh": "未知", "en": "UNKNOWN"},
}


def get_language() -> str:
    """Return the active language code from ``APP_LANG`` (default ``zh``)."""
    lang = (os.getenv("APP_LANG") or DEFAULT_LANG).strip().lower()
    return lang if lang in SUPPORTED_LANGS else DEFAULT_LANG


def t(key: str, lang: str | None = None, **kwargs) -> str:
    """Translate ``key`` into ``lang`` (or the active language).

    Falls back to the default language, then to the key itself, so a missing
    translation never raises. ``kwargs`` fill ``{name}`` placeholders.
    """
    lang = lang or get_language()
    text = _TRANSLATIONS.get(key, {}).get(lang)
    if text is None:
        text = _TRANSLATIONS.get(key, {}).get(DEFAULT_LANG, key)
    return text.format(**kwargs) if kwargs else text
