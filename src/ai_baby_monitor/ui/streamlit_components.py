import datetime as dt
import io
import os

import streamlit as st
from PIL import Image

from ai_baby_monitor.i18n import t
from ai_baby_monitor.keys import room_keys
from ai_baby_monitor.stream import RedisStreamHandler


def display_sidebar(
    room_configs_key: str = "room_configs",
    config_key: str = "selected_config",
    mode_key: str = "selected_mode",
):
    with st.sidebar:
        st.header(t("sidebar_header"))

        selected_config = st.selectbox(
            t("select_room"),
            options=list(st.session_state[room_configs_key].values()),
            index=0,
            key=config_key,
        )

        # Options are stable mode keys; the label is translated for display.
        st.radio(
            t("select_mode"),
            options=["realtime", "historic"],
            format_func=lambda key: t(f"mode_{key}"),
            index=0,
            key=mode_key,
        )

        st.caption(t("instructions_label"))
        for instruction in selected_config.instructions:
            st.markdown(f"- {instruction}")
        st.divider()
        with st.expander(t("camera_details")):
            st.caption(t("camera_uri", uri=selected_config.camera_uri))
            st.caption(t("frame_width", value=selected_config.frame_width))
            st.caption(t("frame_height", value=selected_config.frame_height))
            st.caption(t("subsample_rate", value=selected_config.subsample_rate))
        st.divider()
        with st.expander(t("llm_model_expander")):
            st.caption(t("llm_model_caption", model=os.getenv("LLM_MODEL_NAME")))


@st.cache_resource
def get_cached_redis_handler(
    redis_host: str, redis_port: int, redis_password: str | None = None
):
    return RedisStreamHandler(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
    )


def get_last_image_with_timestamp(
    redis_handler: RedisStreamHandler, room_name: str
) -> tuple[Image.Image | None, dt.datetime | None]:
    frames = redis_handler.get_latest_frames(room_keys(room_name)["realtime"], count=1)

    if not frames:
        return None

    jpeg_bytes = bytes(frames[0].frame_data)
    image = Image.open(io.BytesIO(jpeg_bytes))
    return image, frames[0].timestamp


def fetch_logs(
    redis_handler: RedisStreamHandler,
    room_name: str,
    num_logs: int = 3,
) -> list[dict]:
    logs = redis_handler.get_latest_logs(room_keys(room_name)["logs"], count=num_logs)

    new_logs = []
    for log_id, log_data in logs:
        log_data = redis_handler.deserialize_log(log_data)
        log_data["timestamp"] = dt.datetime.fromtimestamp(float(log_data["timestamp"]))

        # Turn the stored thumbnail bytes into a displayable image.
        frame_bytes = log_data.get("frame")
        if isinstance(frame_bytes, (bytes, bytearray)) and frame_bytes:
            try:
                log_data["image"] = Image.open(io.BytesIO(frame_bytes))
            except Exception:
                log_data["image"] = None
        else:
            log_data["image"] = None

        new_logs.append(log_data)

    return new_logs


def _is_anomalous(log: dict) -> bool:
    """Whether a verdict is worth expanding by default in the timeline."""
    if "error" in log and "should_alert" not in log:
        return True
    return str(log.get("should_alert")) == "1" or log.get("awareness_level") in (
        "MEDIUM",
        "HIGH",
    )


def _colored_level(level: str) -> str:
    label = t(f"awareness_{level}")
    if level == "LOW":
        return f":green[{label}]"
    if level == "MEDIUM":
        return f":yellow[{label}]"
    return f":red[{label}]"


def render_logs(logs: list[dict]):
    """Render the logs as a timeline, newest first, each entry collapsible.

    Every entry carries its representative photo; the only difference is that
    anomalous entries (an alert, or MEDIUM/HIGH awareness) default to expanded,
    while normal ones default to collapsed.
    """
    for log in reversed(logs):  # newest first
        timestamp = log.get("timestamp")

        # Error records: {timestamp, error} — no verdict fields, no photo.
        if "error" in log and "should_alert" not in log:
            with st.expander(f"⚠️ {timestamp}", expanded=False):
                st.write(t("error", error=log["error"]))
            continue

        alert_icon = "🚨" if str(log.get("should_alert")) == "1" else "🟢"
        level = log.get("awareness_level", "UNKNOWN")
        header = f"{alert_icon} {timestamp} · {_colored_level(level)}"
        if str(log.get("night_mode")) == "1":
            header += f" · 🌙 {t('night_mode_short')}"

        with st.expander(header, expanded=_is_anomalous(log)):
            image = log.get("image")
            if image is not None:
                st.image(image, use_container_width=True)
            else:
                st.caption(t("no_photo"))
            if log.get("frame_timestamp"):
                st.caption(t("frame_time", ts=log["frame_timestamp"]))
            st.write(t("reasoning", reasoning=log.get("reasoning", "")))
