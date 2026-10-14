import datetime as dt
import io
import os
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import streamlit as st
from PIL import Image

from ai_baby_monitor.i18n import t
from ai_baby_monitor.keys import camera_keys
from ai_baby_monitor.stream import RedisStreamHandler


def _display_tz() -> dt.tzinfo | None:
    """Timezone for rendering timestamps, from ``APP_TZ`` (IANA name).

    Defaults to the system local timezone when unset/unrecognized. Set
    ``APP_TZ=Asia/Shanghai`` (or similar) so the dashboard shows local time even
    when the container runs in UTC.
    """
    name = os.getenv("APP_TZ")
    if not name:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def _localize(epoch_seconds: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(epoch_seconds, tz=_display_tz())


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
            for camera in selected_config.cameras:
                label = camera.name or t("camera_default")
                st.caption(t("camera_uri_line", camera=label, uri=camera.uri))
                st.caption(
                    t(
                        "camera_geometry",
                        w=camera.frame_width,
                        h=camera.frame_height,
                        rate=camera.subsample_rate,
                    )
                )
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


def _open_image(frame_data) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(bytes(frame_data)))
    except Exception:
        return None


def get_camera_images(
    redis_handler: RedisStreamHandler, room_name: str, cameras
) -> list[tuple[str, Image.Image | None, dt.datetime | None]]:
    """Latest realtime frame per camera: list of ``(label, image, timestamp)``."""
    results = []
    for camera in cameras:
        label = camera.name or t("camera_default")
        frames = redis_handler.get_latest_frames(
            camera_keys(room_name, camera.name)["realtime"], count=1
        )
        if frames:
            results.append((label, _open_image(frames[0].frame_data), frames[0].timestamp))
        else:
            results.append((label, None, None))
    return results


def fetch_logs(
    redis_handler: RedisStreamHandler,
    room_name: str,
    num_logs: int = 3,
) -> list[dict]:
    from ai_baby_monitor.keys import room_keys

    logs = redis_handler.get_latest_logs(room_keys(room_name)["logs"], count=num_logs)

    new_logs = []
    for log_id, log_data in logs:
        log_data = redis_handler.deserialize_log(log_data)
        log_data["timestamp"] = _localize(float(log_data["timestamp"]))

        # Collect the representative photo(s). Legacy single-camera uses `frame`;
        # multi-camera rooms use `frame__{camera}` — show every one, labeled.
        images: list[tuple[str | None, Image.Image]] = []
        if isinstance(log_data.get("frame"), (bytes, bytearray)):
            image = _open_image(log_data["frame"])
            if image is not None:
                images.append((None, image))
        for key in sorted(log_data):
            if key.startswith("frame__"):
                raw = log_data[key]
                if isinstance(raw, (bytes, bytearray)):
                    image = _open_image(raw)
                    if image is not None:
                        images.append((key[len("frame__") :], image))
        log_data["images"] = images

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

    Every entry carries its representative photo(s); the only difference is that
    anomalous entries (an alert, or MEDIUM/HIGH awareness) default to expanded,
    while normal ones default to collapsed.
    """
    for log in reversed(logs):  # newest first
        ts = log.get("timestamp")
        # Show to the second (a wall-clock review doesn't need microseconds).
        timestamp = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, dt.datetime) else ts

        # Error/health records: {timestamp, error|info} — no verdict fields, no photo.
        if "should_alert" not in log:
            if "error" in log:
                with st.expander(f"⚠️ {timestamp}", expanded=False):
                    st.write(t("error", error=log["error"]))
            elif "info" in log:
                with st.expander(f"ℹ️ {timestamp}", expanded=False):
                    st.write(log["info"])
            continue

        alert_icon = "🚨" if str(log.get("should_alert")) == "1" else "🟢"
        level = log.get("awareness_level", "UNKNOWN")
        header = f"{alert_icon} {timestamp} · {_colored_level(level)}"
        if str(log.get("night_mode")) == "1":
            header += f" · 🌙 {t('night_mode_short')}"

        with st.expander(header, expanded=_is_anomalous(log)):
            images = log.get("images") or []
            if images:
                for camera, image in images:
                    if camera:
                        st.caption(t("camera_photo_label", camera=camera))
                    st.image(image, use_container_width=True)
            else:
                st.caption(t("no_photo"))
            st.write(t("reasoning", reasoning=log.get("reasoning", "")))
