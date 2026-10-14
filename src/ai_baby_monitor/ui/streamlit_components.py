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


def log_level(log: dict) -> str | None:
    """Awareness level (LOW/MEDIUM/HIGH) of a verdict, or None for health records."""
    if "should_alert" not in log:
        return None
    return log.get("awareness_level")


def is_alert(log: dict) -> bool:
    """Whether a log entry is an actual alert (should_alert = 1)."""
    return str(log.get("should_alert")) == "1"


def filter_logs(
    logs: list[dict], levels: list[str] | None = None, alerts_only: bool = False
) -> list[dict]:
    """Filter log entries by awareness levels and/or alerts-only.

    An empty ``levels`` means "all levels". Health/error records (no verdict
    level) are kept unless ``alerts_only`` is set.
    """
    levels = set(levels or [])
    out = []
    for log in logs:
        level = log_level(log)
        if alerts_only:
            if not is_alert(log):
                continue
        if levels and (level is None or level not in levels):
            continue
        out.append(log)
    return out


def _colored_level(level: str) -> str:
    # Streamlit supports blue/green/orange/red/violet/gray/rainbow — NOT "yellow".
    label = t(f"awareness_{level}")
    if level == "LOW":
        return f":green[{label}]"
    if level == "MEDIUM":
        return f":orange[{label}]"
    return f":red[{label}]"


# Severity -> node color used by the timeline dots.
_NODE_COLORS = {
    "LOW": "#9e9e9e",
    "MEDIUM": "#f59e0b",
    "HIGH": "#e53935",
    None: "#8ab4f8",  # health/error records
}
_TIMELINE_CSS = """
<style>
@keyframes tl-fade {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.tl-node {
  position: relative;
  display: flex;
  justify-content: center;
  height: 100%;
  min-height: 44px;
}
.tl-dot {
  position: relative;
  z-index: 2;
  width: 13px;
  height: 13px;
  margin-top: 14px;
  border-radius: 50%;
  background: var(--c, #9e9e9e);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--c, #9e9e9e) 22%, transparent);
  animation: tl-fade 0.35s ease-out both;
}
.tl-dot.tl-strong {
  box-shadow: 0 0 0 5px color-mix(in srgb, var(--c) 30%, transparent),
    0 0 12px 2px color-mix(in srgb, var(--c) 55%, transparent);
}
.tl-line {
  position: absolute;
  top: 0;
  bottom: -0.9rem;
  left: 50%;
  width: 2px;
  transform: translateX(-50%);
  background: rgba(128, 128, 128, 0.28);
}
[data-testid="stExpander"] {
  animation: tl-fade 0.35s ease-out both;
}
</style>
"""


def _timeline_node(color: str, strong: bool) -> str:
    """HTML for one timeline node: a colored dot with a connecting rail segment."""
    cls = "tl-dot tl-strong" if strong else "tl-dot"
    return (
        f'<div class="tl-node" style="--c:{color}">'
        f'<div class="tl-line"></div>'
        f'<div class="{cls}"></div>'
        "</div>"
    )


def _render_node(color: str, strong: bool = False) -> None:
    st.markdown(_timeline_node(color, strong), unsafe_allow_html=True)


def _render_timeline_css() -> None:
    st.markdown(_TIMELINE_CSS, unsafe_allow_html=True)


def render_logs(logs: list[dict]):
    """Render the logs as a timeline (newest first) with a rail + severity dots.

    Each entry keeps a native collapsible expander carrying its photo(s); a
    colored node to its left marks the severity. Anomalous entries (an alert, or
    MEDIUM/HIGH awareness) default to expanded, normal ones to collapsed.
    """
    _render_timeline_css()

    for log in reversed(logs):  # newest first
        ts = log.get("timestamp")
        # Show to the second (a wall-clock review doesn't need microseconds).
        timestamp = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, dt.datetime) else ts

        node_col, body_col = st.columns([0.06, 0.94], vertical_alignment="top")

        # Health/error records: {timestamp, error|info} — no verdict, no photo.
        if "should_alert" not in log:
            with node_col:
                _render_node(_NODE_COLORS[None])
            with body_col:
                if "error" in log:
                    with st.expander(f"⚠️ {timestamp}", expanded=False):
                        st.write(t("error", error=log["error"]))
                elif "info" in log:
                    with st.expander(f"ℹ️ {timestamp}", expanded=False):
                        st.write(log["info"])
            continue

        level = log.get("awareness_level", "UNKNOWN")
        anomalous = _is_anomalous(log)
        with node_col:
            _render_node(_NODE_COLORS.get(level, _NODE_COLORS["LOW"]), strong=anomalous)

        alert_icon = "🚨" if is_alert(log) else "🟢"
        header = f"{alert_icon} {timestamp} · {_colored_level(level)}"
        if str(log.get("night_mode")) == "1":
            header += f" · 🌙 {t('night_mode_short')}"

        with body_col:
            with st.expander(header, expanded=anomalous):
                images = log.get("images") or []
                if images:
                    for camera, image in images:
                        if camera:
                            st.caption(t("camera_photo_label", camera=camera))
                        st.image(image, use_container_width=True)
                else:
                    st.caption(t("no_photo"))
                st.write(t("reasoning", reasoning=log.get("reasoning", "")))
