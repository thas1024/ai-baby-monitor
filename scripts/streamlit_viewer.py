import argparse
import os

import streamlit as st
import structlog
from dotenv import load_dotenv

from ai_baby_monitor.config import load_multiple_room_configs
from ai_baby_monitor.i18n import t
from ai_baby_monitor.ui import (
    display_sidebar,
    fetch_logs,
    get_cached_redis_handler,
    get_camera_images,
    render_logs,
)

load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None

logger = structlog.get_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Baby Monitor Stream Viewer")
    parser.add_argument(
        "--config-files",
        nargs="+",  # Accept one or more config files
        required=True,
        help="Paths to room configuration YAML files",
    )

    return parser.parse_known_args()[0]


st.set_page_config(
    page_title=t("app_title"),
    page_icon="👶",
)

# Storing room configs in state to avoid reparsing
if "room_configs" not in st.session_state:
    args = parse_args()
    room_configs = load_multiple_room_configs(args.config_files)
    if not room_configs:
        st.error(t("no_room_config"))
    st.session_state["room_configs"] = room_configs

redis_handler = get_cached_redis_handler(REDIS_HOST, REDIS_PORT, REDIS_PASSWORD)

# Sidebar has a side-effect of setting selected config and real-time stream vs historic logs
display_sidebar(
    room_configs_key="room_configs",
    config_key="selected_config",
    mode_key="selected_mode",
)
selected_config = st.session_state["selected_config"]
selected_mode = st.session_state["selected_mode"]

if selected_mode == "realtime":
    st.title(t("app_title"))

    # Placeholders for the stream and logs
    stream_placeholder = st.empty()
    log_placeholder = st.empty()

    while True:
        with stream_placeholder.container():
            cameras = get_camera_images(
                redis_handler, selected_config.name, selected_config.cameras
            )
            for label, image, timestamp in cameras:
                st.caption(t("camera_photo_label", camera=label))
                if image:
                    st.image(image, use_container_width=True)
                    with st.expander(t("frame_info")):
                        st.caption(t("timestamp", ts=timestamp))
                else:
                    st.caption(t("no_photo"))

        with log_placeholder.container(height=350):
            # No outer expander here: render_logs opens its own expander per log
            # entry, and Streamlit forbids nesting expanders.
            st.subheader(t("llm_logs"), divider=False)
            logs = fetch_logs(
                redis_handler,
                selected_config.name,
                num_logs=1,
            )

            render_logs(logs)

else:
    st.title(t("logs_viewer_title"))

    num_logs = st.number_input(
        t("num_logs_label"),
        min_value=1,
        value=100,
        step=1,
    )
    if st.button(t("fetch_logs_button"), type="primary"):
        # render_logs shows newest-first itself; fetch_logs returns chronological.
        logs = fetch_logs(redis_handler, selected_config.name, num_logs=num_logs)
        with st.container(height=800):
            render_logs(logs)
