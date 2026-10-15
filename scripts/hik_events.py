#!/usr/bin/env python3
"""Publish camera-side HCNetSDK events (motion / smart) to Redis.

Runs a local SDK session for one camera, listens on the alarm channel, and on each
event (leading edge, throttled):

  * arms a short TTL gate at ``{room}:{camera}:gate`` so the watcher analyzes now;
  * appends to the ``{room}:{camera}:events`` stream for history/UI.

The watcher stays SDK-free; it only reads the gate (see ``run_watcher.py``). This
service needs the SDK and reaches the camera on its private port (8000).

    uv run python scripts/hik_events.py --config-file configs/living_room.yaml [--camera NAME]
"""

import argparse
import os
import time
from pathlib import Path

import structlog
from dotenv import load_dotenv

from ai_baby_monitor.config import load_room_config_file
from ai_baby_monitor.keys import camera_keys
from ai_baby_monitor.stream import hik_sdk
from ai_baby_monitor.stream import RedisStreamHandler

logger = structlog.get_logger()
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
GATE_TTL_SECONDS = 5
# Don't re-arm more often than this, so a burst of events doesn't flood.
MIN_EVENT_GAP_SECONDS = 2.0


def _select_camera(room_config, camera_name):
    if camera_name is None:
        return room_config.cameras[0]
    for cam in room_config.cameras:
        if cam.name == camera_name:
            return cam
    raise ValueError(f"Camera {camera_name!r} not in room {room_config.name}")


def run(room_config, camera) -> None:
    parsed = hik_sdk.parse_rtsp_uri(camera.uri)
    if parsed is None:
        raise ValueError("Event source needs an rtsp://user:pass@host URI to derive SDK credentials")

    handler = RedisStreamHandler(
        redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_password=REDIS_PASSWORD
    )
    keys = camera_keys(room_config.name, camera.name)
    session = hik_sdk.HikSession(
        host=parsed["host"],
        user=os.getenv("CAMERA_SDK_USER", parsed["user"]),
        password=os.getenv("CAMERA_SDK_PASSWORD", parsed["password"]),
        port=camera.sdk_port,
        channel=camera.channel,
    )
    if not session.login():
        logger.error("SDK login failed; no events", err=session.last_error())
        return

    state = {"last": 0.0}

    def on_event(kind: str, code: int) -> None:
        now = time.monotonic()
        handler.add_event(keys["events"], {"timestamp": time.time(), "type": kind})
        if now - state["last"] >= MIN_EVENT_GAP_SECONDS:
            handler.set_gate(keys["gate"], ttl_seconds=GATE_TTL_SECONDS)
            state["last"] = now
            logger.info("Event -> gate armed", kind=kind, cmd=f"0x{code:04x}")

    events = hik_sdk.HikEvents(session, on_event)
    if not events.start():
        logger.error("Could not open the alarm channel (device may not support it)")
        session.logout()
        return

    logger.info("Listening for camera events", room=room_config.name, camera=camera.name)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Stopping event listener")
    finally:
        events.stop()
        session.logout()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-file", required=True)
    p.add_argument("--camera", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    room_config = load_room_config_file(args.config_file)
    camera = _select_camera(room_config, args.camera)
    try:
        run(room_config, camera)
    except Exception as e:
        logger.error("Failed to start event listener", error=str(e))
        exit(1)
