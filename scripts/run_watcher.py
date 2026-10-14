import argparse
import os
import time
from pathlib import Path

import structlog
from dotenv import load_dotenv
from playsound3 import playsound

from ai_baby_monitor.alerts import is_alert_worthy, should_emit_alert
from ai_baby_monitor.config import CameraConfig, load_room_config_file
from ai_baby_monitor.imaging import encode_thumbnail
from ai_baby_monitor.keys import camera_keys, room_keys
from ai_baby_monitor.motion import frame_diff_score, is_blank_frame, should_analyze
from ai_baby_monitor.stream import RedisStreamHandler
from ai_baby_monitor.watcher import CameraWindow, Watcher

logger = structlog.get_logger()

load_dotenv()
# Redis is external (see .env); the watcher uses the same REDIS_HOST as the
# containers. When run in Docker it resolves to the container's view of Redis.
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None

# The watcher calls an OpenAI-compatible endpoint (cloud or local) over the
# network. See .env.template for the LLM_* variables.
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
# "chat" (Chat Completions, default) or "responses" (OpenAI Responses API).
LLM_API_MODE = os.getenv("LLM_API_MODE", "chat")
# Play the alert beep on a rule violation. Set to false in headless/container
# deployments where there is no audio output.
ALERT_SOUND_ENABLED = os.getenv("ALERT_SOUND_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)

# How often the gated loop checks the newest frame when it is not due yet.
WATCHER_POLL_SECONDS = 0.3
# Idle wait when there are no frames at all.
NO_FRAMES_SLEEP_SECONDS = 0.3


def _play_alert(sound_path: str = "assets/alert.wav") -> None:
    """Play the alert beep, tolerating a missing audio backend.

    Never raises: a headless host or container has no audio output, and a failed
    beep must not stop the watcher.
    """
    if not ALERT_SOUND_ENABLED:
        logger.info("Alert beep suppressed", reason="ALERT_SOUND_ENABLED is off")
        return
    try:
        playsound(sound_path)
    except Exception as e:
        logger.warning("Could not play alert sound", error=str(e))


def run_watcher(
    room_name: str,
    cameras: list[CameraConfig],
    instructions: list[str],
    base_url: str,
    api_key: str,
    model_name: str,
    num_frames_to_process: int,
    redis_host: str,
    redis_port: int,
    api_mode: str = "chat",
    redis_password: str | None = None,
    min_interval_seconds: float = 10.0,
    max_interval_seconds: float = 30.0,
):
    """
    Run the Watcher continuously to monitor a room's cameras from Redis streams.

    One LLM request merges all cameras: when *any* camera's motion gate opens (and
    the room-wide minimum interval has elapsed, or the heartbeat fires), every
    camera's window is fetched and sent together, each labeled with its camera
    name, so the model sees all angles at once.

    Args:
        room_name: Room name (Redis key base).
        cameras: The room's cameras (each with its own uri-independent knobs).
        instructions: Natural-language rules the model checks against.
        base_url / api_key / model_name / api_mode: LLM endpoint settings.
        num_frames_to_process: Room-level default frames per window.
        redis_host / redis_port / redis_password: external Redis connection.
        min_interval_seconds: Room-wide minimum seconds between model calls.
        max_interval_seconds: Heartbeat — analyze at least this often without motion.
    """
    redis_handler = RedisStreamHandler(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
    )

    nanny_watcher = Watcher(
        instructions=instructions,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        api_mode=api_mode,
    )

    logs_key = room_keys(room_name)["logs"]
    subsampled_keys = {
        cam.name: camera_keys(room_name, cam.name)["subsampled"] for cam in cameras
    }
    logger.info(
        "Starting Watcher monitoring Redis",
        room=room_name,
        cameras=[cam.name for cam in cameras],
        logs_queue_key=logs_key,
        min_interval_seconds=min_interval_seconds,
        max_interval_seconds=max_interval_seconds,
    )
    logger.info("Using model", model_name=model_name, base_url=base_url, api_mode=api_mode)
    logger.info("Monitoring instructions", instructions=instructions)

    # Per-camera gate state. A baseline is the last frame of the last *successfully*
    # analyzed window for that camera, so a failed call is retried next cycle.
    state = {cam.name: {"frame_data": None, "ts": None} for cam in cameras}
    last_call_monotonic: float | None = None

    # Recent alert verdicts, for debouncing the beep (a single flip is ignored).
    recent_verdicts: list[tuple[bool, float]] = []
    alert_gap_seconds = min(1800.0, max(300.0, max_interval_seconds * 3))

    # Leading-edge tracking for the "no live video" health event.
    blank_active = False

    while True:
        try:
            while True:
                now_monotonic = time.monotonic()

                # Gather each camera's newest frame and evaluate its own motion gate.
                newest = {}
                any_gate_open = False
                for cam in cameras:
                    latest = redis_handler.get_latest_frames(subsampled_keys[cam.name], 1)
                    if not latest:
                        continue
                    newest[cam.name] = latest[0]
                    if should_analyze(
                        latest[0].frame_data,
                        state[cam.name]["frame_data"],
                        now_monotonic,
                        last_call_monotonic,
                        cam.motion_threshold,
                        min_interval_seconds,
                        max_interval_seconds,
                        cam.motion_metric,
                    ):
                        any_gate_open = True

                if not newest:
                    logger.warning(
                        "No frames available in any camera stream", room=room_name
                    )
                    time.sleep(NO_FRAMES_SLEEP_SECONDS)
                    continue

                # Apply the room-wide rate floor here, outside the per-camera OR, so a
                # camera with no baseline can't bypass it (should_analyze returns True
                # before the floor check when the baseline is missing).
                floor_elapsed = last_call_monotonic is None or (
                    now_monotonic - last_call_monotonic >= min_interval_seconds
                )
                if not any_gate_open or not floor_elapsed:
                    time.sleep(WATCHER_POLL_SECONDS)
                    continue

                # Gate is open: fetch every camera's window so the merged request is a
                # coherent room snapshot. With a baseline, catch up since it (no gap
                # skipped); otherwise take the newest N frames.
                windows: list[CameraWindow] = []
                motion_scores: dict[str, float | None] = {}
                for cam in cameras:
                    if cam.name not in newest:
                        continue
                    key = subsampled_keys[cam.name]
                    if state[cam.name]["ts"] is not None:
                        frames = redis_handler.get_frames_since(
                            key,
                            state[cam.name]["ts"],
                            max_count=cam.subsampled_stream_maxlen,
                        )
                        if frames and frames[0].timestamp > state[cam.name]["ts"]:
                            logger.warning(
                                "Subsampled stream does not cover the full gap since the "
                                "last analysis; increase subsampled_stream_maxlen",
                                camera=cam.name,
                                needed=max_interval_seconds,
                                maxlen=cam.subsampled_stream_maxlen,
                            )
                    else:
                        frames = redis_handler.get_latest_frames(
                            key, cam.num_frames_to_process or num_frames_to_process
                        )
                    if not frames:
                        frames = [newest[cam.name]]

                    motion_scores[cam.name] = (
                        frame_diff_score(
                            state[cam.name]["frame_data"],
                            newest[cam.name].frame_data,
                            cam.motion_metric,
                        )
                        if state[cam.name]["frame_data"] is not None
                        else None
                    )
                    windows.append(
                        CameraWindow(
                            frames=frames,
                            name=cam.name,
                            motion_metric=cam.motion_metric,
                            motion_threshold=cam.motion_threshold,
                            night_threshold=cam.night_saturation_threshold,
                        )
                    )

                # Drop cameras whose current frame is flat/blank — a dead feed. Don't
                # send a useless gray frame to the model, don't advance its baseline.
                dead_cameras = [
                    w.name for w in windows if is_blank_frame(w.frames[-1].frame_data)
                ]
                if dead_cameras:
                    logger.warning("Cameras with no signal (blank frames)", cameras=dead_cameras)
                    windows = [w for w in windows if w.name not in dead_cameras]

                # Only record the "no live video" health event on the leading edge (and
                # a matching recovery), so a long outage does not flood the logs stream.
                if not windows:
                    logger.warning("All cameras blank; skipping analysis", room=room_name)
                    if not blank_active:
                        blank_active = True
                        redis_handler.add_logs(
                            logs_key,
                            {
                                "timestamp": time.time(),
                                "error": "no live video: all cameras returned blank/flat "
                                "frames (stream likely failed)",
                            },
                        )
                    # Throttle these polls to the room's min interval.
                    last_call_monotonic = time.monotonic()
                    time.sleep(WATCHER_POLL_SECONDS)
                    continue

                if blank_active:
                    blank_active = False
                    redis_handler.add_logs(
                        logs_key,
                        {"timestamp": time.time(), "info": "video restored (live frames again)"},
                    )

                logger.info(
                    "Analyzing cameras",
                    cameras=[w.name for w in windows],
                    num_frames={w.name: len(w.frames) for w in windows},
                    motion_scores=motion_scores,
                )

                last_call_monotonic = time.monotonic()
                result = nanny_watcher.process_room(windows)

                if result["success"]:
                    alert_status = (
                        "🚨 ALERT TRIGGERED"
                        if result["should_alert"]
                        else "✅ No alert needed"
                    )
                    awareness = result["recommended_awareness_level"]
                    night_mode = bool(result.get("night_mode"))

                    logger.info(
                        "Alert status and reasoning",
                        alert_status=alert_status,
                        awareness_level=awareness,
                        night_mode=night_mode,
                        reasoning=result["reasoning"],
                        images_sent=result.get("images_sent"),
                        frames_skipped=result.get("frames_skipped"),
                    )

                    # Stream the verdict + a representative thumbnail per camera.
                    log_data: dict = {
                        "timestamp": time.time(),
                        "should_alert": int(result["should_alert"]),
                        "awareness_level": awareness,
                        "reasoning": result["reasoning"],
                        "night_mode": int(night_mode),
                    }
                    for window in windows:
                        last_frame = window.frames[-1]
                        thumb_field = "frame" if window.name is None else f"frame__{window.name}"
                        ts_field = (
                            "frame_timestamp"
                            if window.name is None
                            else f"frame_timestamp__{window.name}"
                        )
                        log_data[ts_field] = last_frame.timestamp.isoformat()
                        thumbnail = encode_thumbnail(last_frame.frame_data)
                        if thumbnail is not None:
                            log_data[thumb_field] = thumbnail
                    redis_handler.add_logs(logs_key, log_data)

                    # Advance every contributing camera's baseline (only on success).
                    for window in windows:
                        state[window.name]["frame_data"] = bytes(window.frames[-1].frame_data)
                        state[window.name]["ts"] = window.frames[-1].timestamp

                    # Debounce the beep; in night-vision mode only a HIGH-confidence
                    # violation qualifies (LOW/MEDIUM are still logged).
                    alert_worthy = is_alert_worthy(
                        result["should_alert"], night_mode, awareness
                    )
                    recent_verdicts.append((bool(alert_worthy), time.monotonic()))
                    del recent_verdicts[:-5]
                    if alert_worthy and should_emit_alert(recent_verdicts, alert_gap_seconds):
                        _play_alert()
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.error("Error processing frames", error=error_msg)
                    redis_handler.add_logs(
                        logs_key, {"timestamp": time.time(), "error": error_msg}
                    )
                    time.sleep(NO_FRAMES_SLEEP_SECONDS)

        except KeyboardInterrupt:
            logger.info("Watcher stopped by user")
            break
        except Exception as e:
            # A transient failure (e.g. Redis or the LLM endpoint not up yet at
            # boot) must not kill the watcher: log and retry the loop.
            logger.error("Watcher loop error; restarting", error=str(e))
            time.sleep(LOOP_RESTART_SECONDS)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Watcher to monitor Redis stream frames based on room configuration."
    )

    parser.add_argument(
        "--config-file", required=True, help="Path to room configuration YAML file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        room_config = load_room_config_file(args.config_file)
        config_file_path = Path(args.config_file)
        logger.info(
            f"Loaded configuration for room: {room_config.name}",
            config_file=str(config_file_path.resolve()),
            cameras=[cam.name for cam in room_config.cameras],
        )

        if not room_config.instructions:
            logger.error(
                "The 'instructions' list is empty or missing in the configuration file. At least one instruction is required for the watcher.",
                config_file=str(config_file_path.resolve()),
            )
            exit(1)

        missing_env = [
            name
            for name, value in (
                ("LLM_BASE_URL", LLM_BASE_URL),
                ("LLM_API_KEY", LLM_API_KEY),
                ("LLM_MODEL_NAME", LLM_MODEL_NAME),
            )
            if not value
        ]
        if missing_env:
            logger.error(
                "Missing required environment variables for the LLM endpoint",
                missing=missing_env,
                hint="Copy .env.template to .env and fill in the LLM_* values.",
            )
            exit(1)

        run_watcher(
            room_name=room_config.name,
            cameras=room_config.cameras,
            instructions=room_config.instructions,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            model_name=LLM_MODEL_NAME,
            num_frames_to_process=room_config.num_frames_to_process,
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            api_mode=LLM_API_MODE,
            redis_password=REDIS_PASSWORD,
            min_interval_seconds=room_config.min_interval_seconds,
            max_interval_seconds=room_config.max_interval_seconds,
        )
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config_file}")
        exit(1)
    except Exception as e:
        logger.error("Failed to start watcher", error=str(e))
        exit(1)
