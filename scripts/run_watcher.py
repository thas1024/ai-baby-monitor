import argparse
import os
import time
from pathlib import Path

import structlog
from dotenv import load_dotenv
from playsound3 import playsound

from ai_baby_monitor.alerts import is_alert_worthy, should_emit_alert
from ai_baby_monitor.config import load_room_config_file
from ai_baby_monitor.imaging import encode_thumbnail
from ai_baby_monitor.keys import room_keys
from ai_baby_monitor.motion import frame_diff_score, should_analyze
from ai_baby_monitor.stream import RedisStreamHandler
from ai_baby_monitor.watcher import Watcher

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
    redis_stream_key: str,
    redis_host: str,
    redis_port: int,
    instructions: list[str],
    base_url: str,
    api_key: str,
    model_name: str,
    num_frames_to_process: int,
    api_mode: str = "chat",
    redis_password: str | None = None,
    motion_metric: str = "mean",
    motion_threshold: float | None = None,
    night_saturation_threshold: float | None = None,
    min_interval_seconds: float = 10.0,
    max_interval_seconds: float = 30.0,
    subsampled_stream_maxlen: int = 128,
):
    """
    Run the Watcher continuously to monitor frames from Redis stream.

    Request rate is controlled by motion gating (see ``ai_baby_monitor.motion``):
    a cycle only calls the model when the newest frame differs from the last
    analyzed frame, or the heartbeat interval elapses — floored by
    ``min_interval_seconds``. A ``motion_threshold`` of 0 disables gating.

    Args:
        redis_stream_key: Base Redis stream key (e.g., room name from config). Will use {key}:subsampled for video frames and {key}:logs for logs.
        redis_host: Redis server host.
        redis_port: Redis server port.
        instructions: List of monitoring instructions to check (from room config).
        base_url: Base URL of the OpenAI-compatible endpoint.
        api_key: API key for the endpoint.
        model_name: Model name to use for inference (from room config).
        num_frames_to_process: Number of frames to analyze when there is no previous window.
        api_mode: "chat" (Chat Completions) or "responses" (Responses API).
        redis_password: Redis password (None for servers without auth).
        motion_threshold: Pixel-difference threshold for motion gating/compression (0 = off).
        min_interval_seconds: Minimum seconds between model calls (rate floor).
        max_interval_seconds: Heartbeat ceiling — analyze at least this often even without motion.
        subsampled_stream_maxlen: Cap on frames fetched for catch-up (must cover max_interval).
    """
    # Initialize Redis stream handler
    redis_handler = RedisStreamHandler(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_password=redis_password,
    )

    # Initialize Watcher
    nanny_watcher = Watcher(
        instructions=instructions,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        api_mode=api_mode,
        motion_metric=motion_metric,
        motion_threshold=motion_threshold,
        night_saturation_threshold=night_saturation_threshold,
    )

    # Subsampled stream key
    keys = room_keys(redis_stream_key)
    subsampled_key = keys["subsampled"]
    logs_key = keys["logs"]
    logger.info(
        "Starting Watcher monitoring Redis",
        video_queue_key=subsampled_key,
        logs_queue_key=logs_key,
        motion_threshold=motion_threshold,
        motion_metric=motion_metric,
        min_interval_seconds=min_interval_seconds,
        max_interval_seconds=max_interval_seconds,
    )
    logger.info("Using model", model_name=model_name, base_url=base_url, api_mode=api_mode)
    logger.info("Monitoring instructions", instructions=instructions)

    # Motion-gating state. The baseline is the last frame of the last window that
    # was *successfully* analyzed, so a failed call is retried next cycle.
    last_analyzed_frame_data = None
    last_analyzed_ts = None
    last_call_monotonic = None

    # Recent alert verdicts, for debouncing the beep (a single flip is ignored).
    recent_verdicts: list[tuple[bool, float]] = []
    alert_gap_seconds = min(1800.0, max(300.0, max_interval_seconds * 3))

    try:
        while True:
            latest = redis_handler.get_latest_frames(subsampled_key, 1)
            if not latest:
                logger.warning(
                    "No frames available in stream", video_queue_key=subsampled_key
                )
                time.sleep(NO_FRAMES_SLEEP_SECONDS)
                continue

            newest_frame = latest[0]
            now_monotonic = time.monotonic()

            if not should_analyze(
                newest_frame.frame_data,
                last_analyzed_frame_data,
                now_monotonic,
                last_call_monotonic,
                motion_threshold,
                min_interval_seconds,
                max_interval_seconds,
                motion_metric,
            ):
                time.sleep(WATCHER_POLL_SECONDS)
                continue

            # Gate is open: collect the window. With a baseline, catch up on
            # everything since it (so no gap goes unexamined); otherwise take the
            # newest N frames.
            if last_analyzed_ts is not None:
                frames = redis_handler.get_frames_since(
                    subsampled_key, last_analyzed_ts, max_count=subsampled_stream_maxlen
                )
                if frames and frames[0].timestamp > last_analyzed_ts:
                    logger.warning(
                        "Subsampled stream does not cover the full gap since the last "
                        "analysis; increase subsampled_stream_maxlen",
                        needed=max_interval_seconds,
                        maxlen=subsampled_stream_maxlen,
                    )
            else:
                frames = redis_handler.get_latest_frames(
                    subsampled_key, num_frames_to_process
                )

            if not frames:
                frames = [newest_frame]

            motion_score = (
                frame_diff_score(
                    last_analyzed_frame_data, newest_frame.frame_data, motion_metric
                )
                if last_analyzed_frame_data is not None
                else None
            )
            logger.info(
                "Analyzing frames from stream",
                num_frames=len(frames),
                motion_score=motion_score,
            )

            last_call_monotonic = time.monotonic()
            result = nanny_watcher.process_frames(frames)

            if result["success"]:
                alert_status = (
                    "🚨 ALERT TRIGGERED"
                    if result["should_alert"]
                    else "✅ No alert needed"
                )
                awareness = result["recommended_awareness_level"]

                logger.info(
                    "Alert status and reasoning",
                    alert_status=alert_status,
                    awareness_level=awareness,
                    reasoning=result["reasoning"],
                    images_sent=result.get("images_sent"),
                    frames_skipped=result.get("frames_skipped"),
                )

                # Stream logs back to Redis
                night_mode = bool(result.get("night_mode"))
                log_data = {
                    "timestamp": time.time(),
                    "should_alert": int(result["should_alert"]),
                    "awareness_level": awareness,
                    "reasoning": result["reasoning"],
                    "night_mode": int(night_mode),
                    # Representative frame (last of the window) for human review in
                    # the timeline; stored as a small thumbnail to bound stream size.
                    "frame_timestamp": frames[-1].timestamp.isoformat(),
                }
                thumbnail = encode_thumbnail(frames[-1].frame_data)
                if thumbnail is not None:
                    log_data["frame"] = thumbnail
                redis_handler.add_logs(logs_key, log_data)

                # Advance the baseline only on success.
                last_analyzed_frame_data = bytes(frames[-1].frame_data)
                last_analyzed_ts = frames[-1].timestamp

                # Debounce the alert action: only beep once a violation is
                # confirmed by two consecutive positive verdicts (leading edge).
                # In night-vision mode, be less sensitive: only a HIGH-confidence
                # violation beeps (LOW/MEDIUM are still logged).
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

                # Stream error to Redis
                error_data = {
                    "timestamp": time.time(),
                    "error": error_msg,
                }
                redis_handler.add_logs(logs_key, error_data)

                time.sleep(NO_FRAMES_SLEEP_SECONDS)

    except KeyboardInterrupt:
        logger.info("Watcher stopped by user")
    except Exception as e:
        logger.error("Error in Watcher", error=e)


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
        # Load room configuration from file
        room_config = load_room_config_file(args.config_file)
        config_file_path = Path(args.config_file)
        logger.info(
            f"Loaded configuration for room: {room_config.name}",
            config_file=str(config_file_path.resolve()),
        )

        # Extract parameters from config
        redis_stream_key = room_config.name
        instructions = room_config.instructions
        num_frames_to_process = room_config.num_frames_to_process

        # Ensure instructions are provided, as RoomConfig defaults to an empty list if not in YAML.
        if not instructions:
            logger.error(
                "The 'instructions' list is empty or missing in the configuration file. At least one instruction is required for the watcher.",
                config_file=str(config_file_path.resolve()),
            )
            exit(1)

        # Ensure the LLM endpoint settings are present.
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
            redis_stream_key=redis_stream_key,
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            instructions=instructions,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            model_name=LLM_MODEL_NAME,
            num_frames_to_process=num_frames_to_process,
            api_mode=LLM_API_MODE,
            redis_password=REDIS_PASSWORD,
            motion_metric=room_config.motion_metric,
            motion_threshold=room_config.motion_threshold,
            night_saturation_threshold=room_config.night_saturation_threshold,
            min_interval_seconds=room_config.min_interval_seconds,
            max_interval_seconds=room_config.max_interval_seconds,
            subsampled_stream_maxlen=room_config.subsampled_stream_maxlen,
        )
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config_file}")
        exit(1)
    except Exception as e:
        logger.error("Failed to start watcher", error=str(e))
        exit(1)
