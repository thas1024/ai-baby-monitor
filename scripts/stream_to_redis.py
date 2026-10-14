import argparse
import os
import time
from pathlib import Path

import structlog
from dotenv import load_dotenv

from ai_baby_monitor.config import CameraConfig, load_room_config_file
from ai_baby_monitor.keys import camera_keys
from ai_baby_monitor.stream import CameraStream, RedisStreamHandler

logger = structlog.get_logger()

load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
# Frame decode backend: cpu | qsv | vaapi | auto (auto prefers Intel GPU).
CAMERA_DECODE = os.getenv("CAMERA_DECODE", "auto")


def stream_to_redis(
    camera_uri: str,
    redis_stream_key: str,
    redis_host: str,
    redis_port: int,
    subsampled_stream_maxlen: int,
    save_stream_path: str | None,
    frame_width: int | None,
    frame_height: int | None,
    subsample_rate: int = 4,
    redis_password: str | None = None,
    camera_name: str | None = None,
    decode: str = "auto",
):
    """
    Stream camera frames to Redis short realtime and long subsampled queues.

    This function is used by the main entry point and expects configuration
    parameters to be loaded from the room's YAML config file.
    """
    # Convert camera_uri to int if it's a digit (for webcam index)
    if camera_uri.isdigit():
        camera_uri = int(camera_uri)

    # Set frame shape if both dimensions are provided
    frame_shape = None
    if frame_width is not None and frame_height is not None:
        frame_shape = (frame_width, frame_height)

    logger.info("Initializing camera stream", camera_uri=camera_uri, camera=camera_name)
    logger.info(
        "Streaming to Redis",
        redis_stream_key=redis_stream_key,
        redis_host=redis_host,
        redis_port=redis_port,
    )
    logger.info(
        f"Using subsample rate of 1 out of {subsample_rate} for subsampled queue"
    )

    try:
        # Initialize camera stream
        camera = CameraStream(
            uri=camera_uri,
            save_stream_path=save_stream_path,
            frame_shape=frame_shape,
            decode=decode,
        )
        logger.info("Frame decode backend", requested=decode, active=camera.decode)

        # Initialize Redis stream handler
        redis_handler = RedisStreamHandler(
            redis_host=redis_host,
            redis_port=redis_port,
            redis_password=redis_password,
        )

        logger.info("Starting to stream to redis.")

        keys = camera_keys(redis_stream_key, camera_name)

        # Main streaming loop
        while True:
            # Only capture when a new frame is available
            frame = camera.capture_new_frame()
            if frame:
                # Always add frame to realtime queue
                redis_handler.add_frame(frame, keys["realtime"], 3, approximate=False)

                # Add to subsampled queue every nth frame
                if frame.frame_idx % subsample_rate == 0:
                    redis_handler.add_frame(
                        frame,
                        keys["subsampled"],
                        subsampled_stream_maxlen,
                    )
                    logger.info(
                        "Added frame to subsampled queue",
                        frame_idx=frame.frame_idx,
                        timestamp=frame.timestamp,
                    )
            else:
                logger.warning("Failed to capture frame, retrying...")
                time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("Stream interrupted by user")
    except Exception as e:
        logger.error("Error in streaming", error=e)
    finally:
        if "camera" in locals():
            camera.close()
        logger.info("Stream closed")


def parse_args():
    """Parse command line arguments for room-based configuration."""
    parser = argparse.ArgumentParser(
        description="Stream camera frames to Redis based on room configuration."
    )

    parser.add_argument(
        "--config-file", required=True, help="Path to room configuration YAML file"
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Which camera/angle to capture (its 'name' in the room config). "
        "Optional for a legacy single-camera config.",
    )
    parser.add_argument(
        "--save-stream-path",
        default=None,
        help="Path to save the video stream (optional)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Override camera URI with demo footage if available",
    )

    return parser.parse_args()


def _select_camera(room_config, camera_name: str | None) -> CameraConfig:
    """Pick the camera to capture: by ``--camera NAME``, else the only one."""
    cameras = room_config.cameras
    if camera_name is None:
        if len(cameras) != 1:
            names = [c.name for c in cameras]
            raise ValueError(
                f"Room '{room_config.name}' has {len(cameras)} cameras; pass --camera "
                f"with one of {names}"
            )
        return cameras[0]

    for camera in cameras:
        if camera.name == camera_name:
            return camera
    raise ValueError(
        f"Camera '{camera_name}' not found in room '{room_config.name}'; "
        f"available: {[c.name for c in cameras]}"
    )


if __name__ == "__main__":
    args = parse_args()

    # Load room configuration from file
    try:
        room_config = load_room_config_file(args.config_file)
        config_file_path = Path(args.config_file)
        logger.info(
            f"Loaded configuration for room: {room_config.name}",
            config_file=str(config_file_path.resolve()),
        )

        # Select the camera/angle this process captures.
        camera = _select_camera(room_config, args.camera)

        camera_uri = camera.uri
        if args.demo:
            camera_uri = "assets/demo/demo.mp4"
            logger.info("Using demo video source", camera_uri=camera_uri)

        stream_to_redis(
            camera_uri=camera_uri,
            redis_stream_key=room_config.name,
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            subsampled_stream_maxlen=camera.subsampled_stream_maxlen,
            save_stream_path=args.save_stream_path,
            frame_width=camera.frame_width,
            frame_height=camera.frame_height,
            subsample_rate=camera.subsample_rate,
            redis_password=REDIS_PASSWORD,
            camera_name=camera.name,
            decode=CAMERA_DECODE,
        )
    except Exception as e:
        logger.error("Failed to start streaming", error=e)
        exit(1)
