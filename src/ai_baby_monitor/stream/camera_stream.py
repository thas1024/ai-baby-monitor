import datetime as dt
import os
import subprocess
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()

# Intel GPU decode device (VAAPI/QSV). The container needs it passed through.
_DRI_DEVICE = "/dev/dri/renderD128"

# Supported decode backends. "auto" prefers Intel hardware and falls back to CPU.
DECODE_MODES = ("auto", "cpu", "qsv", "vaapi")

# Assumed FPS for the stream writer when we can't ask the capture (ffmpeg path).
_DEFAULT_WRITER_FPS = 15.0


@dataclass
class Frame:
    frame_data: np.ndarray
    timestamp: dt.datetime
    frame_idx: int


def _ffmpeg_hwaccels(ffmpeg_bin: str) -> set[str]:
    """Return the set of hardware acceleration methods the ffmpeg build supports."""
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {line.strip() for line in result.stdout.splitlines()}
    except Exception as e:
        logger.warning("Could not query ffmpeg hwaccels", error=str(e))
        return set()


def resolve_decode(mode: str, ffmpeg_bin: str = "ffmpeg") -> str:
    """Resolve a decode mode to one of ``cpu`` / ``qsv`` / ``vaapi``.

    ``auto`` picks Intel QSV if available, else VAAPI (both require the render
    device), else CPU. Unknown values fall back to ``auto``.
    """
    if mode not in DECODE_MODES:
        logger.warning("Unknown CAMERA_DECODE, using auto", mode=mode)
        mode = "auto"
    if mode == "cpu":
        return "cpu"
    if mode == "auto":
        if not os.path.exists(_DRI_DEVICE):
            return "cpu"
        accels = _ffmpeg_hwaccels(ffmpeg_bin)
        # VAAPI is the most reliable Intel path (works with the iHD driver);
        # QSV needs the MFX/VPL runtime and is often absent.
        if "vaapi" in accels:
            return "vaapi"
        if "qsv" in accels:
            return "qsv"
        logger.warning("No Intel hwaccel in ffmpeg; using CPU", hwaccels=sorted(accels))
        return "cpu"
    return mode


class _FfmpegDecoder:
    """Decodes a stream via an ffmpeg subprocess (optionally Intel hwaccel).

    Runs a background thread that keeps the newest frame in a single slot, and
    restarts ffmpeg if the pipeline ends. Only the latest frame is kept so a slow
    consumer never falls behind.
    """

    def __init__(
        self,
        uri: str,
        mode: str,
        frame_shape: tuple[int, int],
        ffmpeg_bin: str = "ffmpeg",
    ):
        self.uri = uri
        self.mode = mode
        self.width, self.height = frame_shape
        self.ffmpeg_bin = ffmpeg_bin
        self._frame_size = self.width * self.height * 3

        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._first_frame = threading.Event()
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._run, name="ffmpeg-decode", daemon=True)
        self._thread.start()

    def _build_cmd(self) -> list[str]:
        cmd = [self.ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
        if self.uri.startswith("rtsp"):
            cmd += ["-rtsp_transport", "tcp"]
        cmd += ["-fflags", "nobuffer", "-flags", "low_delay"]
        if self.mode == "qsv":
            cmd += ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
        elif self.mode == "vaapi":
            cmd += [
                "-hwaccel", "vaapi",
                "-hwaccel_device", _DRI_DEVICE,
                "-hwaccel_output_format", "vaapi",
            ]
        cmd += [
            "-i", self.uri,
            "-an",
            # Decoded frames stay on the GPU; hwdownload brings them back to system
            # memory so a normal scale/format filters to bgr24 for the pipe.
            "-vf", f"hwdownload,format=nv12,scale={self.width}:{self.height},format=bgr24",
            "-f", "rawvideo",
            "-",
        ]
        return cmd

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._proc = subprocess.Popen(
                    self._build_cmd(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=10**7,
                )
            except Exception as e:
                logger.error("ffmpeg start failed", error=str(e), mode=self.mode)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue

            logger.info("ffmpeg capture started", mode=self.mode, pid=self._proc.pid)
            backoff = 1.0
            self._read_frames()
            if self._stop.is_set():
                break
            logger.warning("ffmpeg stream ended; reconnecting", mode=self.mode)
            self._terminate_proc()
            time.sleep(backoff)

    def _read_exact(self, n: int) -> bytes | None:
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0 and not self._stop.is_set():
            chunk = self._proc.stdout.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks) if chunks else None

    def _read_frames(self) -> None:
        while not self._stop.is_set():
            buf = self._read_exact(self._frame_size)
            if buf is None:
                return
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )
            with self._lock:
                self._latest = frame
            self._first_frame.set()

    def wait_first_frame(self, timeout: float) -> bool:
        """Block until a frame arrives or ``timeout`` elapses (for startup check)."""
        return self._first_frame.wait(timeout)

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    def _terminate_proc(self) -> None:
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        self._proc = None

    def stop(self) -> None:
        self._stop.set()
        self._terminate_proc()
        if self._thread.is_alive():
            self._thread.join(timeout=5)


class CameraStream:
    def __init__(
        self,
        uri: str | int,
        save_stream_path: str | None = None,
        frame_shape: tuple[int, int] | None = None,
        decode: str = "auto",
    ):
        """
        Simple camera stream handler.

        Args:
            uri: Camera URI or device index.
            save_stream_path: Path to save the stream.
            frame_shape: Frame shape (width, height) to capture/resize to.
            decode: Frame decoding backend. "cpu" uses OpenCV decode; "qsv" /
                "vaapi" decode on an Intel GPU via ffmpeg; "auto" prefers Intel
                hardware and falls back to CPU.
        """
        self.frame_shape = frame_shape
        self.frame_idx = 0
        self.uri = uri
        self._ffmpeg_bin = os.getenv("FFMPEG_BIN", "ffmpeg")
        self._decoder: _FfmpegDecoder | None = None
        self.decode = self._init_decode(decode)
        self.capture = self._init_capture(uri) if self.decode == "cpu" else None
        self.stream_writer = self._init_stream_writer(save_stream_path)

    def _init_decode(self, decode: str) -> str:
        # ffmpeg hwaccel only makes sense for a stream URL, not a webcam index.
        if decode == "cpu" or not isinstance(self.uri, str) or self.uri.isdigit():
            return "cpu"

        mode = resolve_decode(decode, self._ffmpeg_bin)
        if mode == "cpu":
            return "cpu"

        shape = self.frame_shape or (640, 360)
        try:
            decoder = _FfmpegDecoder(self.uri, mode, shape, self._ffmpeg_bin)
            if decoder.wait_first_frame(timeout=20.0):
                self._decoder = decoder
                logger.info("Using hardware decode", mode=mode, uri=self.uri)
                return mode
            decoder.stop()
            logger.warning("No frame from hardware decode; falling back to CPU", mode=mode)
        except Exception as e:
            logger.warning("Hardware decode failed; falling back to CPU", mode=mode, error=str(e))
        return "cpu"

    def _init_capture(self, uri: str | int, max_retries: int = 3) -> cv2.VideoCapture:
        """Initialize and return the camera capture."""
        retries = 0
        while retries < max_retries:
            try:
                capture = cv2.VideoCapture(uri)
                if capture.isOpened():
                    logger.info("Successfully connected to RTSP stream", uri=uri)
                    return capture
                raise ConnectionError("Failed to open RTSP stream")
            except Exception as e:
                retries += 1
                logger.error(
                    "Reconnect attempt",
                    uri=uri,
                    retry_attempt=retries,
                    max_retries=max_retries,
                    error=str(e),
                )
                time.sleep(2)
        raise ConnectionError(f"Failed to connect to camera stream: {uri}")

    def _init_stream_writer(
        self, save_stream_path: str | None
    ) -> cv2.VideoWriter | None:
        """Initialize and return the stream writer if a path is provided."""
        if not save_stream_path:
            return None
        fps = (
            self.capture.get(cv2.CAP_PROP_FPS)
            if self.capture is not None
            else _DEFAULT_WRITER_FPS
        )
        if not fps or fps <= 0:
            fps = _DEFAULT_WRITER_FPS
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(save_stream_path, fourcc, fps, self.frame_shape)

    def _grab_raw_frame(self) -> np.ndarray | None:
        """Get the next available frame as a BGR array (no JPEG encoding yet)."""
        if self._decoder is not None:
            return self._decoder.latest()
        if not self.capture.grab():
            logger.warning("No frame available to grab")
            return None
        ret, frame = self.capture.retrieve()
        if not ret:
            logger.warning("Failed to retrieve grabbed frame")
            return None
        return frame

    def capture_new_frame(self) -> Frame | None:
        """Capture a new frame from the camera, encode into jpeg."""
        frame = self._grab_raw_frame()
        if frame is None:
            return None

        timestamp = dt.datetime.now()

        # Resize frame if needed
        if self.frame_shape and (frame.shape[1], frame.shape[0]) != self.frame_shape:
            frame = cv2.resize(frame, self.frame_shape)

        # Write frame to stream writer if it exists
        if self.stream_writer:
            self.stream_writer.write(frame)

        # Convert into jpeg
        ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        if not ret:
            logger.warning("Failed to encode frame as JPEG")
            return None

        frame_obj = Frame(
            frame_data=jpeg,
            timestamp=timestamp,
            frame_idx=self.frame_idx,
        )
        self.frame_idx += 1
        return frame_obj

    def close(self):
        """Close the camera capture, decoder and stream writer."""
        if self._decoder is not None:
            self._decoder.stop()
        if self.capture is not None:
            self.capture.release()
        if self.stream_writer:
            self.stream_writer.release()
