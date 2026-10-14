"""Frame-motion gating and window timeline compression.

Pure image math — no Redis, config or watcher imports — so it is cheap to unit
test and reuse. Two jobs:

* ``frame_diff_score`` / ``frames_are_similar`` — how different two JPEG frames
  are, on a fixed preprocessing pipeline so a threshold value means the same
  thing across runs and cameras.
* ``build_timeline`` — turn a window of frames into a list of ``ImageItem``
  (frames worth showing the model) and ``SkippedItem`` (runs of near-identical
  frames, replaced by a text note). ``should_analyze`` — whether to call the
  model at all this cycle.
"""

import datetime as dt
import math
from dataclasses import dataclass

import cv2
import numpy as np
import structlog

from ai_baby_monitor.stream import Frame

logger = structlog.get_logger()

# Preprocessing is pinned so the threshold is meaningful. Do not change these
# without revisiting the default thresholds documented below.
_PROCESS_WIDTH = 160
_PROCESS_HEIGHT = 90
_BLUR_KERNEL = (5, 5)
_GRID_ROWS = 9
_GRID_COLS = 10
# A pixel counts as "changed" (changed_fraction metric) above this abs-diff, so
# sensor/JPEG noise below it is ignored.
_CHANGED_NOISE_FLOOR = 8

# Supported metrics. Their score scales differ, so a threshold must never be
# reused across metrics (see ``resolve_threshold``/``DEFAULT_THRESHOLDS``).
METRICS = ("mean", "grid_max", "changed_fraction")
DEFAULT_THRESHOLDS: dict[str, float] = {
    "mean": 3.0,  # mean abs diff over the whole frame (0-255)
    "grid_max": 6.0,  # max per-cell mean abs diff (0-255) — sensitive to small motion
    "changed_fraction": 0.01,  # fraction of pixels above the noise floor (0-1)
}


def resolve_threshold(metric: str, threshold: float | None) -> float:
    """Return ``threshold`` if set, else the default for ``metric``.

    ``None`` means "use the metric's default"; an explicit ``0`` means "disabled".
    """
    if threshold is not None:
        return threshold
    return DEFAULT_THRESHOLDS.get(metric, DEFAULT_THRESHOLDS["mean"])


def _resolve_metric(metric: str) -> str:
    if metric not in METRICS:
        logger.warning("Unknown motion metric, falling back to 'mean'", metric=metric)
        return "mean"
    return metric


def _decode_small_gray(jpeg_bytes: bytes) -> np.ndarray | None:
    """Decode JPEG bytes to a small blurred grayscale image, or None on failure."""
    array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (_PROCESS_WIDTH, _PROCESS_HEIGHT))
    return cv2.GaussianBlur(small, _BLUR_KERNEL, 0)


def frame_diff_score(prev_frame_data, cur_frame_data, metric: str = "mean") -> float:
    """How different two JPEG frames are, on the pipeline pinned above.

    Returns 0.0 for byte-identical frames (fast path) and ``math.inf`` when a
    frame cannot be decoded — an undecodable frame must be treated as "changed",
    never crash the caller. The score scale depends on ``metric``.
    """
    prev_bytes = bytes(prev_frame_data)
    cur_bytes = bytes(cur_frame_data)
    if prev_bytes == cur_bytes:
        return 0.0

    prev_gray = _decode_small_gray(prev_bytes)
    cur_gray = _decode_small_gray(cur_bytes)
    if prev_gray is None or cur_gray is None:
        return math.inf

    diff = cv2.absdiff(prev_gray, cur_gray)
    metric = _resolve_metric(metric)

    if metric == "grid_max":
        # Split into cells and take the busiest cell, so a small localized change
        # is not diluted by a large static background.
        cells = diff.reshape(
            _GRID_ROWS,
            _PROCESS_HEIGHT // _GRID_ROWS,
            _GRID_COLS,
            _PROCESS_WIDTH // _GRID_COLS,
        )
        return float(cells.mean(axis=(1, 3)).max())

    if metric == "changed_fraction":
        return float(np.mean(diff > _CHANGED_NOISE_FLOOR))

    return float(np.mean(diff))


def frames_are_similar(
    prev_frame_data, cur_frame_data, threshold: float, metric: str = "mean"
) -> bool:
    """Whether two frames differ by no more than ``threshold``.

    A non-positive ``threshold`` disables the comparison (always "not similar"),
    which reproduces the pre-gating behaviour of sending every frame.
    """
    if threshold is None:
        raise ValueError("threshold must be resolved before comparison")
    if threshold <= 0:
        return False
    return frame_diff_score(prev_frame_data, cur_frame_data, metric) <= threshold


@dataclass
class ImageItem:
    """A frame to send to the model, at position ``index`` in the window."""

    frame: Frame
    index: int


@dataclass
class SkippedItem:
    """A run of near-identical frames collapsed into a text note."""

    start_ts: dt.datetime
    end_ts: dt.datetime
    count: int
    first_index: int
    last_index: int


TimelineItem = ImageItem | SkippedItem


def build_timeline(
    frames: list[Frame], threshold: float | None, metric: str = "mean"
) -> list[TimelineItem]:
    """Compress ``frames`` into image items plus skipped-run notes.

    The first frame is always an image (the anchor). Each later frame is compared
    against the last frame that was actually included, so slow drift accumulates
    until it crosses the threshold. A window always yields at least one image.
    """
    if not frames:
        return []

    resolved = resolve_threshold(metric, threshold)

    items: list[TimelineItem] = [ImageItem(frame=frames[0], index=0)]
    anchor = frames[0]

    skip_start: dt.datetime | None = None
    skip_end: dt.datetime | None = None
    skip_first_index = 0
    skip_count = 0

    def flush_skip() -> None:
        nonlocal skip_start, skip_end, skip_count
        if skip_count:
            items.append(
                SkippedItem(
                    start_ts=skip_start,
                    end_ts=skip_end,
                    count=skip_count,
                    first_index=skip_first_index,
                    last_index=skip_first_index + skip_count - 1,
                )
            )
        skip_start = skip_end = None
        skip_count = 0

    for index in range(1, len(frames)):
        frame = frames[index]
        if frames_are_similar(anchor.frame_data, frame.frame_data, resolved, metric):
            if skip_count == 0:
                skip_start = frame.timestamp
                skip_first_index = index
            skip_count += 1
            skip_end = frame.timestamp
        else:
            flush_skip()
            items.append(ImageItem(frame=frame, index=index))
            anchor = frame

    flush_skip()
    return items


def should_analyze(
    newest_frame_data,
    last_analyzed_frame_data,
    now_monotonic: float,
    last_call_monotonic: float | None,
    threshold: float | None,
    min_interval_seconds: float,
    max_interval_seconds: float,
    metric: str = "mean",
) -> bool:
    """Decide whether to call the model this cycle.

    Call when the newest frame differs materially from the last analyzed frame,
    or when the heartbeat interval has elapsed. A ``min_interval_seconds`` floor
    caps the request rate. A non-positive resolved threshold (or a missing
    baseline) disables gating and always calls — reproducing pre-gating behaviour.
    """
    resolved = resolve_threshold(metric, threshold)

    if last_analyzed_frame_data is None:
        return True
    if resolved <= 0:
        return True

    elapsed = (
        now_monotonic - last_call_monotonic
        if last_call_monotonic is not None
        else math.inf
    )
    if min_interval_seconds > 0 and elapsed < min_interval_seconds:
        return False

    if not frames_are_similar(
        last_analyzed_frame_data, newest_frame_data, resolved, metric
    ):
        return True

    # No motion: only re-check on the heartbeat so nothing stays unseen forever.
    return max_interval_seconds > 0 and elapsed >= max_interval_seconds
