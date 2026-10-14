"""Small image helpers for storing representative frames in the logs stream."""

import cv2
import numpy as np

# Thumbnails keep the logs stream small (a full 640x360 JPEG can be ~60-90 KB;
# at a few logs per minute over the retention window that adds up).
_DEFAULT_MAX_WIDTH = 480
_DEFAULT_QUALITY = 80


def encode_thumbnail(
    frame_data,
    max_width: int = _DEFAULT_MAX_WIDTH,
    quality: int = _DEFAULT_QUALITY,
) -> bytes | None:
    """Downscale a JPEG frame and re-encode it as a small JPEG, or None on failure.

    Grey world only for storage/inspection — the analyzer never sees this. Returns
    None (never raises) when the input cannot be decoded, so a bad frame just means
    the log entry has no photo.
    """
    array = np.frombuffer(bytes(frame_data), dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        return None

    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        image = cv2.resize(image, (max_width, max(1, round(height * scale))))

    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return buffer.tobytes()
