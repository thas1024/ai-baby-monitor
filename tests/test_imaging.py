import cv2
import numpy as np

from ai_baby_monitor.imaging import encode_thumbnail


def _jpeg(width=640, height=360, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = (rng.random((height, width, 3)) * 255).astype("uint8")
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer


def test_thumbnail_decodes_and_is_downscaled():
    original = _jpeg(width=640)
    thumb = encode_thumbnail(original, max_width=480)
    assert thumb is not None
    decoded = cv2.imdecode(np.frombuffer(thumb, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 480
    # Smaller than the source at 640px.
    assert len(thumb) < len(original)


def test_small_image_is_not_upscaled():
    small = _jpeg(width=320)
    thumb = encode_thumbnail(small, max_width=480)
    decoded = cv2.imdecode(np.frombuffer(thumb, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 320


def test_invalid_input_returns_none():
    assert encode_thumbnail(np.array([1, 2, 3], dtype=np.uint8)) is None
