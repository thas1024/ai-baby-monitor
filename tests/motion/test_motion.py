import datetime as dt
import math

import cv2
import numpy as np

from ai_baby_monitor.motion import (
    ImageItem,
    SkippedItem,
    build_timeline,
    frame_diff_score,
    frames_are_similar,
    resolve_threshold,
    should_analyze,
)
from ai_baby_monitor.stream import Frame


def _jpeg(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = (rng.random((90, 160, 3)) * 255).astype("uint8")
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer


def _frame(jpeg: np.ndarray, second: int = 0) -> Frame:
    return Frame(
        frame_data=jpeg,
        timestamp=dt.datetime(2026, 1, 1, 12, 0, second),
        frame_idx=second,
    )


# --- frame_diff_score / frames_are_similar ---

def test_identical_frames_score_zero():
    jpeg = _jpeg(1)
    assert frame_diff_score(jpeg, jpeg.copy()) == 0.0
    assert frames_are_similar(jpeg, jpeg.copy(), threshold=3.0) is True


def test_different_frames_score_high():
    score = frame_diff_score(_jpeg(1), _jpeg(2))
    assert score > 3.0
    assert frames_are_similar(_jpeg(1), _jpeg(2), threshold=3.0) is False


def test_undecodable_frame_is_treated_as_changed():
    bad = np.array([1, 2, 3], dtype=np.uint8)
    assert frame_diff_score(bad, _jpeg(1)) == math.inf


def test_non_positive_threshold_disables_comparison():
    jpeg = _jpeg(1)
    assert frames_are_similar(jpeg, jpeg.copy(), threshold=0) is False


# --- build_timeline ---

def test_first_frame_is_always_an_image():
    items = build_timeline([_frame(_jpeg(1))], threshold=3.0)
    assert len(items) == 1
    assert isinstance(items[0], ImageItem)
    assert items[0].index == 0


def test_unchanged_run_is_skipped():
    a = _jpeg(1)
    frames = [_frame(a, 0), _frame(a.copy(), 1), _frame(a.copy(), 2)]
    items = build_timeline(frames, threshold=3.0)
    assert isinstance(items[0], ImageItem)
    assert isinstance(items[1], SkippedItem)
    assert items[1].count == 2
    assert items[1].first_index == 1
    assert items[1].last_index == 2


def test_motion_flushes_skip_and_starts_new_anchor():
    a, b = _jpeg(1), _jpeg(2)
    frames = [_frame(a, 0), _frame(a.copy(), 1), _frame(b, 2), _frame(b.copy(), 3)]
    items = build_timeline(frames, threshold=3.0)
    kinds = [type(i) for i in items]
    assert kinds == [ImageItem, SkippedItem, ImageItem, SkippedItem]
    assert items[2].index == 2  # the changed frame
    assert items[1].count == 1
    assert items[3].count == 1


def test_trailing_skip_run_is_flushed():
    a, b = _jpeg(1), _jpeg(2)
    frames = [_frame(a, 0), _frame(b, 1), _frame(b.copy(), 2)]
    items = build_timeline(frames, threshold=3.0)
    assert [type(i) for i in items] == [ImageItem, ImageItem, SkippedItem]
    assert items[-1].count == 1


def test_empty_window_yields_no_items():
    assert build_timeline([], threshold=3.0) == []


def test_threshold_zero_sends_every_frame():
    a = _jpeg(1)
    frames = [_frame(a, 0), _frame(a.copy(), 1), _frame(a.copy(), 2)]
    items = build_timeline(frames, threshold=0)
    assert all(isinstance(i, ImageItem) for i in items)
    assert len(items) == 3


# --- should_analyze ---

def test_should_analyze_first_time_always_true():
    assert should_analyze(_jpeg(1), None, 0.0, None, 3.0, 10.0, 30.0) is True


def test_should_analyze_disabled_when_threshold_non_positive():
    jpeg = _jpeg(1)
    assert should_analyze(jpeg, jpeg.copy(), 0.0, 0.0, 0.0, 10.0, 30.0) is True


def test_should_analyze_skips_within_min_interval():
    jpeg = _jpeg(1)
    # identical frames, only 1s since last call, min interval 10s -> skip
    assert should_analyze(jpeg, jpeg.copy(), 1.0, 0.0, 3.0, 10.0, 30.0) is False


def test_should_analyze_calls_on_motion():
    assert should_analyze(_jpeg(2), _jpeg(1), 100.0, 0.0, 3.0, 10.0, 30.0) is True


def test_should_analyze_heartbeat_calls_without_motion():
    jpeg = _jpeg(1)
    # no motion, 40s elapsed (>= max 30s) -> call
    assert should_analyze(jpeg, jpeg.copy(), 40.0, 0.0, 3.0, 10.0, 30.0) is True


def test_should_analyze_no_motion_before_heartbeat_skips():
    jpeg = _jpeg(1)
    # no motion, 15s elapsed (>= min 10, < max 30) -> skip
    assert should_analyze(jpeg, jpeg.copy(), 15.0, 0.0, 3.0, 10.0, 30.0) is False


# --- metrics (F1) ---

def _solid(value: int, shape=(90, 160, 3)) -> np.ndarray:
    ok, buffer = cv2.imencode(".jpg", np.full(shape, value, np.uint8))
    assert ok
    return buffer


def _with_block(value: int, block_value: int, y: int = 40, x: int = 80, size: int = 20) -> np.ndarray:
    image = np.full((90, 160, 3), value, np.uint8)
    image[y:y + size, x:x + size] = block_value
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer


def test_grid_max_detects_localized_change_mean_misses():
    base = _solid(100)
    local = _with_block(100, 220, size=12)
    mean_score = frame_diff_score(base, local, metric="mean")
    grid_score = frame_diff_score(base, local, metric="grid_max")
    # A small localized change barely moves the whole-frame mean, but dominates one cell.
    assert mean_score < 3.0
    assert grid_score > 10.0


def test_changed_fraction_ignores_uniform_noise():
    rng = np.random.default_rng(0)
    base = np.full((90, 160, 3), 100, np.uint8)
    noisy = np.clip(base.astype(int) + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
    ok, noisy_jpeg = cv2.imencode(".jpg", noisy)
    assert ok
    ok, base_jpeg = cv2.imencode(".jpg", base)
    assert ok
    # Noise below the floor -> tiny changed fraction -> "similar" at 0.01.
    assert frame_diff_score(base_jpeg, noisy_jpeg, metric="changed_fraction") < 0.01


def test_changed_fraction_detects_block():
    base = _solid(100)
    local = _with_block(100, 220)
    assert frame_diff_score(base, local, metric="changed_fraction") > 0.01


def test_resolve_threshold_uses_metric_default():
    assert resolve_threshold("mean", None) == 3.0
    assert resolve_threshold("grid_max", None) == 6.0
    assert resolve_threshold("changed_fraction", None) == 0.01
    assert resolve_threshold("mean", 9.0) == 9.0


def test_unknown_metric_falls_back_to_mean():
    base = _solid(100)
    other = _solid(200)
    assert frame_diff_score(base, other, metric="bogus") == frame_diff_score(base, other, metric="mean")


# --- night-vision detection + day/night transitions ---

def _gray_jpeg(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = (rng.random((90, 160)) * 255).astype("uint8")
    ok, buffer = cv2.imencode(".jpg", gray)
    assert ok
    return buffer


def test_night_frame_detected_for_grayscale():
    from ai_baby_monitor.motion import is_night_frame

    assert is_night_frame(_gray_jpeg(1)) is True


def test_color_frame_is_not_night():
    from ai_baby_monitor.motion import is_night_frame

    assert is_night_frame(_jpeg(1)) is False


def test_undecodable_frame_is_not_night():
    from ai_baby_monitor.motion import is_night_frame

    assert is_night_frame(np.array([1, 2, 3], dtype=np.uint8)) is False


def test_image_item_carries_night_flag():
    from ai_baby_monitor.motion import ImageItem

    items = build_timeline([_frame(_gray_jpeg(1))], threshold=3.0)
    assert isinstance(items[0], ImageItem)
    assert items[0].night is True


def test_day_to_night_transition_resets_anchor_and_annotates():
    from ai_baby_monitor.motion import ModeTransitionItem

    color = _jpeg(1)
    gray = _gray_jpeg(9)
    frames = [
        _frame(color, 0),
        _frame(color.copy(), 1),
        _frame(gray, 2),
        _frame(gray.copy(), 3),
    ]
    items = build_timeline(frames, threshold=3.0)

    kinds = [type(i) for i in items]
    assert kinds == [ImageItem, SkippedItem, ModeTransitionItem, ImageItem, SkippedItem]
    transition = items[2]
    assert transition.to_night is True
    # The post-switch grayscale frames share the new anchor -> only one image, no burst.
    assert items[3].night is True
    assert items[4].count == 1


# --- blank / no-signal frames ---

def _flat_gray(value=128) -> np.ndarray:
    ok, buffer = cv2.imencode(".jpg", np.full((90, 160, 3), value, np.uint8))
    assert ok
    return buffer


def test_blank_frame_detected():
    from ai_baby_monitor.motion import is_blank_frame

    assert is_blank_frame(_flat_gray(128)) is True
    assert is_blank_frame(_jpeg(1)) is False  # real content


def test_blank_frame_is_not_night():
    # A flat gray frame from a dead stream is grayscale (sat 0) but has no
    # content, so it must NOT be classified as night-vision.
    from ai_baby_monitor.motion import is_night_frame

    assert is_night_frame(_flat_gray(128)) is False


def test_real_grayscale_is_still_night():
    from ai_baby_monitor.motion import is_night_frame

    assert is_night_frame(_gray_jpeg(1)) is True
