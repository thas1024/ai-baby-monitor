from unittest.mock import MagicMock, patch

from ai_baby_monitor.ui import streamlit_components as sc


def _patch_st():
    patcher = patch.object(sc, "st")
    mock_st = patcher.start()
    mock_st.expander.return_value = MagicMock()
    return patcher, mock_st


def test_render_logs_handles_verdict_record():
    patcher, _ = _patch_st()
    try:
        sc.render_logs(
            [
                {
                    "timestamp": "t",
                    "should_alert": "1",
                    "awareness_level": "HIGH",
                    "reasoning": "baby climbing",
                }
            ]
        )
    finally:
        patcher.stop()


def test_render_logs_handles_error_record_without_verdict_fields():
    # The watcher writes {timestamp, error} records on failure; rendering them
    # must not raise KeyError on the missing verdict fields.
    patcher, _ = _patch_st()
    try:
        sc.render_logs([{"timestamp": "t", "error": "Request timed out."}])
    finally:
        patcher.stop()


def test_render_logs_is_timeline_newest_first():
    patcher, mock_st = _patch_st()
    try:
        sc.render_logs(
            [
                {"timestamp": "older", "should_alert": "0", "awareness_level": "LOW"},
                {"timestamp": "newer", "should_alert": "0", "awareness_level": "LOW"},
            ]
        )
    finally:
        patcher.stop()

    headers = [call.args[0] for call in mock_st.expander.call_args_list]
    assert "newer" in headers[0]
    assert "older" in headers[1]


def test_render_logs_expands_anomalies_collapses_normals():
    patcher, mock_st = _patch_st()
    try:
        sc.render_logs(
            [
                {"timestamp": "t1", "should_alert": "0", "awareness_level": "LOW"},
                {"timestamp": "t2", "should_alert": "1", "awareness_level": "HIGH"},
                {"timestamp": "t3", "should_alert": "0", "awareness_level": "MEDIUM"},
            ]
        )
    finally:
        patcher.stop()

    expanded_by_ts = {
        c.args[0]: c.kwargs.get("expanded") for c in mock_st.expander.call_args_list
    }

    def expanded_for(ts):
        return next(v for k, v in expanded_by_ts.items() if ts in k)

    assert expanded_for("t1") is False  # normal -> collapsed
    assert expanded_for("t2") is True  # alert -> expanded
    assert expanded_for("t3") is True  # MEDIUM -> expanded


def test_render_logs_shows_photo_when_present():
    patcher, mock_st = _patch_st()
    try:
        sc.render_logs(
            [
                {
                    "timestamp": "1",
                    "should_alert": "1",
                    "awareness_level": "HIGH",
                    "image": MagicMock(name="pil_image"),
                }
            ]
        )
    finally:
        patcher.stop()

    assert mock_st.image.called
