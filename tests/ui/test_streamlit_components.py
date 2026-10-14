from unittest.mock import MagicMock, patch

from ai_baby_monitor.ui import streamlit_components as sc


def _patch_st():
    patcher = patch.object(sc, "st")
    mock_st = patcher.start()
    mock_st.expander.return_value = MagicMock()
    # The timeline lays each entry out in a [node, body] column pair.
    mock_st.columns.return_value = [MagicMock(), MagicMock()]
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
                    "images": [("closeup", MagicMock(name="pil_image"))],
                }
            ]
        )
    finally:
        patcher.stop()

    assert mock_st.image.called


def test_render_logs_shows_multiple_camera_photos():
    patcher, mock_st = _patch_st()
    try:
        sc.render_logs(
            [
                {
                    "timestamp": "1",
                    "should_alert": "1",
                    "awareness_level": "HIGH",
                    "images": [
                        ("closeup", MagicMock(name="img1")),
                        ("wide", MagicMock(name="img2")),
                    ],
                }
            ]
        )
    finally:
        patcher.stop()

    assert mock_st.image.call_count == 2


def test_render_logs_handles_info_record():
    patcher, mock_st = _patch_st()
    try:
        sc.render_logs([{"timestamp": "t", "info": "video restored"}])
    finally:
        patcher.stop()
    assert mock_st.expander.called


def test_localize_uses_app_tz(monkeypatch):
    import datetime as dt

    monkeypatch.setenv("APP_TZ", "Asia/Shanghai")
    epoch = dt.datetime(2026, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc).timestamp()
    out = sc._localize(epoch)
    assert out.hour == 8  # UTC+8
    assert out.utcoffset() == dt.timedelta(hours=8)


def test_localize_bad_tz_falls_back(monkeypatch):
    monkeypatch.setenv("APP_TZ", "Not/AZone")
    assert sc._display_tz() is None


# --- timeline node / colors ---

def test_medium_level_uses_orange_not_yellow():
    # Streamlit has no ":yellow[...]" — must be a supported color name.
    assert ":orange[" in sc._colored_level("MEDIUM")
    assert ":yellow[" not in sc._colored_level("MEDIUM")


def test_node_html_colors_per_level():
    assert sc._NODE_COLORS["LOW"] in sc._timeline_node(sc._NODE_COLORS["LOW"], False)
    med = sc._timeline_node(sc._NODE_COLORS["MEDIUM"], False)
    assert "#f59e0b" in med
    strong = sc._timeline_node(sc._NODE_COLORS["HIGH"], True)
    assert "tl-strong" in strong


def test_render_logs_draws_a_node_per_entry():
    patcher, mock_st = _patch_st()
    try:
        sc.render_logs(
            [
                {"timestamp": "a", "should_alert": "0", "awareness_level": "LOW"},
                {"timestamp": "b", "should_alert": "1", "awareness_level": "HIGH"},
            ]
        )
    finally:
        patcher.stop()
    # one node markdown per entry (the CSS block also mentions the class)
    node_calls = [
        c for c in mock_st.markdown.call_args_list if 'class="tl-node"' in str(c.args[0])
    ]
    assert len(node_calls) == 2


# --- filtering ---

def test_log_level_and_is_alert():
    assert sc.log_level({"should_alert": "1", "awareness_level": "HIGH"}) == "HIGH"
    assert sc.log_level({"timestamp": "t", "error": "x"}) is None
    assert sc.is_alert({"should_alert": "1"}) is True
    assert sc.is_alert({"should_alert": "0"}) is False


def test_filter_logs_by_level():
    logs = [
        {"should_alert": "0", "awareness_level": "LOW"},
        {"should_alert": "1", "awareness_level": "HIGH"},
        {"should_alert": "0", "awareness_level": "MEDIUM"},
    ]
    assert len(sc.filter_logs(logs, ["LOW", "MEDIUM", "HIGH"])) == 3
    assert sc.filter_logs(logs, ["HIGH"]) == [logs[1]]
    assert len(sc.filter_logs(logs, [])) == 3  # empty = all


def test_filter_logs_alerts_only_excludes_health_records():
    logs = [
        {"should_alert": "1", "awareness_level": "HIGH"},
        {"should_alert": "0", "awareness_level": "LOW"},
        {"timestamp": "t", "error": "no live video"},
    ]
    out = sc.filter_logs(logs, [], alerts_only=True)
    assert out == [logs[0]]


def test_filter_logs_level_keeps_health_records_when_not_alerts_only():
    logs = [
        {"timestamp": "t", "error": "no live video"},
        {"should_alert": "0", "awareness_level": "LOW"},
    ]
    out = sc.filter_logs(logs, ["HIGH"])
    # health record has no level -> dropped by a level filter; LOW not selected
    assert out == []
