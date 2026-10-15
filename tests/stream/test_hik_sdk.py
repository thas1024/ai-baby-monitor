from unittest.mock import MagicMock, patch

from ai_baby_monitor.stream import hik_sdk


def test_parse_rtsp_uri():
    p = hik_sdk.parse_rtsp_uri("rtsp://admin:secret@10.0.0.9:554/Streaming/Channels/101")
    assert p == {"user": "admin", "password": "secret", "host": "10.0.0.9", "port": 554}
    assert hik_sdk.parse_rtsp_uri("rtsp://admin:pw@10.0.0.5/x")["port"] == 554
    assert hik_sdk.parse_rtsp_uri("0") is None
    assert hik_sdk.parse_rtsp_uri("") is None


def test_hik_available_false_when_import_fails():
    with patch.object(hik_sdk, "_import_sdk", return_value=(None, "No module")):
        assert hik_sdk.hik_available() is False


def test_classify_event():
    assert hik_sdk._classify_event(0x4000) == "alarm"
    assert hik_sdk._classify_event(0x1100) == "alarm"
    assert hik_sdk._classify_event(0x1102) == "smart"
    assert hik_sdk._classify_event(0x6009) == "isapi"
    assert hik_sdk._classify_event(0x9999) is None


# --- CameraStream source selection ---

def test_source_rtsp_forced_skips_sdk():
    from ai_baby_monitor.stream.camera_stream import CameraStream

    with patch("ai_baby_monitor.stream.camera_stream.cv2.VideoCapture") as vc:
        vc.return_value.isOpened.return_value = True
        vc.return_value.get.return_value = 15.0
        stream = CameraStream(uri="rtsp://admin:pw@host/s", source="rtsp", decode="cpu")
    assert stream.source == "rtsp"
    assert stream._hik_frames is None


def test_source_falls_back_to_rtsp_when_sdk_import_fails():
    from ai_baby_monitor.stream.camera_stream import CameraStream

    with patch.object(hik_sdk, "_import_sdk", return_value=(None, "nope")), \
         patch("ai_baby_monitor.stream.camera_stream.cv2.VideoCapture") as vc:
        vc.return_value.isOpened.return_value = True
        vc.return_value.get.return_value = 15.0
        stream = CameraStream(uri="rtsp://admin:pw@host/s", source="hik-sdk", decode="cpu")
    assert stream.source == "rtsp"  # graceful fallback, no exception


def test_source_falls_back_when_login_fails():
    from ai_baby_monitor.stream.camera_stream import CameraStream

    fake_session = MagicMock()
    fake_session.login.return_value = False
    with patch("ai_baby_monitor.stream.hik_sdk.HikSession", return_value=fake_session), \
         patch.object(hik_sdk, "_import_sdk", return_value=(MagicMock(), None)), \
         patch("ai_baby_monitor.stream.camera_stream.cv2.VideoCapture") as vc:
        vc.return_value.isOpened.return_value = True
        vc.return_value.get.return_value = 15.0
        stream = CameraStream(uri="rtsp://admin:pw@host/s", source="hik-sdk", decode="cpu")
    assert stream.source == "rtsp"
    fake_session.login.assert_called_once()


def test_source_hik_sdk_used_when_available():
    from ai_baby_monitor.stream.camera_stream import CameraStream

    fake_session = MagicMock()
    fake_session.login.return_value = True
    fake_session.serial.return_value = "SN123"
    fake_frames = MagicMock()
    fake_frames.start.return_value = True
    with patch("ai_baby_monitor.stream.hik_sdk.HikSession", return_value=fake_session), \
         patch("ai_baby_monitor.stream.hik_sdk.HikFrameSource", return_value=fake_frames), \
         patch.object(hik_sdk, "_import_sdk", return_value=(MagicMock(), None)):
        stream = CameraStream(uri="rtsp://admin:pw@host/s", source="hik-sdk")
    assert stream.source == "hik-sdk"
    assert stream._hik_frames is fake_frames
    fake_frames.start.assert_called_once()


def test_source_falls_back_when_frame_start_fails():
    from ai_baby_monitor.stream.camera_stream import CameraStream

    fake_session = MagicMock()
    fake_session.login.return_value = True
    fake_frames = MagicMock()
    fake_frames.start.return_value = False
    with patch("ai_baby_monitor.stream.hik_sdk.HikSession", return_value=fake_session), \
         patch("ai_baby_monitor.stream.hik_sdk.HikFrameSource", return_value=fake_frames), \
         patch.object(hik_sdk, "_import_sdk", return_value=(MagicMock(), None)), \
         patch("ai_baby_monitor.stream.camera_stream.cv2.VideoCapture") as vc:
        vc.return_value.isOpened.return_value = True
        vc.return_value.get.return_value = 15.0
        stream = CameraStream(uri="rtsp://admin:pw@host/s", source="hik-sdk", decode="cpu")
    assert stream.source == "rtsp"
    fake_session.logout.assert_called_once()  # cleaned up on fallback
