import datetime as dt
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from ai_baby_monitor.stream.camera_stream import CameraStream, Frame


@pytest.fixture
def mock_cv2_video_capture():
    """Fixture for a mock cv2.VideoCapture instance."""
    capture = MagicMock(spec=cv2.VideoCapture)
    capture.isOpened.return_value = True
    capture.get.return_value = 30.0  # Mock FPS
    capture.grab.return_value = True
    # Mock frame data: height, width, channels
    mock_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    capture.retrieve.return_value = (True, mock_frame)
    return capture


@pytest.fixture
def mock_cv2_video_writer():
    """Fixture for a mock cv2.VideoWriter instance."""
    writer = MagicMock(spec=cv2.VideoWriter)
    return writer


@pytest.fixture
@patch("ai_baby_monitor.stream.camera_stream.cv2.VideoCapture")
def camera_stream_no_save(mock_video_capture_constructor, mock_cv2_video_capture):
    """Fixture for CameraStream without saving the stream."""
    mock_video_capture_constructor.return_value = mock_cv2_video_capture
    stream = CameraStream(uri="rtsp://mock", decode="cpu")
    return stream, mock_cv2_video_capture


@pytest.fixture
@patch("ai_baby_monitor.stream.camera_stream.cv2.VideoWriter")
@patch("ai_baby_monitor.stream.camera_stream.cv2.VideoWriter_fourcc")
@patch("ai_baby_monitor.stream.camera_stream.cv2.VideoCapture")
def camera_stream_with_save(
    mock_video_capture_constructor,
    mock_fourcc_constructor_in_fixture,
    mock_video_writer_constructor_in_fixture,
    mock_cv2_video_capture,
    mock_cv2_video_writer,
    tmp_path,
):
    """Fixture for CameraStream with saving the stream."""
    mock_video_capture_constructor.return_value = mock_cv2_video_capture
    mock_fourcc_constructor_in_fixture.return_value = "MP4V_FROM_FIXTURE"
    mock_video_writer_constructor_in_fixture.return_value = mock_cv2_video_writer

    save_path = str(tmp_path / "test_video.mp4")
    stream = CameraStream(
        uri="rtsp://mock",
        save_stream_path=save_path,
        frame_shape=(320, 240),
        decode="cpu",
    )
    # Return the class mocks as well so the test can assert on them if needed, or just the instances.
    return (
        stream,
        mock_cv2_video_capture,
        mock_cv2_video_writer,
        save_path,
        mock_fourcc_constructor_in_fixture,
        mock_video_writer_constructor_in_fixture,
    )


def test_camera_stream_init_successful(camera_stream_no_save):
    """Test successful initialization of CameraStream."""
    stream, mock_capture = camera_stream_no_save
    assert stream.capture == mock_capture
    assert stream.stream_writer is None
    assert stream.frame_idx == 0
    mock_capture.isOpened.assert_called_once()


def test_camera_stream_init_with_save(camera_stream_with_save):
    """Test CameraStream initialization with stream saving enabled."""
    stream, _, mock_writer_instance, _, mock_fourcc_class, mock_writer_class = (
        camera_stream_with_save
    )
    assert stream.stream_writer is not None
    assert (
        stream.stream_writer == mock_writer_instance
    )  # Verifies it's the correct mock instance

    # Basic check that constructors were called (via the fixture's setup)
    mock_fourcc_class.assert_called_once()
    mock_writer_class.assert_called_once()


@patch(
    "ai_baby_monitor.stream.camera_stream.cv2.imencode",
    return_value=(True, np.array([255, 0, 255], dtype=np.uint8)),
)
@patch(
    "ai_baby_monitor.stream.camera_stream.dt",
    MagicMock(
        datetime=MagicMock(
            now=MagicMock(return_value=dt.datetime(2023, 1, 1, 10, 0, 0))
        )
    ),
)
def test_capture_new_frame_successful(mock_imencode, camera_stream_no_save):
    """Test successful frame capture and JPEG encoding."""
    stream, mock_capture = camera_stream_no_save
    mock_raw_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    mock_capture.retrieve.return_value = (True, mock_raw_frame)

    frame_obj = stream.capture_new_frame()

    assert isinstance(frame_obj, Frame)
    assert frame_obj.frame_idx == 0  # First frame
    assert np.array_equal(frame_obj.frame_data, np.array([255, 0, 255], dtype=np.uint8))
    assert frame_obj.timestamp == dt.datetime(2023, 1, 1, 10, 0, 0)
    assert stream.frame_idx == 1  # Incremented

    mock_capture.grab.assert_called_once()
    mock_capture.retrieve.assert_called_once()
    mock_imencode.assert_called_once()
    # Check that the raw frame was passed to imencode
    assert np.array_equal(mock_imencode.call_args[0][1], mock_raw_frame)


def test_close_no_writer(camera_stream_no_save):
    """Test close method when no stream writer is active."""
    stream, mock_capture = camera_stream_no_save
    stream.close()
    mock_capture.release.assert_called_once()


def test_close_with_writer(camera_stream_with_save):
    """Test close method when a stream writer is active."""
    # Unpack all 6 items from the fixture, even if not all are used in this test
    stream, mock_capture, mock_writer_instance, _, _, _ = camera_stream_with_save
    stream.close()
    mock_capture.release.assert_called_once()
    mock_writer_instance.release.assert_called_once()


# --- hardware decode selection ---

from ai_baby_monitor.stream.camera_stream import resolve_decode, _FfmpegDecoder


def test_resolve_decode_cpu_passthrough():
    assert resolve_decode("cpu") == "cpu"


def test_resolve_decode_explicit_hwacecl_passthrough():
    assert resolve_decode("qsv") == "qsv"
    assert resolve_decode("vaapi") == "vaapi"


def test_resolve_decode_unknown_falls_back_to_auto(monkeypatch):
    # unknown -> auto -> cpu when no render device
    monkeypatch.setattr(
        "ai_baby_monitor.stream.camera_stream.os.path.exists", lambda p: False
    )
    assert resolve_decode("bogus") == "cpu"


def test_resolve_decode_auto_no_device_is_cpu(monkeypatch):
    monkeypatch.setattr(
        "ai_baby_monitor.stream.camera_stream.os.path.exists", lambda p: False
    )
    assert resolve_decode("auto") == "cpu"


def test_resolve_decode_auto_prefers_vaapi(monkeypatch):
    monkeypatch.setattr(
        "ai_baby_monitor.stream.camera_stream.os.path.exists", lambda p: True
    )
    monkeypatch.setattr(
        "ai_baby_monitor.stream.camera_stream._ffmpeg_hwaccels",
        lambda bin: {"vaapi", "qsv"},
    )
    assert resolve_decode("auto") == "vaapi"


def test_resolve_decode_auto_qsv_when_no_vaapi(monkeypatch):
    monkeypatch.setattr(
        "ai_baby_monitor.stream.camera_stream.os.path.exists", lambda p: True
    )
    monkeypatch.setattr(
        "ai_baby_monitor.stream.camera_stream._ffmpeg_hwaccels", lambda bin: {"qsv"}
    )
    assert resolve_decode("auto") == "qsv"


def test_ffmpeg_cmd_qsv():
    with patch("ai_baby_monitor.stream.camera_stream.subprocess.Popen") as popen, \
         patch("ai_baby_monitor.stream.camera_stream.threading.Thread"):
        popen.return_value = MagicMock()
        dec = _FfmpegDecoder("rtsp://cam/s", "qsv", (640, 360))
        cmd = dec._build_cmd()
    assert "-hwaccel" in cmd and "qsv" in cmd
    assert "scale=640:360" in " ".join(cmd)
    assert cmd[-1] == "-"


def test_ffmpeg_cmd_vaapi_uses_device():
    with patch("ai_baby_monitor.stream.camera_stream.subprocess.Popen") as popen, \
         patch("ai_baby_monitor.stream.camera_stream.threading.Thread"):
        popen.return_value = MagicMock()
        dec = _FfmpegDecoder("rtsp://cam/s", "vaapi", (640, 360))
        cmd = dec._build_cmd()
    assert "-hwaccel_device" in cmd
    assert "/dev/dri/renderD128" in cmd


def test_ffmpeg_decoder_reads_frame():
    """A full frame of pipe bytes is reshaped into a BGR array."""
    frame_bytes = bytes(640 * 360 * 3)
    fake_proc = MagicMock()
    fake_proc.stdout.read.return_value = frame_bytes

    with patch("ai_baby_monitor.stream.camera_stream.subprocess.Popen", return_value=fake_proc), \
         patch("ai_baby_monitor.stream.camera_stream.threading.Thread"):
        dec = _FfmpegDecoder("rtsp://cam/s", "vaapi", (640, 360))

    dec._proc = fake_proc
    # one successful frame then EOF
    fake_proc.stdout.read.side_effect = [frame_bytes, b""]
    dec._read_frames()
    assert dec.latest() is not None
    assert dec.latest().shape == (360, 640, 3)


def test_ffmpeg_decoder_next_frame_blocks_for_new_frame():
    """next_frame returns each decoded frame once, then times out with no new one."""
    frame_bytes = bytes(640 * 360 * 3)
    fake_proc = MagicMock()
    with patch("ai_baby_monitor.stream.camera_stream.subprocess.Popen", return_value=fake_proc), \
         patch("ai_baby_monitor.stream.camera_stream.threading.Thread"):
        dec = _FfmpegDecoder("rtsp://cam/s", "vaapi", (640, 360))

    dec._proc = fake_proc
    fake_proc.stdout.read.side_effect = [frame_bytes, b""]
    dec._read_frames()  # decodes exactly 1 frame -> seq=1

    first = dec.next_frame(timeout=1.0)
    assert first is not None
    # No new frame since -> times out returning None (does NOT re-serve the same one)
    assert dec.next_frame(timeout=0.1) is None
