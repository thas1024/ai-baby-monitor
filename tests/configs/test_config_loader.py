from pathlib import Path

import pytest
import yaml

from ai_baby_monitor.config import RoomConfig, load_room_config_file


def test_load_room_config_file_with_optional_fields(tmp_path: Path):
    """Test loading a config file with optional fields specified."""
    sample = {
        "name": "custom_room",
        "camera": {
            "uri": "rtsp://example.com/stream",
            "frame_width": 1280,
            "frame_height": 720,
            "subsampled_stream_maxlen": 100,
            "subsample_rate": 8,
        },
        "instructions": ["custom instruction"],
    }

    config_path = tmp_path / "custom_room.yaml"
    config_path.write_text(yaml.safe_dump(sample))

    config = load_room_config_file(config_path)

    # Check required fields
    assert config.name == "custom_room"
    assert config.camera_uri == "rtsp://example.com/stream"
    assert config.instructions == ["custom instruction"]

    # Check optional fields are loaded correctly
    assert config.frame_width == 1280
    assert config.frame_height == 720
    assert config.subsampled_stream_maxlen == 100
    assert config.subsample_rate == 8

    assert isinstance(config, RoomConfig)


def test_load_room_config_file_with_partial_optional_fields(tmp_path: Path):
    """Test loading a config file with only some optional fields specified."""
    sample = {
        "name": "partial_room",
        "camera": {
            "uri": "2",
            # Only specify frame_width, leaving other optional fields as defaults
            "frame_width": 800,
        },
    }

    config_path = tmp_path / "partial_room.yaml"
    config_path.write_text(yaml.safe_dump(sample))

    config = load_room_config_file(config_path)

    # Check the specified optional field
    assert config.frame_width == 800

    # Check that other optional fields keep their defaults
    assert config.frame_height == 360  # default
    assert config.subsampled_stream_maxlen == 128  # default
    assert config.subsample_rate == 4  # default
    assert config.instructions == []  # default


def test_camera_uri_expands_env_var(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CAMERA_URI", "rtsp://u:p@host:554/s")
    sample = {"name": "r", "camera": {"uri": "${CAMERA_URI}"}}
    config_path = tmp_path / "r.yaml"
    config_path.write_text(yaml.safe_dump(sample))
    assert load_room_config_file(config_path).camera_uri == "rtsp://u:p@host:554/s"


def test_camera_uri_missing_env_var_raises(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CAMERA_URI", raising=False)
    sample = {"name": "r", "camera": {"uri": "${CAMERA_URI}"}}
    config_path = tmp_path / "r.yaml"
    config_path.write_text(yaml.safe_dump(sample))
    with pytest.raises(ValueError, match="CAMERA_URI"):
        load_room_config_file(config_path)


def test_motion_settings_defaults():
    config = RoomConfig(name="r", camera_uri="0")
    assert config.motion_metric == "mean"
    assert config.motion_threshold is None  # None -> metric default
    assert config.min_interval_seconds == 10.0
    assert config.max_interval_seconds == 30.0


def test_motion_settings_from_yaml(tmp_path: Path):
    sample = {
        "name": "r",
        "camera": {
            "uri": "0",
            "motion_metric": "grid_max",
            "motion_threshold": 5.5,
            "min_interval_seconds": 20,
            "max_interval_seconds": 120,
        },
    }
    config_path = tmp_path / "r.yaml"
    config_path.write_text(yaml.safe_dump(sample))
    config = load_room_config_file(config_path)
    assert config.motion_metric == "grid_max"
    assert config.motion_threshold == 5.5
    assert config.min_interval_seconds == 20
    assert config.max_interval_seconds == 120
