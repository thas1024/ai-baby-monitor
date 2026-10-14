import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_CAMERA_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# Keys allowed on a single camera entry (new ``cameras:`` form). They may also
# appear in a legacy single ``camera:`` block.
_CAMERA_KEYS = (
    "subsampled_stream_maxlen",
    "frame_width",
    "frame_height",
    "subsample_rate",
    "motion_metric",
    "motion_threshold",
    "night_saturation_threshold",
    "num_frames_to_process",
)
# Room-level keys. In the new form these live at the top level; in the legacy
# single-``camera:`` form they were tucked inside the camera block.
_ROOM_KEYS = ("min_interval_seconds", "max_interval_seconds", "num_frames_to_process")


def _expand_env(value: str) -> str:
    """Expand ``${VAR}`` references in a config value from the environment.

    Lets secrets (e.g. a camera RTSP URL with credentials) live in ``.env``
    instead of the tracked YAML. Raises if a referenced variable is unset.
    """

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ValueError(f"Environment variable '{name}' referenced in config is not set")
        return os.environ[name]

    return _ENV_VAR_PATTERN.sub(replace, value)


@dataclass(slots=True, kw_only=True)
class CameraConfig:
    """One camera in a room. ``name`` labels this view in the prompt and scopes
    its Redis keys; ``None`` means a legacy unnamed single camera (unscoped keys).
    """

    name: str | None
    uri: str
    subsampled_stream_maxlen: int = 128
    frame_width: int = 640
    frame_height: int = 360
    subsample_rate: int = 4
    motion_metric: str = "mean"
    motion_threshold: float | None = None
    night_saturation_threshold: float | None = None
    # Per-camera override; None falls back to the room-level default.
    num_frames_to_process: int | None = None

    def __str__(self):
        return self.name or "default"


@dataclass(slots=True, kw_only=True)
class RoomConfig:
    name: str
    cameras: list[CameraConfig]
    instructions: list[str] = field(default_factory=list)
    min_interval_seconds: float = 10.0
    max_interval_seconds: float = 30.0
    num_frames_to_process: int = 16

    def __str__(self):
        return self.name

    def num_frames_for(self, camera: CameraConfig) -> int:
        """Frames to process for a camera (its override, else the room default)."""
        return camera.num_frames_to_process or self.num_frames_to_process

    # --- Backward-compatible delegating properties (first camera) ---
    # Kept so existing consumers (e.g. the Streamlit sidebar) keep working while
    # the codebase migrates to the ``cameras`` list.

    @property
    def camera_uri(self) -> str:
        return self.cameras[0].uri

    @property
    def frame_width(self) -> int:
        return self.cameras[0].frame_width

    @property
    def frame_height(self) -> int:
        return self.cameras[0].frame_height

    @property
    def subsample_rate(self) -> int:
        return self.cameras[0].subsample_rate

    @property
    def subsampled_stream_maxlen(self) -> int:
        return self.cameras[0].subsampled_stream_maxlen

    @property
    def motion_metric(self) -> str:
        return self.cameras[0].motion_metric

    @property
    def motion_threshold(self) -> float | None:
        return self.cameras[0].motion_threshold

    @property
    def night_saturation_threshold(self) -> float | None:
        return self.cameras[0].night_saturation_threshold


def _build_camera(raw: dict, *, allow_name: bool) -> CameraConfig:
    """Build a ``CameraConfig`` from a YAML mapping, expanding ${VAR} in the uri."""
    if "uri" not in raw:
        raise ValueError("Camera entry is missing required field: uri")

    name = raw.get("name")
    if allow_name:
        if name is not None and not _CAMERA_NAME_PATTERN.match(str(name)):
            raise ValueError(
                f"Invalid camera name {name!r}: use only letters, digits, '_' or '-'"
            )
    else:
        # Legacy single-camera block has no name.
        name = None

    knobs = {k: raw[k] for k in _CAMERA_KEYS if k in raw}
    return CameraConfig(name=name, uri=_expand_env(str(raw["uri"])), **knobs)


def _validate_cameras(cameras: list[CameraConfig]) -> None:
    if not cameras:
        raise ValueError("At least one camera is required")

    names = [c.name for c in cameras if c.name is not None]
    if len(names) != len(set(names)):
        raise ValueError(f"Camera names must be unique within a room, got {names}")
    if any(c.name is None for c in cameras) and len(cameras) > 1:
        raise ValueError("Unnamed cameras are only allowed as a single legacy camera")


def load_room_config_file(config_path: str | Path) -> RoomConfig:
    """Load a single room configuration from a specified YAML file.

    Accepts either a legacy single ``camera:`` mapping (synthesized into one
    unnamed camera) or a new ``cameras:`` list (one entry per camera/angle).
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        data = yaml.safe_load(config_path.read_text())

        if "camera" in data and "cameras" in data:
            raise ValueError("Specify either 'camera' or 'cameras' in the config, not both")

        room_kwargs = {k: data[k] for k in _ROOM_KEYS if k in data}

        if "cameras" in data:
            cameras = [_build_camera(c, allow_name=True) for c in data["cameras"]]
        elif "camera" in data:
            legacy = data["camera"]
            cameras = [_build_camera(legacy, allow_name=False)]
            # Legacy kept room-level knobs inside the camera block.
            room_kwargs.update({k: legacy[k] for k in _ROOM_KEYS if k in legacy})
        else:
            raise ValueError("Config is missing required field: 'camera' or 'cameras'")

        _validate_cameras(cameras)

        return RoomConfig(
            name=data["name"],
            cameras=cameras,
            instructions=data.get("instructions", []),
            **room_kwargs,
        )
    except KeyError as e:
        raise ValueError(f"Missing required field in config: {e}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")


def load_multiple_room_configs(config_files: list[str | Path]) -> dict[str, RoomConfig]:
    """Load multiple room configurations from a list of YAML files."""
    configs = {}
    for config_file in config_files:
        config = load_room_config_file(config_file)
        configs[config.name] = config
    return configs
