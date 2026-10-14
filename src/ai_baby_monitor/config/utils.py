import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


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
class RoomConfig:
    name: str
    camera_uri: str
    instructions: list[str] = field(default_factory=list)
    # Retention must cover max_interval_seconds at the subsampled frame rate
    # (~2.9 fps here → 128 frames ≈ 44s), or catch-up cannot see the whole gap.
    subsampled_stream_maxlen: int = 128
    frame_width: int = 640
    frame_height: int = 360
    subsample_rate: int = 4
    num_frames_to_process: int = 16
    # Motion gating / request throttling (see docs and motion.py).
    # motion_metric selects how frame difference is scored; motion_threshold=None
    # uses that metric's default, and an explicit 0 disables gating entirely.
    motion_metric: str = "mean"
    motion_threshold: float | None = None
    min_interval_seconds: float = 10.0
    max_interval_seconds: float = 30.0

    def __str__(self):
        return self.name


def load_room_config_file(config_path: str | Path) -> RoomConfig:
    """Load a single room configuration from a specified YAML file"""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        data = yaml.safe_load(config_path.read_text())
        
        config_kwargs = {
            # Required fields
            "name": data["name"],
            "camera_uri": _expand_env(str(data["camera"]["uri"])),
            "instructions": data.get("instructions", []),
            
            # Optional fields (only include if specified in YAML)
            **{k: data["camera"][k] for k in
               ["subsampled_stream_maxlen", "frame_width", "frame_height", "subsample_rate", "num_frames_to_process", "motion_metric", "motion_threshold", "min_interval_seconds", "max_interval_seconds"]
               if k in data["camera"]}
        }
        
        return RoomConfig(**config_kwargs)
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
