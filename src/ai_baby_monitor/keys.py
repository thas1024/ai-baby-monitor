"""Shared Redis stream key naming.

The producer (``stream_to_redis``), the watcher, and the Streamlit viewer all
address the same per-room streams, so the naming lives in one place to keep them
in sync.

Keys follow ``{prefix}:{room}:{kind}`` for the room-level ``logs`` stream, and
``{prefix}:{room}:{camera}:{kind}`` for a named camera's frame streams. A legacy
unnamed camera (``camera is None``) keeps unscoped frame keys
(``{prefix}:{room}:{kind}``) so existing deployments don't move data.
"""

KEY_PREFIX = "ai_baby_monitor"


def room_keys(room_name: str) -> dict[str, str]:
    """Return the room-level Redis stream keys.

    Args:
        room_name: The room's ``name`` from its YAML config.

    Returns:
        Mapping with the room-level ``logs`` key (and, for a legacy single camera,
        the ``realtime``/``subsampled`` keys too).
    """
    base = f"{KEY_PREFIX}:{room_name}"
    return {
        "realtime": f"{base}:realtime",
        "subsampled": f"{base}:subsampled",
        "logs": f"{base}:logs",
    }


def camera_keys(room_name: str, camera: str | None) -> dict[str, str]:
    """Return a camera's ``realtime``/``subsampled`` frame stream keys.

    ``camera=None`` (legacy unnamed camera) keeps the unscoped keys.
    """
    if camera is None:
        base = f"{KEY_PREFIX}:{room_name}"
    else:
        base = f"{KEY_PREFIX}:{room_name}:{camera}"
    return {
        "realtime": f"{base}:realtime",
        "subsampled": f"{base}:subsampled",
    }
