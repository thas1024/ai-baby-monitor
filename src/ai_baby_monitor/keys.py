"""Shared Redis stream key naming.

The producer (``stream_to_redis``), the watcher, and the Streamlit viewer all
address the same per-room streams, so the naming lives in one place to keep them
in sync. Keys follow ``{prefix}:{room}:{kind}``.
"""

KEY_PREFIX = "ai_baby_monitor"


def room_keys(room_name: str) -> dict[str, str]:
    """Return the Redis stream keys for a room.

    Args:
        room_name: The room's ``name`` from its YAML config.

    Returns:
        Mapping with ``realtime``, ``subsampled`` and ``logs`` keys.
    """
    base = f"{KEY_PREFIX}:{room_name}"
    return {
        "realtime": f"{base}:realtime",
        "subsampled": f"{base}:subsampled",
        "logs": f"{base}:logs",
    }
