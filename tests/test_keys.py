from ai_baby_monitor.keys import KEY_PREFIX, room_keys


def test_room_keys_prefixed():
    keys = room_keys("living_room")
    assert keys == {
        "realtime": "ai_baby_monitor:living_room:realtime",
        "subsampled": "ai_baby_monitor:living_room:subsampled",
        "logs": "ai_baby_monitor:living_room:logs",
    }
    assert KEY_PREFIX == "ai_baby_monitor"


def test_room_keys_distinct_per_room():
    assert room_keys("a")["logs"] != room_keys("b")["logs"]
