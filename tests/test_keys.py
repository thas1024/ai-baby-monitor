from ai_baby_monitor.keys import KEY_PREFIX, camera_keys, room_keys


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



def test_camera_keys_scoped_when_named():
    keys = camera_keys("living_room", "closeup")
    assert keys["realtime"] == "ai_baby_monitor:living_room:closeup:realtime"
    assert keys["subsampled"] == "ai_baby_monitor:living_room:closeup:subsampled"
    assert keys["gate"] == "ai_baby_monitor:living_room:closeup:gate"
    assert keys["events"] == "ai_baby_monitor:living_room:closeup:events"
    assert keys["ptz"] == "ai_baby_monitor:living_room:closeup:ptz"


def test_camera_keys_unscoped_when_unnamed():
    keys = camera_keys("living_room", None)
    assert keys["realtime"] == "ai_baby_monitor:living_room:realtime"
    assert keys["subsampled"] == "ai_baby_monitor:living_room:subsampled"
    assert keys["gate"] == "ai_baby_monitor:living_room:gate"
