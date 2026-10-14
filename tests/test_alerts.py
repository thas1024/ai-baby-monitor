from ai_baby_monitor.alerts import should_emit_alert


def test_no_history_does_not_emit():
    assert should_emit_alert([], 300.0) is False
    assert should_emit_alert([(True, 0.0)], 300.0) is False


def test_single_flip_does_not_emit():
    # A lone True after a False must not trigger a beep.
    assert should_emit_alert([(False, 0.0), (True, 5.0)], 300.0) is False


def test_two_consecutive_alerts_emit():
    assert should_emit_alert([(False, 0.0), (True, 5.0), (True, 10.0)], 300.0) is True


def test_pair_too_far_apart_does_not_emit():
    assert should_emit_alert([(False, 0.0), (True, 5.0), (True, 1000.0)], 300.0) is False


def test_persistent_alert_only_emits_on_leading_edge():
    verdicts = [(True, 0.0), (True, 10.0), (True, 20.0)]
    assert should_emit_alert(verdicts, 300.0) is False  # already confirmed earlier


def test_re_alert_after_clear_emits_again():
    verdicts = [(True, 0.0), (True, 10.0), (False, 20.0), (True, 30.0), (True, 40.0)]
    assert should_emit_alert(verdicts, 300.0) is True
