"""Debouncing for alert actions.

The vision model can jitter a single ``should_alert`` verdict. Rather than
beeping on every flip, an alert action is only emitted once a violation has been
confirmed by two consecutive positive verdicts close together in time — and only
on the leading edge, so a violation that persists does not beep forever.

Only the *action* is debounced. Every verdict is still written to the logs
stream; see ``run_watcher.py``.
"""

Verdict = tuple[bool, float]  # (should_alert, monotonic timestamp)


def is_alert_worthy(
    should_alert: bool, night_mode: bool, awareness_level: str, high_level: str = "HIGH"
) -> bool:
    """Whether a verdict is eligible to raise an alert action at all.

    In night-vision mode the monitor is deliberately less sensitive (grayscale/IR
    frames are noisier), so only a HIGH-confidence violation qualifies. In color
    mode any positive verdict qualifies. This gates the *action*; the verdict is
    logged regardless.
    """
    if not should_alert:
        return False
    if night_mode and awareness_level != high_level:
        return False
    return True


def should_emit_alert(verdicts: list[Verdict], max_gap_seconds: float) -> bool:
    """Whether the just-recorded verdict confirms a *new* alert.

    ``verdicts`` is chronological, ending with the verdict just produced. Returns
    True only when the last two verdicts both alert and are within
    ``max_gap_seconds`` of each other, and the pair before that was not itself a
    confirmed alert (leading edge).
    """
    if len(verdicts) < 2:
        return False

    (prev_alert, prev_ts), (last_alert, last_ts) = verdicts[-2], verdicts[-1]
    if not (prev_alert and last_alert):
        return False
    if last_ts - prev_ts > max_gap_seconds:
        return False

    # Leading edge only: if the previous verdict was already part of a confirmed
    # alert pair, this is a continuation, not a new confirmation.
    if len(verdicts) >= 3:
        before_alert, before_ts = verdicts[-3]
        if before_alert and prev_ts - before_ts <= max_gap_seconds:
            return False

    return True
