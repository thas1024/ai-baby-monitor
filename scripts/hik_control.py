#!/usr/bin/env python3
"""Direct HCNetSDK control CLI for a Hikvision/EZVIZ camera.

Talks to the camera's local device session (port 8000): login, PTZ, snapshot,
device events, and an optional single-frame grab. Use it to verify the SDK works
and to drive the camera by hand; the app itself uses ``stream/hik_sdk.py``.

Credentials come from ``--user``/``--password`` or from ``CAMERA_URI`` in the
environment/.env. Run from the repo root:

    uv run python scripts/hik_control.py --host <CAMERA_IP> --password <DEVICE_PASSWORD> status
    uv run python scripts/hik_control.py --host <CAMERA_IP> --password <DEVICE_PASSWORD> ptz left
    uv run python scripts/hik_control.py --host <CAMERA_IP> --password <DEVICE_PASSWORD> snapshot --out /tmp/s.jpg
    uv run python scripts/hik_control.py --host <CAMERA_IP> --password <DEVICE_PASSWORD> events --seconds 30
"""

import argparse
import os
import re
import sys
from pathlib import Path

import structlog
from dotenv import load_dotenv

logger = structlog.get_logger()
load_dotenv()


def _creds_from_env() -> dict:
    uri = os.getenv("CAMERA_URI", "")
    m = re.match(r"rtsp://([^:]+):([^@]+)@([^:/]+)", uri)
    if not m:
        return {}
    return {"user": m.group(1), "password": m.group(2), "host": m.group(3)}


def main() -> int:
    env = _creds_from_env()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=env.get("host") or os.getenv("CAMERA_SDK_HOST"))
    p.add_argument("--port", type=int, default=int(os.getenv("CAMERA_SDK_PORT", "8000")))
    p.add_argument("--channel", type=int, default=1)
    p.add_argument("--user", default=os.getenv("CAMERA_SDK_USER", env.get("user", "admin")))
    p.add_argument("--password", default=os.getenv("CAMERA_SDK_PASSWORD", env.get("password")))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="login and print device info / PTZ support")
    pt = sub.add_parser("ptz", help="nudge the camera")
    pt.add_argument("direction", choices=sorted(
        ["left", "right", "up", "down", "upleft", "upright", "downleft", "downright",
         "zoom_in", "zoom_out", "focus_near", "focus_far"]))
    pt.add_argument("--speed", type=int, default=4)
    pt.add_argument("--seconds", type=float, default=0.6)
    snap = sub.add_parser("snapshot", help="capture a JPEG")
    snap.add_argument("--out", default="/tmp/hik_snapshot.jpg")
    ev = sub.add_parser("events", help="listen for device events")
    ev.add_argument("--seconds", type=float, default=30.0)

    args = p.parse_args()
    if not args.host or not args.password:
        print("Need --host and --password (or CAMERA_URI / CAMERA_SDK_* in .env)", file=sys.stderr)
        return 2

    from ai_baby_monitor.stream import hik_sdk

    if not hik_sdk.hik_available():
        print("HCNetSDK is not installed/importable here.", file=sys.stderr)
        return 3

    session = hik_sdk.HikSession(
        host=args.host, user=args.user, password=args.password,
        port=args.port, channel=args.channel,
    )
    if not session.login():
        print(f"Login failed (err={session.last_error()}). "
              "err=1 password wrong, err=153 locked, other=network.", file=sys.stderr)
        return 1
    print(f"OK: serial={session.serial()}")

    try:
        if args.cmd == "status":
            print("PTZ supported:", session.ptz_supported(), "| position:", session.ptz_position())
        elif args.cmd == "snapshot":
            data = session.snapshot()
            if data:
                Path(args.out).write_bytes(data)
                print(f"snapshot -> {args.out} ({len(data)} bytes)")
            else:
                print("snapshot failed")
        elif args.cmd == "ptz":
            ok = session.ptz_move(args.direction, args.speed, args.seconds)
            print("ptz", args.direction, "->", ok)
        elif args.cmd == "events":
            import time

            events = hik_sdk.HikEvents(session, lambda kind, code: print(f"event {kind} cmd=0x{code:04x}"))
            if not events.start():
                print("Could not open the alarm channel (device may not support it).")
            else:
                print(f"Listening for events for {args.seconds:.0f}s (Ctrl-C to stop)...")
                try:
                    time.sleep(args.seconds)
                except KeyboardInterrupt:
                    pass
                events.stop()
    finally:
        session.logout()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
