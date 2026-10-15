#!/usr/bin/env python3
"""Probe a Hikvision/EZVIZ camera on the LAN and grab a single frame.

Tries the common local access methods and saves whatever works, so you can
confirm connectivity without running the whole stack:

  1. RTSP main / sub stream (Hikvision-style paths) via ffmpeg (one frame).
  2. ISAPI HTTP snapshot (no SDK needed).

Credentials come from --user/--password or from CAMERA_URI in the environment /
.env. Run from the repo root:

    uv run python tools/probe_camera.py --host <CAMERA_IP>
    uv run python tools/probe_camera.py --host <CAMERA_IP> --user admin --password <CODE>
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is a project dependency
    load_dotenv = None

RTSP_PATHS = {
    "main-hik": "/Streaming/Channels/101",
    "sub-hik": "/Streaming/Channels/102",
    "main-ezviz": "/h264/ch1/main/av_stream",
    "sub-ezviz": "/h264/ch1/sub/av_stream",
}


def _from_camera_uri() -> dict:
    """Parse CAMERA_URI (rtsp://user:pass@host:port/...) into connection parts."""
    uri = os.getenv("CAMERA_URI", "")
    m = re.match(r"rtsp://(?P<user>[^:]+):(?P<pw>[^@]+)@(?P<host>[^:/]+)(?::(?P<port>\d+))?", uri)
    if not m:
        return {}
    return {"user": m["user"], "password": m["pw"], "host": m["host"], "port": m["port"] or "554"}


def grab_rtsp_ffmpeg(uri: str, out: Path, timeout: int = 20) -> bool:
    """Grab one frame with ffmpeg over RTSP (TCP)."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp", "-i", uri, "-frames:v", "1", "-y", str(out),
    ]
    try:
        subprocess.run(cmd, timeout=timeout, check=True)
    except Exception as e:
        print(f"    ffmpeg failed: {e}")
        return False
    return out.exists() and out.stat().st_size > 0


def grab_rtsp_opencv(uri: str, out: Path) -> bool:
    """Grab one frame with OpenCV (no ffmpeg binary needed)."""
    try:
        import cv2
    except Exception as e:
        print(f"    OpenCV unavailable: {e}")
        return False
    cap = cv2.VideoCapture(uri)
    try:
        if not cap.isOpened():
            print("    OpenCV could not open the stream")
            return False
        ok, frame = cap.read()
        if not ok or frame is None:
            print("    OpenCV read no frame")
            return False
        return bool(cv2.imwrite(str(out), frame))
    finally:
        cap.release()


def grab_rtsp(uri: str, out: Path) -> bool:
    """Try ffmpeg, then OpenCV, to grab a single frame."""
    if shutil.which("ffmpeg"):
        if grab_rtsp_ffmpeg(uri, out):
            return True
    return grab_rtsp_opencv(uri, out)


def grab_isapi_snapshot(
    host: str, port: str, user: str, password: str, channel: str, out: Path, timeout: int = 8
) -> bool:
    """Grab a JPEG snapshot via the ISAPI HTTP endpoint (basic auth)."""
    import base64

    url = f"http://{host}:{port}/ISAPI/Streaming/channels/{channel}/picture"
    req = urllib.request.Request(url)
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    req.add_header("Authorization", f"Basic {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except Exception as e:
        print(f"    ISAPI {port} failed: {e}")
        return False
    if not data or data[:2] != b"\xff\xd8":  # JPEG magic
        print(f"    ISAPI {port} returned non-JPEG data ({len(data)} bytes)")
        return False
    out.write_bytes(data)
    return True


def main() -> int:
    if load_dotenv:
        load_dotenv()

    parsed = _from_camera_uri()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=parsed.get("host"), help="camera IP")
    p.add_argument("--port", default=parsed.get("port", "554"), help="RTSP port (default 554)")
    p.add_argument("--user", default=parsed.get("user", "admin"))
    p.add_argument("--password", default=parsed.get("password"))
    p.add_argument("--out-dir", default="/tmp", help="where to write the frames")
    args = p.parse_args()

    if not args.host or not args.password:
        print("Need --host and --password (or a CAMERA_URI in .env)", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    auth = f"{args.user}:{args.password}"
    ok_any = False

    print(f"== Camera {args.host} ==")

    print("[ISAPI snapshot over HTTP]")
    for http_port in ("80", "8000"):
        for ch in ("101", "102", "1"):
            out = out_dir / f"snapshot_{http_port}_{ch}.jpg"
            if grab_isapi_snapshot(args.host, http_port, args.user, args.password, ch, out):
                print(f"    OK -> {out} ({out.stat().st_size} bytes)")
                ok_any = True

    print("[RTSP via ffmpeg]")
    for name, path in RTSP_PATHS.items():
        uri = f"rtsp://{auth}@{args.host}:{args.port}{path}"
        out = out_dir / f"frame_{name}.jpg"
        print(f"  {uri.replace(args.password, '***')}")
        if grab_rtsp(uri, out):
            print(f"    OK -> {out} ({out.stat().st_size} bytes)")
            ok_any = True

    print("== done ==" if ok_any else "== nothing worked ==")
    return 0 if ok_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
