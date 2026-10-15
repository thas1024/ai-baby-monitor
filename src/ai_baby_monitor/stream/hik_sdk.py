"""Hikvision/EZVIZ device-network SDK (HCNetSDK) integration.

Wraps the vendor SDK via the ``HCNetSDK-python`` wheel (a ctypes binding that
bundles the vendor ``.so`` files). Provides, beyond RTSP:

* a login **session** (``NET_DVR_Login_V40``),
* **device events** (motion / smart detection) over the alarm channel,
* **PTZ** control and on-demand **JPEG snapshot**,
* an optional **frame source** (``RealPlay`` + PlayM4 decode).

Everything here is defensive: the SDK is a closed-source ``.so`` and the package
may be absent (e.g. non-x86_64), so imports and calls are guarded and every
public helper degrades to a clear failure instead of crashing the process. Callers
(``CameraStream``) fall back to RTSP.

Loading notes (verified): only ``libhcnetsdk.so`` is needed for login/events/PTZ/
snapshot — ``libPlayCtrl.so`` (used only for the frame decoder) hangs a headless
loader because it has no RPATH for its bundled render libs, so we preload those
by absolute path with ``RTLD_GLOBAL`` rather than setting a global
``LD_LIBRARY_PATH`` (which would shadow system libs).
"""

import ctypes
import datetime as dt
import os
import re
import threading
import time
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()

# Common Hikvision/EZVIZ RTSP shape: rtsp://user:pass@host[:port]/...
_RTSP_RE = re.compile(r"^rtsp://(?P<user>[^:]+):(?P<pw>[^@]+)@(?P<host>[^:/]+)(?::(?P<port>\d+))?")

# PTZ commands (subset of the SDK's PTZCommand enum).
PTZ_COMMANDS = {
    "left": 23, "right": 24, "up": 21, "down": 22,
    "upleft": 25, "upright": 26, "downleft": 27, "downright": 28,
    "zoom_in": 11, "zoom_out": 12, "focus_near": 13, "focus_far": 14,
}
PTZ_GOTO_PRESET = 39
PTZ_SET_PRESET = 8
PTZ_CLEAR_PRESET = 9


def parse_rtsp_uri(uri: str) -> dict | None:
    """Split an RTSP URL into {user, password, host, port}; None if it doesn't match."""
    m = _RTSP_RE.match(uri or "")
    if not m:
        return None
    return {
        "user": m["user"],
        "password": m["pw"],
        "host": m["host"],
        "port": int(m["port"] or 554),
    }


# --- package / library loading (guarded) ---

_import_error: str | None = None


def _import_sdk():
    """Import the HCNetSDK package, returning (module_or_None, error_str)."""
    global _import_error
    try:
        import HCNetSDK  # noqa: F401  (bundles the vendor .so)

        return HCNetSDK, None
    except Exception as e:  # ImportError, OSError from missing .so, ...
        _import_error = f"{type(e).__name__}: {e}"
        return None, _import_error


def hik_available() -> bool:
    """Whether the HCNetSDK package can be imported on this machine/image."""
    return _import_sdk()[0] is not None


def _preload_render_libs() -> None:
    """Best-effort preload of libPlayCtrl's siblings (needed only for decoding)."""
    try:
        import HCNetSDK

        libdir = os.path.join(os.path.dirname(HCNetSDK.__file__), "Libs", "linux")
        for name in ("libAudioRender.so", "libSuperRender.so"):
            path = os.path.join(libdir, name)
            if os.path.exists(path):
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        logger.debug("Skipping render-lib preload", error=str(e))


# --- login session ---

class HikSession:
    """A logged-in HCNetSDK session (``NET_DVR_Login_V40`` / ``NET_DVR_Logout``).

    Not thread-safe beyond the SDK's own guarantees; use one per process/camera.
    """

    def __init__(self, host: str, user: str, password: str, port: int = 8000, channel: int = 1):
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.channel = channel
        self.user_id: int | None = None
        self._sdk = None
        self._device = None

    def login(self) -> bool:
        module, err = _import_sdk()
        if module is None:
            logger.warning("HCNetSDK unavailable", error=err)
            return False
        try:
            from HCNetSDK import NetClient
            from HCNetSDK.SDK_Struct import NET_DVR_DEVICEINFO_V40

            self._sdk = NetClient()  # singleton: __init__ loads lib + SetSDKInitCfg
            self._sdk.Init()
            self._sdk.SetConnectTime(3000, 2)
            self._sdk.SetReconnect(10000, True)
            self._device = NET_DVR_DEVICEINFO_V40()
            user_id, self._device = self._sdk.Login_V40(
                self.host, self.port, self.user, self.password
            )
            self.user_id = int(user_id)
            if self.user_id < 0:
                logger.warning(
                    "HCNetSDK login failed", host=self.host, err=self.last_error()
                )
                return False
            logger.info("HCNetSDK login ok", host=self.host, serial=self.serial())
            return True
        except Exception as e:
            logger.warning("HCNetSDK login error", host=self.host, error=str(e))
            self.user_id = None
            return False

    def last_error(self) -> int:
        try:
            return int(self._sdk.sdk.NET_DVR_GetLastError())
        except Exception:
            return -1

    def serial(self) -> str:
        try:
            raw = bytes(self._device.struDeviceV30.sSerialNumber)
            return raw.split(b"\x00")[0].decode(errors="ignore")
        except Exception:
            return ""

    def logout(self) -> None:
        if self._sdk is not None and self.user_id and self.user_id >= 0:
            try:
                self._sdk.Logout(self.user_id)
            except Exception as e:
                logger.debug("HCNetSDK logout error", error=str(e))
        self.user_id = None

    # --- control-plane helpers (raw CDLL; the wheel does not wrap these) ---

    def _lib(self):
        return self._sdk.sdk  # the raw CDLL of libhcnetsdk.so

    def snapshot(self) -> bytes | None:
        """Capture a JPEG snapshot directly from the device (no stream needed)."""
        if not self.user_id or self.user_id < 0:
            return None
        try:
            from HCNetSDK import NET_DVR_JPEGPARA

            jpeg = NET_DVR_JPEGPARA()
            jpeg.wPicSize = 0xFF  # capture at current resolution
            jpeg.wPicQuality = 0
            buf = ctypes.create_string_buffer(2 * 1024 * 1024)
            got = ctypes.c_uint(0)
            fn = self._lib().NET_DVR_CaptureJPEGPicture_NEW
            fn.restype = ctypes.c_bool
            ok = fn(
                ctypes.c_long(self.user_id), self.channel, ctypes.byref(jpeg),
                buf, ctypes.sizeof(buf), ctypes.byref(got),
            )
            if not ok or got.value < 2:
                logger.warning("Snapshot failed", err=self.last_error())
                return None
            return buf.raw[: got.value]
        except Exception as e:
            logger.warning("Snapshot error", error=str(e))
            return None

    def ptz_move(self, direction: str, speed: int = 4, seconds: float = 0.6) -> bool:
        """Nudge the camera: send start, hold briefly, send stop."""
        cmd = PTZ_COMMANDS.get(direction.lower())
        if cmd is None or not self.user_id or self.user_id < 0:
            return False
        try:
            fn = self._lib().NET_DVR_PTZControlWithSpeed_Other
            fn.restype = ctypes.c_bool
            a = (ctypes.c_long(self.user_id), self.channel, cmd, speed)
            fn(*a, 0)  # start
            time.sleep(seconds)
            fn(*a, 1)  # stop
            return True
        except Exception as e:
            logger.warning("PTZ move error", error=str(e))
            return False

    def ptz_preset(self, command: str, index: int) -> bool:
        cmds = {"goto": PTZ_GOTO_PRESET, "set": PTZ_SET_PRESET, "clear": PTZ_CLEAR_PRESET}
        cmd = cmds.get(command.lower())
        if cmd is None or not self.user_id or self.user_id < 0:
            return False
        try:
            fn = self._lib().NET_DVR_PTZPreset_Other
            fn.restype = ctypes.c_bool
            return bool(fn(ctypes.c_long(self.user_id), self.channel, cmd, index))
        except Exception as e:
            logger.warning("PTZ preset error", error=str(e))
            return False

    def ptz_position(self) -> dict | None:
        """Read the current P/T/Z; returns None on error."""
        try:
            return self._sdk.GetDVRConfig_PTZ(self.user_id)
        except Exception as e:
            logger.debug("PTZ position unavailable", error=str(e))
            return None

    def ptz_supported(self) -> bool:
        """Cheap heuristic: a working PTZ reports a non-zero position."""
        pos = self.ptz_position() or {}
        return any(int(pos.get(k, 0) or 0) for k in ("wPanPos", "wTiltPos", "wZoomPos"))


# --- events (alarm channel) ---

class HikEvents:
    """Realtime device events (motion / smart) via the SDK alarm channel.

    ``on_event(kind, code)`` is invoked from the SDK's callback thread — it must be
    cheap and must not call back into the SDK.
    """

    def __init__(self, session: HikSession, on_event):
        self.session = session
        self.on_event = on_event
        self._cb = None  # keep a strong ref: ctypes GCs otherwise
        self._handle = None

    def start(self) -> bool:
        if not self.session.user_id or self.session.user_id < 0:
            return False
        try:
            from HCNetSDK import MSGCallBack_V31, NET_DVR_SETUPALARM_PARAM

            self._cb = MSGCallBack_V31(self._on_message)
            sdk = self.session._lib()
            setcb = sdk.NET_DVR_SetDVRMessageCallBack_V31
            setcb.restype = ctypes.c_bool
            setcb(self._cb, None)

            param = NET_DVR_SETUPALARM_PARAM()
            param.dwSize = ctypes.sizeof(param)
            param.byLevel = 1
            param.byAlarmInfoType = 1
            setup = sdk.NET_DVR_SetupAlarmChan_V41
            setup.restype = ctypes.c_long
            self._handle = setup(ctypes.c_long(self.session.user_id), ctypes.byref(param))
            if not self._handle or int(self._handle) < 0:
                logger.warning("Alarm channel setup failed", err=self.session.last_error())
                return False
            logger.info("Alarm channel open")
            return True
        except Exception as e:
            logger.warning("Alarm channel error", error=str(e))
            return False

    def _on_message(self, lCommand, pAlarmer, pAlarmInfo, dwBufLen, pUser) -> bool:
        try:
            kind = _classify_event(int(lCommand))
            if kind:
                self.on_event(kind, int(lCommand))
        except Exception as e:  # never raise across the C boundary
            logger.debug("Event callback error", error=str(e))
        return True

    def stop(self) -> None:
        if self._handle and self.session._sdk is not None:
            try:
                close = self.session._lib().NET_DVR_CloseAlarmChan_V30
                close.restype = ctypes.c_bool
                close(ctypes.c_long(self._handle))
            except Exception as e:
                logger.debug("Alarm channel close error", error=str(e))
        self._handle = None


def _classify_event(l_command: int) -> str | None:
    """Map a comms command into a coarse event kind (best-effort; firmware varies)."""
    # COMM_ALARM 0x1100, COMM_ALARM_V30 0x4000 -> classic (dwAlarmType==3 is motion)
    if l_command in (0x1100, 0x4000):
        return "alarm"
    if l_command == 0x1102:  # COMM_ALARM_RULE (smart: line/intrusion/human)
        return "smart"
    if l_command == 0x6009:  # COMM_ISAPI_ALARM (JSON payload on newer firmware)
        return "isapi"
    return None


# --- optional frame source (RealPlay + PlayM4 decode) ---

class HikFrameSource:
    """Decoded BGR frames from the device via RealPlay + PlayM4.

    Best-effort: requires ``libPlayCtrl.so`` to load (see module docstring). Keeps a
    single newest-frame slot with ``next_frame(timeout)`` mirroring the ffmpeg
    decoder in ``camera_stream`` so ``CameraStream`` can treat both the same way.
    """

    def __init__(self, session: HikSession, stream_type: int = 1):
        self.session = session
        self.stream_type = stream_type  # 1 = sub stream
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._served = 0
        self._stop = threading.Event()
        self._real_handle = None
        self._port = -1
        self._real_cb = None
        self._dec_cb = None

    def start(self, timeout: float = 15.0) -> bool:
        if not self.session.user_id or self.session.user_id < 0:
            return False
        _preload_render_libs()
        try:
            from HCNetSDK import (
                NET_DVR_PREVIEWINFO,
                REALDATACALLBACK,
                DECCBFUNWIN,
            )

            sdk = self.session._lib()
            sdk.NET_DVR_RealPlay_V40.restype = ctypes.c_long
            self._real_cb = REALDATACALLBACK(self._on_stream)
            self._dec_cb = DECCBFUNWIN(self._on_decoded)
            preview = NET_DVR_PREVIEWINFO()
            preview.lChannel = self.session.channel
            preview.dwStreamType = self.stream_type
            preview.dwLinkMode = 0     # TCP private protocol
            preview.hPlayWnd = 0
            preview.bBlocked = 0
            self._real_handle = sdk.NET_DVR_RealPlay_V40(
                ctypes.c_long(self.session.user_id), ctypes.byref(preview), self._real_cb, None
            )
            if not self._real_handle or int(self._real_handle) < 0:
                logger.warning("RealPlay failed", err=self.session.last_error())
                return False
            with self._cond:
                self._cond.wait_for(lambda: self._seq > 0, timeout)
            return self._seq > 0
        except Exception as e:
            logger.warning("RealPlay error", error=str(e))
            return False

    def _on_stream(self, handle, data_type, buf, size, user) -> None:
        # data_type 1 = SYSHEAD, 2 = STREAMDATA (RealDataType)
        try:
            client = self.session._sdk
            if data_type == 1 and self._port < 0:
                ok, port = client.GetPort()
                self._port = int(port)
                client.SetStreamOpenMode(self._port, 0)  # STREAM_REALTIME
                client.OpenStream(self._port, buf, size, 1024 * 1024)
                client.SetDecCallBackExMend(self._port, self._dec_cb)
                client.Play(self._port, None)
            elif data_type == 2 and self._port >= 0:
                client.InputData(self._port, buf, size)
        except Exception as e:
            logger.debug("Stream callback error", error=str(e))

    def _on_decoded(self, port, buf, size, frame_info, user, reserved) -> None:
        # Only convert when a consumer is waiting (the app runs ~2.5 fps; decode is
        # full-rate, the YUV->BGR conversion is the expensive part we can skip).
        with self._cond:
            if self._seq == self._served:
                return
        try:
            import cv2
            import numpy as np

            info = frame_info.contents
            w, h = int(info.nWidth), int(info.nHeight)
            raw = ctypes.string_at(buf, size)
            yuv = np.frombuffer(raw, np.uint8, count=w * h * 3 // 2).reshape(h * 3 // 2, w)
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YV12)
            with self._cond:
                self._frame = bgr
                self._seq += 1
                self._cond.notify_all()
        except Exception as e:
            logger.debug("Decode callback error", error=str(e))

    def next_frame(self, timeout: float):
        with self._cond:
            if self._seq == self._served:
                self._cond.wait_for(lambda: self._seq != self._served, timeout)
            if self._seq == self._served:
                return None
            self._served = self._seq
            return self._frame

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        try:
            client = self.session._sdk
            if self._real_handle and int(self._real_handle) >= 0:
                client.StopRealPlay(self._real_handle)
            if self._port >= 0:
                client.Stop(self._port)
                client.CloseStream(self._port)
                client.FreePort(self._port)
        except Exception as e:
            logger.debug("Frame source stop error", error=str(e))
        self._real_handle = None
        self._port = -1
