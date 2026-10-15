# Backlog / TODO

Open work and known follow-ups. Kept in the repo so it travels with the code
(see `CLAUDE.md` for the current architecture).

## TODO-1 — Persist logs to files for long-term (≥7 day) retention

**Problem.** Logs live only in the Redis stream `ai_baby_monitor:{room}:logs`,
capped at `maxlen = 3600*6` (≈6 hours). Beyond that, verdicts and representative
thumbnails are gone — too short for reviewing "what happened last week".

**Goal.** Write logs (verdict + per-camera thumbnails + non-verdict health
records) to files with **at least 7 days** of retention. Redis stays the hot /
recent source the dashboard reads.

**Design points.**
- Per-day files, e.g. `logs/{room}/{YYYY-MM-DD}.jsonl`; thumbnails inline base64
  or as separate `frames/{date}/{ts}_{camera}.jpg`.
- Sink options: the watcher appends the file alongside its Redis write (same
  process; a file error must not affect alerting), or a separate archiver worker
  consumes the Redis stream.
- Read layer: the historic view needs to read across Redis + files (recent from
  Redis, older from files).
- Rolling retention: delete files older than N days, configurable
  (`LOG_RETENTION_DAYS`, default 7).
- Volume mounting + permissions in the containers.
- No historical backfill — start writing at deploy time.

## TODO-2 — Ezviz cloud connection as an alternative input

Connect via the **Ezviz Open Platform** (`open.ys7.com`) — `EZOpenSDK` /
OpenAPI — instead of local RTSP/HCNetSDK.

- Credentials: `appKey` + `appSecret` → `accessToken`; device needs **SN +
  verification code** and must be **bound to an Ezviz account**.
- Cloud relay (P2P) — works remotely, could be a fallback when local access is
  unavailable; but it sends video off-network, needs an account/binding, some
  APIs are billed, and latency is higher.
- **Cannot** read/change the device verification code or password, or clear a
  local lock — those are local-only operations.
- No official Python SDK; would need a thin HTTP client over the OpenAPI
  (`accessToken` → live-address / snapshot / PTZ / alarm endpoints).
- Status: **not implemented** — recorded as an alternative only.

## Operational follow-ups (not code)

- **The camera is refusing all local access**: HCNetSDK login returns `err=153`
  (user locked) and RTSP DESCRIBE 404s for every credential / path. Needs a
  **local power-cycle**, and the Ezviz app must have **LAN/RTSP enabled**. Then
  verify with:
  ```bash
  uv run python tools/probe_camera.py --host <CAMERA_IP>                      # RTSP
  uv run python scripts/hik_control.py --password <DEVICE_PASSWORD> status    # SDK
  ```
  SDK device-management password is separate from the RTSP verification code
  (`CAMERA_SDK_PASSWORD` in `.env`, gitignored).
- **RTSP substream**: the camera's main stream is 4K H.265 — prefer the substream
  (`Channels/102`, 720p) to keep CPU down; the deployed config already uses it.
