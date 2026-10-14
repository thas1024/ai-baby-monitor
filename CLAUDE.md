# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A baby monitor. A camera stream is pushed to Redis, a vision LLM checks the frames against natural-language rules from YAML config, and a single beep fires (`should_alert`) if a rule is violated. A Streamlit dashboard shows the live frame plus the LLM reasoning logs.

The vision backend is any **OpenAI-compatible endpoint** (cloud or custom), selected via `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL_NAME` / `LLM_API_MODE` in `.env`. A local vLLM (Qwen2.5-VL) service is kept as an optional compose profile for offline/self-hosted use, but is **not** started by default.

Redis runs as a compose service (`redis:7-alpine`, host port **16379** → 6379, persisted in the `redis-data` volume). Bridge services reach it as `redis:6379`; the host-networked watcher reaches it as `127.0.0.1:16379` (set in its service `environment`). Point `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` elsewhere if you'd rather use an external Redis.

## Commands

```bash
# Install deps (uv manages the venv; Python 3.12, .python-version)
uv sync

# Run the stack in Docker (redis, stream_to_redis, streamlit, watcher) — no GPU needed.
cp .env.template .env        # first time; fill in CAMERA_URI and LLM_*; .env is gitignored
docker compose up -d --build

# Optional local inference backend (GPU), instead of a cloud endpoint
docker compose --profile local up -d vllm

# Run the watcher on the HOST instead of in Docker — only needed to hear the
# alert beep (containers have no audio output). See "Watcher" below.
uv run scripts/run_watcher.py --config-file configs/living_room.yaml

# Streamlit dev server outside Docker (dashboard on :8501)
uv run streamlit run scripts/streamlit_viewer.py -- --config-files configs/living_room.yaml

# Re-lock after editing pyproject.toml (Dockerfile uses `uv sync --locked`)
uv lock

# Tests
uv run pytest                                   # all
uv run pytest tests/watcher/test_watcher.py     # one file
uv run pytest tests/watcher/test_watcher.py::test_watcher_init   # one test
uv run pytest -k fps                            # by name
```

There is no linter/formatter config in the repo. Tests are `pytest`, all mock-based (no real camera, Redis, or LLM endpoint needed).

## Architecture

A room has **one or more cameras** (angles). Frame streams are per camera
(`keys.camera_keys(room, camera)`); the logs stream is per room:

- `ai_baby_monitor:{room}:{camera}:realtime` — last 3 frames, `approximate=False`; read by the viewer. An **unnamed legacy camera omits `{camera}`** (`ai_baby_monitor:{room}:realtime`) so old deployments don't move keys.
- `ai_baby_monitor:{room}:{camera}:subsampled` — every Nth frame (`subsample_rate`), capped at `subsampled_stream_maxlen`; read by the watcher.
- `ai_baby_monitor:{room}:logs` — one room-level timeline: the merged verdict (`should_alert`, `awareness_level`, `reasoning`, `night_mode`) plus a representative-frame **thumbnail per camera** (`frame` for a legacy single camera, `frame__{camera}` for named ones) and non-verdict health records (`{timestamp, error}` / `{timestamp, info}`). The watcher records "no live video" / "video restored" only on the **leading edge** of a transition, so a long outage is two entries, not one per poll. `render_logs` shows a timeline; `deserialize_log` keeps `frame`/`frame__*` as bytes. Timestamps are stored as UTC epoch and rendered in `APP_TZ` (see below).

Data flow: `CameraStream` grabs a frame → JPEG-encodes it into a `Frame(frame_data: np.ndarray, timestamp, frame_idx)` → `RedisStreamHandler` stores raw bytes → the watcher deserializes and sends each frame as its own `image_url` content part (`data:image/jpeg;base64,...`) alongside the instructions text.

The watcher writes `watcher/base_prompt.py`'s prompt, which demands a JSON object matching `WatcherResponse`. The response is validated with Pydantic. `AwarenessLevel` is an enum (LOW/MEDIUM/HIGH); `WatcherResponse` and `AwarenessLevel` live in `watcher/watcher.py`.

### Structured output negotiation

Endpoints differ in structured-output support, so `Watcher._create_completion` tries, in order, `json_schema` → `json_object` → neither (prompt-only), catching `openai.BadRequestError` to fall back and caching the first option that works (moved to the front of `_response_format_options`). Both protocols are supported via `api_mode`:

- `api_mode="chat"` (default) — Chat Completions (`client.chat.completions.create`, `response_format=`, `max_tokens`, `image_url` parts).
- `api_mode="responses"` — Responses API (`client.responses.create`, `text={"format": ...}`, `max_output_tokens`, `input_image` parts). **Always streamed** (`stream=True`): some gateways reject non-streaming Responses requests with "Stream must be set to true", so `_consume_response_stream` reassembles the text from `response.output_text.delta` events.

`self.json_schema` is passed through `_make_strict`, which recursively adds `additionalProperties: false` and a full `required` list — OpenAI strict Structured Outputs reject Pydantic's default schema without them.

`_request` / `_build_response_format` / `_consume_response_stream` branch on `api_mode`. `_calculate_fps` is retained as a utility but is no longer used by the request path (frames are sent as individual images, not a vLLM video blob).

### Motion gating & request throttling

`src/ai_baby_monitor/motion.py` (pure functions, no Redis/config) controls how often the model is called and how many images each request carries:

- `frame_diff_score` / `frames_are_similar` — difference between two JPEG frames on a pinned pipeline (grayscale → 160x90 → Gaussian blur). Byte-identical → 0.0; undecodable → `inf` (treated as "changed", never raises). `threshold <= 0` disables comparison. Three metrics via `motion_metric`: `mean` (whole-frame mean abs diff), `grid_max` (max per-cell mean — sensitive to small localized motion), `changed_fraction` (share of pixels above a noise floor). Their scales differ, so `resolve_threshold` maps `motion_threshold=None` to a per-metric default; never reuse a value across metrics.
- `frame_saturation` / `is_night_frame` — detect infrared/night-vision frames (grayscale, HSV saturation ~0; threshold `night_saturation_threshold`, default 12.0). Decode failure → not night.
- `build_timeline(frames, threshold, metric, night_threshold)` — the first frame is always an image anchor; later frames are compared to the **last included** frame (slow drift accumulates); near-identical runs collapse into a `SkippedItem`. A **color↔night-vision switch** emits a `ModeTransitionItem` and re-anchors to the new mode, so a run of grayscale frames is compared against each other instead of the pre-switch color anchor (which would otherwise mark them all "changed"). `ImageItem.night` marks night frames. A window always yields ≥1 image.
- `should_analyze(...)` — the cycle gate: call when the newest frame differs from the last analyzed frame, or when the heartbeat (`max_interval_seconds`) elapses, floored by `min_interval_seconds`. `threshold <= 0` or no baseline → always call.

`Watcher._build_content` turns the timeline into protocol-neutral `TextPart`/`ImagePart` (each image is preceded by a `[frame N @ <time>]` annotation — with a `夜视/红外模式` marker when `night`; skipped runs and mode switches become text notes with no image). `_request` maps parts to chat/responses content. For multi-camera rooms, `process_room([CameraWindow, ...])` emits a `[camera: NAME]` header plus per-frame `[NAME · frame N @ …]` annotations, merging all cameras into one request; `process_frames(frames)` is a single-camera wrapper. `run_watcher.py` holds the gate state:

- **per-camera** baselines (last frame of the last **successful** window), plus a **single** shared `last_call_monotonic` (the heartbeat is room-wide);
- each cycle skips cameras with no frames; the model is called if **any** camera's gate opens **and** the room-wide `min_interval_seconds` floor has elapsed — the floor is applied in the loop, outside the per-camera OR, so a camera with no baseline can't bypass it;
- on a call a window is fetched for **every** camera (since its own baseline) so the merged snapshot is coherent; `night_mode` = ALL cameras night (desensitize only on pure IR); on success every camera's baseline advances, on failure none does.

The alert **action** is debounced (`src/ai_baby_monitor/alerts.py` `should_emit_alert`): the beep only fires on the leading edge of a violation confirmed by two consecutive positive verdicts within `min(1800, max(300, max_interval_seconds*3))` seconds. It is also gated by `is_alert_worthy`: in **night-vision mode** only a HIGH-confidence violation qualifies (grayscale/IR frames are noisier), so LOW/MEDIUM alerts are logged but do not beep. Every verdict is still written to `:logs` (with a `night_mode` flag) — only the beep is debounced/desensitized.

### Localization (i18n)

User-facing text is bilingual, selected by `APP_LANG` (`zh` default, or `en`).

- `src/ai_baby_monitor/i18n.py` holds all UI strings as `key -> {lang: text}` tables; call `t("key", **kwargs)` (placeholders are `{name}`). Missing keys fall back to the default language, then the key itself. The enum values `LOW`/`MEDIUM`/`HIGH` stay English in storage/schema; only their display labels (`awareness_LOW`…) are translated.
- `watcher/base_prompt.py` builds the LLM prompt per language (`get_instructions_prompt(instructions, language=...)`, defaults to `get_language()`); the `zh` prompt asks the model to answer in Chinese while keeping the JSON field names English. Both prompts explain that IR night-vision produces grayscale frames (normal, not a fault) and to be more conservative in them. The same module hosts the annotation formatters (`format_frame_annotation(..., night=...)`, `format_skipped_annotation`, `format_mode_transition_annotation`).
- `configs/*.yaml` sample instructions are Chinese; the streamlit mode options are stored as stable keys (`realtime`/`historic`) and translated for display via `st.radio(format_func=...)`.
- structlog log messages and code identifiers stay English on purpose.

Streamlit reads frames/logs back and renders them. The historic-logs view is a **timeline (newest first)**: every entry is a `st.expander` carrying its representative photo (`imaging.encode_thumbnail` stores a downscaled JPEG in the log entry); anomalous entries (alert or MEDIUM/HIGH awareness) default to expanded, normal ones to collapsed. Timestamps render in `APP_TZ` (IANA name; the container defaults to `TZ`). Note: `render_logs` opens an expander per entry, so callers must not wrap it in another expander (Streamlit forbids nesting). The viewer loop (`while True` in `scripts/streamlit_viewer.py`) polls Redis for the newest frame and log each rerun.

### Watcher

`scripts/run_watcher.py` runs either in Docker (the `watcher` compose service, default) or on the host. It reads `REDIS_HOST` / `REDIS_PORT` from the environment (the compose service overrides them to `127.0.0.1:16379` because it is host-networked). The main loop is wrapped in a restart-on-error loop with a short backoff, so a dependency (Redis or the LLM endpoint) that is slow to come up at boot does not kill the watcher.

The only reason to run it on the host is the alert beep: `_play_alert()` plays `assets/alert.wav` on a violation, and containers have no audio output. It is controlled by `ALERT_SOUND_ENABLED` (default true; the compose service sets it to `false`). `_play_alert` never raises — a missing audio backend (headless host) is caught and logged, so the watcher keeps running.

The `watcher` service uses `network_mode: host` so it can reach host-local services (Redis published on a LAN IP, and the OpenAI-compatible gateway bound on the host) exactly like a host process. A bridge-networked container cannot reach non-published host ports — that was the cause of earlier `APITimeoutError`s. The other services stay on the bridge network and reach Redis at its published LAN IP.

### Config

`RoomConfig` (`config/utils.py`, a `slots=True, kw_only=True` dataclass) is loaded from `configs/*.yaml`. A room has either a legacy single `camera:` mapping or a `cameras:` list (one `CameraConfig` per angle); specifying both raises. Room-level keys: `name`, `instructions`, `min_interval_seconds`, `max_interval_seconds`, `num_frames_to_process`. Per-camera keys (on each `CameraConfig`, or in the legacy `camera:` block): `uri`, optional `name` (required and unique for `cameras:`, `[A-Za-z0-9_-]+`), and the optional `subsampled_stream_maxlen`, `frame_width`, `frame_height`, `subsample_rate`, `motion_metric`, `motion_threshold`, `night_saturation_threshold`, `num_frames_to_process` (per-camera override of the room default). `RoomConfig` keeps read-only delegating properties (`camera_uri`, `motion_metric`, …) to the first camera for legacy consumers. `camera_uri` is a string ("0" webcam index or an RTSP URL); `${VAR}` is expanded from the environment. `stream_to_redis.py` converts all-digit URIs to `int`.

### Adding a room or camera (touches several places)

1. Add `configs/{room}.yaml` with a `camera:` (single) or `cameras:` (multi-angle) block.
2. Add **one `stream_to_redis` service per camera** in `docker-compose.yml`, each with its own `--camera NAME` (omit `--camera` for a single-camera config).
3. Add **one `watcher` service per room** (it handles all the room's cameras itself).
4. Pass the new config to the `streamlit_viewer` `--config-files` (space-separated list).

## Docker

`Dockerfile.python` builds a shared `python-base` service; `stream_to_redis`, `streamlit_viewer` and `watcher` `extends:` it. The image is `python:3.12-slim-bookworm` + `uv`; `ffmpeg libsm6 libxext6` are apt-installed (needed by OpenCV — removing them breaks `cv2`). `./scripts`, `./configs`, and `./src` are bind-mounted, so script/src/config edits don't require a rebuild.

`python-base` is a build-only base with no real work, so it runs `sleep infinity` (children override `command`); without a command it would exit and restart-loop. It `depends_on: [redis]`. `redis`, `stream_to_redis` and `streamlit_viewer` are on the bridge `app-network`; `watcher` is on the host network (see "Watcher"). The `vllm` service is gated behind `profiles: ["local"]` (GPU reservation, healthcheck, and `${VLLM_PORT}` live inside it) and is not created by a plain `docker compose up`.

`.env` is gitignored; `.env.template` is the tracked source of truth for the variables (`REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `CAMERA_URI`, `CAMERA_DECODE`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_NAME`, `LLM_API_MODE`, `ALERT_SOUND_ENABLED`, `APP_LANG`, `APP_TZ`, `TZ`, plus `VLLM_PORT` for the local profile).

### Frame decoding (CPU vs Intel GPU)

`CAMERA_DECODE` selects how `stream_to_redis` decodes frames: `cpu` (OpenCV decode), `vaapi` / `qsv` (decode on an Intel GPU via an `ffmpeg` subprocess), or `auto` (default — prefers **VAAPI**, then QSV, then CPU). `CameraStream` (`stream/camera_stream.py`) keeps the same `capture_new_frame()` interface; the GPU path pipes `bgr24` rawvideo from `ffmpeg` (`-hwaccel vaapi -hwaccel_device /dev/dri/renderD128 -hwaccel_output_format vaapi -vf hwdownload,format=nv12,scale=W:H,format=bgr24`), keeps only the newest frame, and restarts ffmpeg on stream end. If no Intel hardware is available (no `/dev/dri`, or ffmpeg lacks the hwaccel), `auto` falls back to `cpu`.

The `stream_to_redis` service passes the GPU through: `devices: [/dev/dri:/dev/dri]` and `group_add: ["44", "105"]` (video/render gids). The image installs `libva2 libva-drm2 intel-media-va-driver vainfo`. Note: decode moves to the GPU but the 4K→target `scale` and JPEG encode stay on the CPU — on a weak CPU prefer the camera's **substream** (`Channels/102`) as well.

## Gotchas

- `pyproject.toml` lists every module imported directly (`numpy`, `pillow`, `pydantic`, `pyyaml`, `openai`) — these used to arrive transitively via `vllm`, which is no longer a dependency. Adding a new direct import means adding it here too, then `uv lock`.
- `Frame.frame_data` holds JPEG bytes as a `np.ndarray` (from `cv2.imencode`), not a decoded image, until the viewer re-opens it with PIL. Don't decode it in the transport path. The watcher additionally stores a small thumbnail of the last window frame in each log entry (`imaging.encode_thumbnail`) for the UI timeline.
- `RedisStreamHandler.deserialize_log` keeps binary fields (`frame`) as bytes and only decodes text fields — decoding image bytes as UTF-8 would corrupt/raise.
- `RedisStreamHandler` takes `redis_password` (`None` = no auth); all three entry points read it from `REDIS_PASSWORD`.
- `RedisStreamHandler.get_latest_entries` uses `xrevrange` (newest-first) and reverses the result to return chronological order — the reversal is intentional and asserted by tests.
- `run_watcher.py` exits(1) if `instructions` is empty or if any `LLM_*` env var is missing.
- The alert sound path is `assets/alert.wav` relative to CWD; run the watcher from the repo root. It is skipped entirely when `ALERT_SOUND_ENABLED=false`.
- Motion gating changes behavior by design: (1) it suppresses repeat `should_alert` beeps while a violation persists (the model is only re-asked when motion/heartbeat allows); (2) a static scene is only re-evaluated at the heartbeat cadence; (3) an event that occurs and resolves entirely inside a skipped gap is invisible. `motion_threshold: 0` restores the old "send everything, call every cycle" behavior.
- `motion_threshold` is not portable across cameras (AGC/IR/compression shift the noise floor). Every analyze logs `motion_score`; to calibrate, run on an empty room with gating off and set the threshold just above the idle noise. For a safety monitor, prefer false positives (lower threshold) and rely on `min_interval_seconds` for cost control.
- `subsampled_stream_maxlen` must cover `max_interval_seconds` at the subsampled frame rate, otherwise catch-up after a long heartbeat truncates the gap; the watcher warns ("does not cover the full gap") when this happens.
- A grayscale frame is **night-vision (IR) mode**, not a camera fault — don't treat "the picture is black-and-white" as an error. The prompt tells the model this, night frames are annotated, and in night mode only HIGH-confidence violations beep (`is_alert_worthy`). If a camera's "B&W" has a slight tint, lower `night_saturation_threshold`.
- A **flat/blank frame** (uniform gray value ~128, grayscale std ~0) means the **stream has no signal** (RTSP pull failed), *not* night-vision — `is_night_frame` requires actual content (`is_blank_frame` excludes uniform frames). The watcher drops blank cameras, logs a `no live video` error record (throttled to `min_interval_seconds`) instead of calling the model, and does not advance their baseline; if this appears often, fix the capture side.
- Multi-camera cost scales with the number of cameras × per-camera images per request (up to `num_frames_to_process` each on a first window). Keep `--limit-mm-per-prompt image=` (vLLM) above the total, and rely on per-camera `motion_threshold` (a static angle compresses to one anchor image) plus the room-level `min_interval_seconds`.
- Migrating a room from the legacy `camera:` block to a named `cameras:` list **changes its Redis keys** (`{room}:subsampled` → `{room}:{camera}:subsampled`); no data migration is needed, just restart the producers — but an unnamed camera keeps the legacy unscoped keys.
- Hardware decode (`CAMERA_DECODE=vaapi`) needs `/dev/dri` passed into the container and the Intel VAAPI driver; the `stream_to_redis` service already mounts it. Verified working on an Intel N100 (iHD driver, HEVC Main/Main10). QSV is often unavailable in containers (needs the MFX/oneVPL runtime) — `auto` prefers VAAPI for that reason.
