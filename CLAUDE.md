# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A baby monitor. A camera stream is pushed to Redis, a vision LLM checks the frames against natural-language rules from YAML config, and a single beep fires (`should_alert`) if a rule is violated. A Streamlit dashboard shows the live frame plus the LLM reasoning logs.

The vision backend is any **OpenAI-compatible endpoint** (cloud or custom), selected via `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL_NAME` / `LLM_API_MODE` in `.env`. A local vLLM (Qwen2.5-VL) service is kept as an optional compose profile for offline/self-hosted use, but is **not** started by default.

Redis is also **external** (not managed by compose) — point `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` at an existing instance.

## Commands

```bash
# Install deps (uv manages the venv; Python 3.12, .python-version)
uv sync

# Run the stack in Docker (stream_to_redis, streamlit, watcher) — no GPU needed.
# Requires an external Redis reachable at REDIS_HOST:REDIS_PORT (see .env).
cp .env.template .env        # first time; fill in REDIS_* and LLM_*; .env is gitignored
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

Three per-room Redis streams, all built by `keys.room_keys(room_name)` (`src/ai_baby_monitor/keys.py`) — this is the contract between the three scripts, which all address the same keys via that helper:

- `ai_baby_monitor:{room}:realtime` — last 3 frames, `approximate=False`; read by the Streamlit viewer for the live image.
- `ai_baby_monitor:{room}:subsampled` — every Nth frame (`subsample_rate`), capped at `subsampled_stream_maxlen`; read by the watcher in batches of `num_frames_to_process`.
- `ai_baby_monitor:{room}:logs` — watcher results (`should_alert`, `awareness_level`, `reasoning`) and error records (`{timestamp, error}`); read by the viewer. `render_logs` handles both shapes.

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

- `frame_diff_score` / `frames_are_similar` — difference between two JPEG frames on a pinned pipeline (grayscale → 160x90 → Gaussian blur → mean abs diff). Byte-identical → 0.0; undecodable → `inf` (treated as "changed", never raises). `threshold <= 0` disables comparison.
- `build_timeline(frames, threshold)` — the first frame is always an image anchor; each later frame is compared to the **last included** frame (slow drift accumulates); near-identical runs collapse into a `SkippedItem`. A window always yields ≥1 image.
- `should_analyze(...)` — the cycle gate: call when the newest frame differs from the last analyzed frame, or when the heartbeat (`max_interval_seconds`) elapses, floored by `min_interval_seconds`. `threshold <= 0` or no baseline → always call.

`Watcher._build_content` turns the timeline into protocol-neutral `TextPart`/`ImagePart` (each image is preceded by a `[frame N @ <time>]` annotation; skipped runs become a text note with no image). `_request` maps parts to chat/responses content. `run_watcher.py` holds the gate state:

- baseline = the last frame of the last **successful** window (a failed call is retried next cycle);
- interval timing uses `time.monotonic()`, not frame timestamps;
- when the gate opens it fetches frames **since the last analyzed window** (`RedisStreamHandler.get_frames_since`) so no gap is skipped — capped by `subsampled_stream_maxlen`, and it warns if that cap truncates the gap.

### Localization (i18n)

User-facing text is bilingual, selected by `APP_LANG` (`zh` default, or `en`).

- `src/ai_baby_monitor/i18n.py` holds all UI strings as `key -> {lang: text}` tables; call `t("key", **kwargs)` (placeholders are `{name}`). Missing keys fall back to the default language, then the key itself. The enum values `LOW`/`MEDIUM`/`HIGH` stay English in storage/schema; only their display labels (`awareness_LOW`…) are translated.
- `watcher/base_prompt.py` builds the LLM prompt per language (`get_instructions_prompt(instructions, language=...)`, defaults to `get_language()`); the `zh` prompt asks the model to answer in Chinese while keeping the JSON field names English.
- `configs/*.yaml` sample instructions are Chinese; the streamlit mode options are stored as stable keys (`realtime`/`historic`) and translated for display via `st.radio(format_func=...)`.
- structlog log messages and code identifiers stay English on purpose.

Streamlit reads frames/logs back and renders them; the viewer loop (`while True` in `scripts/streamlit_viewer.py`) polls Redis for the newest frame and log each rerun.

### Watcher

`scripts/run_watcher.py` runs either in Docker (the `watcher` compose service, default) or on the host. It reads `REDIS_HOST` from `.env`. Redis is external and addressed the same way from both places.

The only reason to run it on the host is the alert beep: `_play_alert()` plays `assets/alert.wav` on a violation, and containers have no audio output. It is controlled by `ALERT_SOUND_ENABLED` (default true; the compose service sets it to `false`). `_play_alert` never raises — a missing audio backend (headless host) is caught and logged, so the watcher keeps running.

The `watcher` service uses `network_mode: host` so it can reach host-local services (Redis published on a LAN IP, and the OpenAI-compatible gateway bound on the host) exactly like a host process. A bridge-networked container cannot reach non-published host ports — that was the cause of earlier `APITimeoutError`s. The other services stay on the bridge network and reach Redis at its published LAN IP.

### Config

`RoomConfig` (`config/utils.py`, a `slots=True, kw_only=True` dataclass) is loaded from `configs/*.yaml`. Required top-level keys: `name`, `camera.uri`, `instructions`. Optional camera keys (`subsampled_stream_maxlen`, `frame_width`, `frame_height`, `subsample_rate`, `num_frames_to_process`, `motion_threshold`, `min_interval_seconds`, `max_interval_seconds`) fall back to dataclass defaults when absent — the loader only passes keys present in the YAML. `camera_uri` is a string in YAML ("0" for webcam index, or an RTSP URL); `stream_to_redis.py` converts all-digit URIs to `int`. The motion knobs are per-room (cameras differ); see "Motion gating & request throttling".

### Adding a room (touches several places)

1. Add `configs/{room}.yaml`.
2. Add a `stream_to_redis` service in `docker-compose.yml` pointing at that config.
3. Add a `watcher` service (or run `run_watcher.py` on the host) with that config.
4. Pass the new config to the `streamlit_viewer` `--config-files` (space-separated list).

## Docker

`Dockerfile.python` builds a shared `python-base` service; `stream_to_redis`, `streamlit_viewer` and `watcher` `extends:` it. The image is `python:3.12-slim-bookworm` + `uv`; `ffmpeg libsm6 libxext6` are apt-installed (needed by OpenCV — removing them breaks `cv2`). `./scripts`, `./configs`, and `./src` are bind-mounted, so script/src/config edits don't require a rebuild.

`python-base` is a build-only base with no real work, so it runs `sleep infinity` (children override `command`); without a command it would exit and restart-loop. It has no `depends_on` (Redis is external). `stream_to_redis` and `streamlit_viewer` are on the bridge `app-network`; `watcher` is on the host network (see "Watcher"). The `vllm` service is gated behind `profiles: ["local"]` (GPU reservation, healthcheck, and `${VLLM_PORT}` live inside it) and is not created by a plain `docker compose up`.

`.env` is gitignored; `.env.template` is the tracked source of truth for required variables (`REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `CAMERA_URI`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_NAME`, `LLM_API_MODE`, `ALERT_SOUND_ENABLED`, `APP_LANG`, plus `VLLM_PORT` for the local profile).

## Gotchas

- `pyproject.toml` lists every module imported directly (`numpy`, `pillow`, `pydantic`, `pyyaml`, `openai`) — these used to arrive transitively via `vllm`, which is no longer a dependency. Adding a new direct import means adding it here too, then `uv lock`.
- `Frame.frame_data` holds JPEG bytes as a `np.ndarray` (from `cv2.imencode`), not a decoded image, until the viewer re-opens it with PIL. Don't decode it in the transport path.
- `RedisStreamHandler` takes `redis_password` (`None` = no auth); all three entry points read it from `REDIS_PASSWORD`.
- `RedisStreamHandler.get_latest_entries` uses `xrevrange` (newest-first) and reverses the result to return chronological order — the reversal is intentional and asserted by tests.
- `run_watcher.py` exits(1) if `instructions` is empty or if any `LLM_*` env var is missing.
- The alert sound path is `assets/alert.wav` relative to CWD; run the watcher from the repo root. It is skipped entirely when `ALERT_SOUND_ENABLED=false`.
- Motion gating changes behavior by design: (1) it suppresses repeat `should_alert` beeps while a violation persists (the model is only re-asked when motion/heartbeat allows); (2) a static scene is only re-evaluated at the heartbeat cadence; (3) an event that occurs and resolves entirely inside a skipped gap is invisible. `motion_threshold: 0` restores the old "send everything, call every cycle" behavior.
- `motion_threshold` is not portable across cameras (AGC/IR/compression shift the noise floor). Every analyze logs `motion_score`; to calibrate, run on an empty room with gating off and set the threshold just above the idle noise. For a safety monitor, prefer false positives (lower threshold) and rely on `min_interval_seconds` for cost control.
- `subsampled_stream_maxlen` must cover `max_interval_seconds` at the subsampled frame rate, otherwise catch-up after a long heartbeat truncates the gap; the watcher warns ("does not cover the full gap") when this happens.
