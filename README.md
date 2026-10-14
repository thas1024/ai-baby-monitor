# 👶🧠 AI Baby Monitor (Local Video‑LLM Nanny)

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![DeepWiki](https://img.shields.io/badge/DeepWiki-zeenolife%2Fai--baby--monitor-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/zeenolife/ai-baby-monitor)

**English** | [简体中文](README.zh-CN.md)



> **Your second pair of eyes, powered by local video LLMs. Because, you know... it does take a village.**

The **AI Baby Monitor** watches a video stream (webcam, RTSP camera,  …) and a simple list of safety rules. If a rule is broken it issues a *single* gentle beep so you can quickly glance over and check on your baby.

---

## 📸 Demo
Obviously, I'm not going to put my child in danger just for the demo, so here're videos of:
1. People using smartphones, when rules say you shouldn't ❌
2. Baby being safe and playful with a parent ✅

<table width="100%">
  <tr>
    <td align="center" width="65%">
      <img src="assets/demo_phone.gif" height="300"><br/>
      <sub>📵 "No smartphones" rule – alert fired</sub>
    </td>
    <td align="center" width="35%">
      <img src="assets/demo_baby.gif" height="300"><br/>
      <sub>👶 Baby walking – no alert</sub>
    </td>
  </tr>
</table>

---

## ✨ Features

|                       |                                                                                                                 |
| --------------------- | ----------------------------------------------------------------------------------------------------------------|
| 🛡️ **Private‑first**  | Point it at a local endpoint and nothing leaves your network; or use any cloud endpoint you trust.              |
| ⚡  **Realtime-ish**   | A few requests per minute on a cloud endpoint; ~1 req/s on a local GPU.                                         |
| 🦾 **Video LLM**      | Works with any OpenAI‑compatible vision endpoint — cloud or self‑hosted.                                         |
| 🔔 **One beep alert** | Deliberately minimal & quiet  —  just look when it beeps.                                                       |
| 🖥 **Live dashboard** | Streamlit viewer shows the live stream + LLM reasoning logs in real time.                                       |
| 📝 **Easy rules**     | "The baby shouldn’t climb out of the crib", "Baby should always be accompanied by adult" … just edit YAML.      |
| 🏘️ **Multi-rooms**    | Supports multiple rooms. Just add another YAML with instructions.                                               |

---

## 🚀 Quick start

> **Prerequisites** • Docker + docker‑compose • Python 3.12 with [uv](https://github.com/astral-sh/uv) • An external Redis (host/port/password in .env) • An OpenAI‑compatible vision endpoint (cloud, or local via the optional `vllm` profile)

```bash
# 1 — clone
$ git clone https://github.com/zeenolife/ai-baby-monitor.git && cd ai-baby-monitor

# 2 — copy .env.template into .env and fill in the LLM_* settings
$ cp .env.template .env

# 3 — build & start the services (video streamer, Streamlit viewer, watcher)
$ docker compose up --build -d

# 4 — (optional) run the watcher on the **host** only if you want the audible beep —
#     containers have no audio output. Otherwise it already runs inside compose.
$ ALERT_SOUND_ENABLED=true uv run scripts/run_watcher.py --config-file configs/living_room.yaml
# 5 — open the dashboard 👉 http://localhost:8501. You can also open the dashboard on your phone http://{host_network_ip}:8501
```

> **Heads‑up** The first run builds the docker image and may take a few minutes. With the optional local `vllm` profile it also downloads the model first.

---

## 🛠 Configuration

Add or tweak rooms in `configs/*.yaml`:

```yaml
name: "living_room"

camera:
  uri: "0"            # webcam index or RTSP URI. Use "${CAMERA_URI}" to keep
                      # the URL (with credentials) out of the repo, in .env.

instructions:          # natural‑language rules for the nanny model
  - "The baby shouldn't do anything dangerous."
  - "An adult should be in the room if the baby is awake."
```

* **Multiple rooms**? Add a `stream_to_redis` and a `watcher` service per room in `docker-compose.yml`, and pass the new room config to the streamlit viewer.
* **Multiple cameras per room** (e.g. a close‑up of the baby plus a whole‑room wide angle)? Give the room a `cameras:` list (see `configs/example_multi_camera.yaml`), run **one `stream_to_redis` process per camera** with `--camera NAME`, and keep one `watcher` — a single request merges every angle, each image labeled with its camera, so the model judges all views together. Each camera can have its own `subsample_rate` / `motion_threshold` / frame size.
* **Swap the model / backend**? Set `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL_NAME` in `.env` to point at any OpenAI‑compatible vision endpoint. Set `LLM_API_MODE=responses` to use the OpenAI Responses API instead of the default Chat Completions API. To fall back to local inference, run `docker compose --profile local up -d` and set `LLM_BASE_URL=http://localhost:8000/v1`.
* **Cut cost / requests**? The watcher only calls the model when the picture actually changes, plus a heartbeat. Per‑room knobs (under `camera:`): `motion_metric` (`mean` / `grid_max` / `changed_fraction`), `motion_threshold` (omit for the metric default; `0` disables gating and sends every frame), `min_interval_seconds` (rate floor), `max_interval_seconds` (heartbeat). Changed frames are sent as annotated images (`[frame N @ <time>]`); unchanged runs are sent as a text note only, so a request usually carries far fewer than `num_frames_to_process` images. The alert beep is debounced to the leading edge of a confirmed violation. See "Motion gating" below.

### Motion gating & tuning

`ai_baby_monitor/motion.py` decides *whether* to call the model (compare the newest frame to the last analyzed one; call on change or heartbeat) and *which* frames to send (only the changed ones — unchanged runs become a `[skipped N frames: … (no visible change)]` note). `motion_metric` picks how frames are compared: `mean` (whole‑frame), `grid_max` (busiest cell — catches small localized motion a whole‑frame average misses), or `changed_fraction` (share of changed pixels, noise‑robust). Every analyze logs `motion_score`; `motion_threshold` is camera‑specific — run on an empty room with gating off and set the threshold just above the idle noise. For a safety monitor, prefer false positives (lower threshold) and use `min_interval_seconds` for cost control. `subsampled_stream_maxlen` must cover `max_interval_seconds` at the subsampled frame rate (the watcher warns otherwise). The alert **beep** is debounced (`alerts.py`): it fires once when a violation is confirmed by two consecutive positive verdicts, not on every flip; all verdicts are still logged.

**Night vision (IR).** When the camera switches to infrared, frames are grayscale — normal, not a fault. The prompt tells the model this, night frames are annotated (`· night-vision/IR`), a color↔grayscale switch is noted without re‑sending the whole gray run, and in night mode the monitor is less sensitive: only a `HIGH`‑awareness violation beeps (others are logged). Tune grayscale detection with `night_saturation_threshold` if your camera's "B&W" is slightly tinted.

---

## 🏗 Architecture (high level)
<center><img src="assets/architecture_diagram.png" width="70%"></center>  

---

1. **`stream_to_redis.py`** captures frames and pushes them to Redis (short *realtime* & long *subsampled* frames queues).
2. **`run_watcher.py`** waits for the picture to change (motion gating) or a heartbeat, then sends only the changed frames — each time‑annotated — to the OpenAI‑compatible endpoint, receives structured JSON, writes logs & plays a beep if it receives `should_alert = True`.
3. **Streamlit** live‑updates the latest frame + llm logs; the **Historic logs** tab is a timeline (newest first) where every entry carries a photo of that moment — alerts expand by default, normal entries stay collapsed.

---

## 🛑 Disclaimer

This project is **NOT** a replacement for adult supervision. You should **NEVER** leave your baby alone.

It's meant as an additional guard for situations when you inevitably get distracted for a tiny moment, and your child
is doing something dangerous. Thus just a beep sound as a notification.

It’s an experimental hobby tool — use responsibly and at your own risk.

---

## 📝 License

[MIT](LICENSE) © 2025 @zeenolife


## Credits
Notification sound from [Mixkit](https://mixkit.co/), used under the Mixkit Free Sound Effects License.  
Videos used for demo from [Pexels](https://www.pexels.com/), used under their license
