# 👶🧠 AI 婴儿监控（本地视频大模型「保姆」）

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![DeepWiki](https://img.shields.io/badge/DeepWiki-zeenolife%2Fai--baby--monitor-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/zeenolife/ai-baby-monitor)

[English](README.md) | **简体中文**

> **你的第二双眼睛，由本地视频大模型驱动。毕竟，养个娃确实「需要一个村的人」。**

**AI 婴儿监控** 会盯着一路视频流（摄像头、RTSP 摄像机……）和一份简单的安全规则清单。一旦规则被违反，它就发出**一声**轻柔的提示音，让你赶紧瞄一眼宝宝的情况。

---

## 📸 演示

当然，我不会为了演示把自家孩子置于危险之中，所以这里是两段视频：

1. 有人在使用手机，而规则说这不允许 ❌
2. 宝宝在大人的陪伴下安全地玩耍 ✅

<table width="100%">
  <tr>
    <td align="center" width="65%">
      <img src="assets/demo_phone.gif" height="300"><br/>
      <sub>📵「禁止使用手机」规则 —— 触发告警</sub>
    </td>
    <td align="center" width="35%">
      <img src="assets/demo_baby.gif" height="300"><br/>
      <sub>👶 宝宝在走动 —— 不告警</sub>
    </td>
  </tr>
</table>

---

## ✨ 功能特性

|                       |                                                                                                                 |
| --------------------- | ----------------------------------------------------------------------------------------------------------------|
| 🛡️ **隐私优先**        | 指向本地端点则数据不出内网；也可以使用你信任的任意云端端点。                                                     |
| ⚡  **准实时**          | 云端端点每分钟数次请求；本地 GPU 约 1 次/秒。                                                                   |
| 🦾 **视频大模型**       | 兼容任意 OpenAI 协议的视觉端点 —— 云端或自建均可。                                                              |
| 🔔 **单声蜂鸣告警**     | 刻意做到极简、安静 —— 响了就瞄一眼。                                                                            |
| 🖥 **实时面板**         | Streamlit 面板实时展示画面 + 大模型的判断日志。                                                                 |
| 📝 **规则易改**         | 「宝宝不能爬出婴儿床」「宝宝醒着必须有大人在场」…… 改 YAML 即可。                                              |
| 🏘️ **多房间**          | 支持多个房间，加一个 YAML 配置即可。                                                                            |

---

## 🚀 快速开始

> **前置条件** • Docker + docker‑compose • Python 3.12 与 [uv](https://github.com/astral-sh/uv) • 一个外部 Redis（在 .env 里填 host/port/password）• 一个 OpenAI 协议的视觉端点（云端，或用可选的 `vllm` profile 本地跑）

```bash
# 1 — 克隆
$ git clone https://github.com/zeenolife/ai-baby-monitor.git && cd ai-baby-monitor

# 2 — 复制 .env.template 为 .env，并填好 LLM_* 等设置
$ cp .env.template .env

# 3 — 构建并启动各服务（视频采集、Streamlit 面板、watcher）
$ docker compose up --build -d

# 4 —（可选）只有想听那声蜂鸣时，才在**宿主机**上跑 watcher ——
#     容器没有音频输出。否则它已经在 compose 里运行了。
$ ALERT_SOUND_ENABLED=true uv run scripts/run_watcher.py --config-file configs/living_room.yaml

# 5 — 打开面板 👉 http://localhost:8501 ，手机上也可访问 http://{宿主机内网IP}:8501
```

> **提示** 首次运行会构建 Docker 镜像，可能需要几分钟。若使用可选的本地 `vllm` profile，还会先下载模型。

---

## 🛠 配置

在 `configs/*.yaml` 中新增或调整房间：

```yaml
name: "living_room"

camera:
  uri: "${CAMERA_URI}"   # 摄像头索引或 RTSP 地址；用 ${CAMERA_URI} 把带密码的地址放在 .env 里

instructions:            # 给「保姆」大模型看的自然语言规则
  - "宝宝不能做任何危险的动作。"
  - "如果宝宝醒着，房间里必须有大人在场。"
```

* **多房间**？编辑 `docker-compose.yml`，为每个房间各加一个 `stream_to_redis` 和一个 `watcher` 服务，并把新的房间配置传给 Streamlit 面板。
* **更换模型 / 后端**？在 `.env` 里设置 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL_NAME`，指向任意 OpenAI 协议的视觉端点。设 `LLM_API_MODE=responses` 走 OpenAI Responses API，默认走 Chat Completions。想回退到本地推理，执行 `docker compose --profile local up -d`，并把 `LLM_BASE_URL=http://localhost:8000/v1`。
* **降低频率 / 省 token**？watcher 只在画面**真正变化**时才调用模型，另加心跳兜底。每个房间可调（放在 `camera:` 下）：`motion_metric`（`mean` / `grid_max` / `changed_fraction`）、`motion_threshold`（省略则用该指标的默认值，`0` 关闭门控、每帧都发）、`min_interval_seconds`（最小间隔）、`max_interval_seconds`（心跳上限）。发生变化的帧会带时间标注发送（`[帧N @ <时间>]`）；未变化的连续帧只发一条文字说明、不带图，因此单次请求的图片数通常远小于 `num_frames_to_process`。告警蜂鸣做了去抖，只在违规被确认时响一次。详见下节。

### 运动门控与调参

`ai_baby_monitor/motion.py` 决定**是否**调用模型（把最新帧与"上次分析过的帧"比较，有变化或到心跳就调）以及**发送哪些帧**（只发变化了的帧，未变化的连续段折叠成一条 `[跳过 N 帧：… （画面无明显变化）]` 说明）。`motion_metric` 选择比较方式：`mean`（整帧均值）、`grid_max`（取最"忙"的网格，能抓到整帧平均会漏掉的局部小动作）、`changed_fraction`（变化像素占比，抗噪声）。每次分析都会记录 `motion_score`；`motion_threshold` **因摄像头而异**——在空房间里关掉门控跑一会儿，把阈值设在空闲噪声之上一点。作为安全监控**宁可误报不可漏报**：拿不准就把阈值调低，用 `min_interval_seconds` 控成本。`subsampled_stream_maxlen` 必须覆盖 `max_interval_seconds` 对应的帧数（否则 watcher 会打 warning）。告警**蜂鸣**做了去抖（`alerts.py`）：连续两次判定都告警才响一次，不会每帧翻转都响；所有判定仍然全部写入日志。
* **界面语言**？设 `APP_LANG=zh`（默认）或 `en`，同时决定界面文案和模型输出的语言。

---

## 🏗 架构（高层）

<center><img src="assets/architecture_diagram.png" width="70%"></center>

---

1. **`stream_to_redis.py`** 采集画面帧并推送到 Redis（短*实时*队列 & 长*抽样*队列）。
2. **`run_watcher.py`** 等画面变化（运动门控）或到心跳后，只把**变化的帧**（带时间标注）发给 OpenAI 协议端点，接收结构化 JSON，写日志，并在收到 `should_alert = True` 时播放蜂鸣。
3. **Streamlit** 实时刷新最新画面 + 大模型日志。

---

## 🛑 免责声明

本项目**并非**成人看护的替代品。**绝不要**让宝宝独处。

它只是在你难免分神的一小会儿、而孩子正在做危险动作时，多一道提醒。因此通知仅仅是一声蜂鸣。

这是一个实验性的业余工具 —— 请自行承担风险、理性使用。

---

## 📝 许可

[MIT](LICENSE) © 2025 @zeenolife


## 鸣谢

通知音效来自 [Mixkit](https://mixkit.co/)，依据 Mixkit Free Sound Effects License 使用。
演示视频来自 [Pexels](https://www.pexels.com/)，依据其许可协议使用。
