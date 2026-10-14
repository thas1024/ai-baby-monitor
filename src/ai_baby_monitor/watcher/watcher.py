import base64
from dataclasses import dataclass
from enum import Enum

import structlog
from openai import BadRequestError, OpenAI
from pydantic import BaseModel

from ai_baby_monitor.motion import (
    ImageItem,
    ModeTransitionItem,
    build_timeline,
    is_blank_frame,
    is_night_frame,
)
from ai_baby_monitor.stream import Frame
from ai_baby_monitor.watcher import (
    format_camera_header,
    format_frame_annotation,
    format_mode_transition_annotation,
    format_skipped_annotation,
    get_instructions_prompt,
)

logger = structlog.get_logger()


@dataclass
class TextPart:
    """A text segment in the request content."""

    text: str


@dataclass
class ImagePart:
    """A base64-encoded JPEG image segment in the request content."""

    base64_data: str


ContentPart = TextPart | ImagePart


@dataclass
class CameraWindow:
    """One camera's frames for a single analysis, with its own motion settings.

    ``name`` labels the view in the prompt (``None`` for a legacy single camera).
    The motion/night settings default per camera so a multi-camera room can use a
    different threshold per angle.
    """

    frames: list[Frame]
    name: str | None = None
    motion_metric: str = "mean"
    motion_threshold: float | None = None
    night_threshold: float | None = None


class AwarenessLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class WatcherResponse(BaseModel):
    should_alert: bool
    reasoning: str
    recommended_awareness_level: AwarenessLevel


class Watcher:
    def __init__(
        self,
        instructions: list[str],
        model_name: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str = "EMPTY",
        api_mode: str = "chat",
        motion_metric: str = "mean",
        motion_threshold: float | None = None,
        night_saturation_threshold: float | None = None,
    ):
        """
        Initialize the Watcher with instructions and OpenAI-compatible endpoint details.

        Args:
            instructions: List of monitoring instructions to check against frames
            model_name: Name of the vision model to use for inference
            base_url: Base URL of the OpenAI-compatible endpoint (cloud or local)
            api_key: API key for the endpoint ("EMPTY" works for servers without auth)
            api_mode: Which OpenAI protocol to speak. "chat" for Chat Completions
                (``/v1/chat/completions``), "responses" for the Responses API
                (``/v1/responses``). Chat Completions is near-universally supported
                by compatible endpoints, so it is the default.
            motion_threshold: Mean pixel-difference above which two frames count as
                changed. ``None`` uses the metric's default; ``0`` disables
                intra-window compression (every frame is sent as an image).
            motion_metric: How frame difference is scored — "mean" (whole-frame),
                "grid_max" (busiest cell, sensitive to small motion), or
                "changed_fraction" (share of changed pixels). See ``motion.py``.
            night_saturation_threshold: Mean HSV saturation at or below which a
                frame is treated as night-vision/greyscale. None uses the default.
        """
        if api_mode not in ("chat", "responses"):
            raise ValueError(
                f"api_mode must be 'chat' or 'responses', got {api_mode!r}"
            )

        self.instructions_prompt = get_instructions_prompt(instructions)
        self.json_schema = self._make_strict(WatcherResponse.model_json_schema())
        self.motion_metric = motion_metric
        self.motion_threshold = motion_threshold
        self.night_saturation_threshold = night_saturation_threshold

        self.model_name = model_name
        self.base_url = base_url
        self.api_mode = api_mode

        # Initialize OpenAI client for any OpenAI-compatible endpoint.
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # Structured-output options, most strict first, expressed mode-agnostically
        # as the response_format "type". Compatible endpoints differ in what they
        # support, so we negotiate lazily on the first request and cache the winner
        # by moving it to the front.
        self._response_format_options: list[str | None] = [
            "json_schema",
            "json_object",
            None,
        ]

        logger.info(
            "Initialized Watcher",
            model_name=model_name,
            base_url=base_url,
            api_mode=api_mode,
            motion_metric=motion_metric,
            motion_threshold=motion_threshold,
            instructions=instructions,
        )

    @staticmethod
    def _make_strict(schema: dict) -> dict:
        """Make a Pydantic JSON schema OpenAI-Structured-Outputs compatible.

        OpenAI strict mode requires every object to set ``additionalProperties:
        false`` and to list all its properties in ``required``. Pydantic omits
        both by default, so add them recursively.
        """

        def walk(node) -> None:
            if isinstance(node, dict):
                if node.get("type") == "object" and "properties" in node:
                    node["additionalProperties"] = False
                    node["required"] = list(node["properties"].keys())
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(schema)
        return schema

    def _build_content(self, windows: list[CameraWindow]) -> list[ContentPart]:
        """Turn camera windows into request content parts.

        ``build_timeline`` compresses near-identical frames into time-annotated
        text notes so only changed frames are sent as images. Each camera's block
        is preceded by a header naming the view, and every frame annotation is
        labeled with the camera name too (models sometimes ignore a lone header).
        """
        parts: list[ContentPart] = []
        for window in windows:
            if window.name:
                parts.append(TextPart(format_camera_header(window.name)))

            for item in build_timeline(
                window.frames,
                window.motion_threshold,
                window.motion_metric,
                window.night_threshold,
            ):
                if isinstance(item, ImageItem):
                    parts.append(
                        TextPart(
                            format_frame_annotation(
                                item.index,
                                item.frame.timestamp,
                                night=item.night,
                                camera=window.name,
                            )
                        )
                    )
                    parts.append(
                        ImagePart(
                            base64.b64encode(bytes(item.frame.frame_data)).decode("utf-8")
                        )
                    )
                elif isinstance(item, ModeTransitionItem):
                    parts.append(
                        TextPart(
                            format_mode_transition_annotation(
                                item.to_night, item.timestamp, camera=window.name
                            )
                        )
                    )
                else:
                    parts.append(
                        TextPart(
                            format_skipped_annotation(
                                item.count,
                                item.start_ts,
                                item.end_ts,
                                camera=window.name,
                            )
                        )
                    )
        return parts

    def _calculate_fps(self, frames: list[Frame], default_fps: int = 2) -> int:
        """Calculate FPS from frame timestamps, defaulting to 2 if calculation fails.

        Retained as a utility; the request path no longer needs it now that frames
        are sent as individual images rather than a vLLM video blob.
        """
        if len(frames) < 2:
            logger.warning("Too few frames to calculate FPS, using default of 2")
            return default_fps

        try:
            time_diff = (frames[-1].timestamp - frames[0].timestamp).total_seconds()
            if time_diff <= 0:
                return default_fps

            fps = round((len(frames) - 1) / time_diff)

            # Return default if calculated FPS is unreasonable
            return default_fps if fps < 0.02 or fps > 60 else fps

        except Exception as e:
            logger.warning("FPS calculation error", error=e)
            return default_fps

    def _build_response_format(self, kind: str) -> dict:
        """Translate a mode-agnostic format kind into the selected API's payload."""
        if self.api_mode == "responses":
            if kind == "json_schema":
                return {
                    "type": "json_schema",
                    "name": "WatcherResponse",
                    "schema": self.json_schema,
                    "strict": True,
                }
            return {"type": "json_object"}

        # chat completions
        if kind == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "WatcherResponse",
                    "schema": self.json_schema,
                    "strict": True,
                },
            }
        return {"type": "json_object"}

    def _request(self, parts: list[ContentPart], fmt_kind: str | None) -> str:
        """Issue one request using the configured protocol; return the text output.

        Content parts are protocol-neutral; this method maps them to the selected
        API's content schema and appends the instructions text last.
        """
        system_prompt = "You are a helpful assistant and baby sitter."

        if self.api_mode == "responses":
            content: list[dict] = []
            for part in parts:
                if isinstance(part, TextPart):
                    content.append({"type": "input_text", "text": part.text})
                else:
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{part.base64_data}",
                        }
                    )
            content.append({"type": "input_text", "text": self.instructions_prompt})
            kwargs: dict = {
                "model": self.model_name,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                "temperature": 0.1,
                "max_output_tokens": 512,
                # Some Responses endpoints (incl. gateway/proxy setups) reject
                # non-streaming requests with "Stream must be set to true", so
                # always stream and reassemble the text from the deltas.
                "stream": True,
            }
            if fmt_kind is not None:
                kwargs["text"] = {"format": self._build_response_format(fmt_kind)}
            return self._consume_response_stream(self.client.responses.create(**kwargs))

        # chat completions: annotated images, then the instructions.
        content = []
        for part in parts:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            else:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{part.base64_data}"
                        },
                    }
                )
        content.append({"type": "text", "text": self.instructions_prompt})
        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }
        if fmt_kind is not None:
            kwargs["response_format"] = self._build_response_format(fmt_kind)
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    @staticmethod
    def _consume_response_stream(stream) -> str:
        """Concatenate the text deltas from a Responses API event stream."""
        parts: list[str] = []
        for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                parts.append(event.delta)
            elif event_type in ("response.failed", "error"):
                raise RuntimeError(f"Responses stream error: {event}")
        return "".join(parts)

    def _create_completion(self, parts: list[ContentPart]) -> str:
        """Call the endpoint, negotiating structured-output support.

        Endpoints vary in how they enforce JSON: OpenAI Structured Outputs
        (``json_schema``), plain JSON mode (``json_object``), or neither. Options
        are tried in ``_response_format_options`` order; the first that works is
        moved to the front so the fallback only costs one request. Returns the raw
        text content of the response.
        """
        last_error: Exception | None = None

        for fmt_kind in self._response_format_options:
            try:
                text = self._request(parts, fmt_kind)
            except BadRequestError as e:
                logger.warning(
                    "Endpoint rejected response_format, trying next option",
                    response_format=fmt_kind,
                    error=str(e),
                )
                last_error = e
                continue

            # Remember the working option by moving it to the front.
            self._response_format_options.remove(fmt_kind)
            self._response_format_options.insert(0, fmt_kind)
            return text

        raise last_error

    def process_frames(self, frames: list[Frame]) -> dict[str, str | bool]:
        """Process a single camera's frames (single-camera wrapper over process_room).

        Returns:
            dict containing:
                - success (bool): Whether processing completed successfully
                - should_alert (bool): If an alert should be triggered
                - reasoning (str): Explanation for the decision
                - recommended_awareness_level (str): Suggested monitoring level. Should be one of: LOW, MEDIUM, HIGH
                - raw_response (str): Original model response
                - images_sent / frames_skipped (int): compression accounting
                - night_mode (bool): whether the last frame is night-vision
                - error (str, optional): Error message if processing failed
        """
        return self.process_room(
            [
                CameraWindow(
                    frames=frames,
                    name=None,
                    motion_metric=self.motion_metric,
                    motion_threshold=self.motion_threshold,
                    night_threshold=self.night_saturation_threshold,
                )
            ]
        )

    def process_room(self, windows: list[CameraWindow]) -> dict[str, str | bool]:
        """Analyze one or more cameras in a single merged LLM request.

        Each window's frames are annotated with the camera name so the model sees
        all angles together. An empty window is ignored; if no window has frames
        the call is skipped with an error result.
        """
        windows = [w for w in windows if w.frames]
        if not windows:
            logger.warning("No frames provided for processing")
            return {"success": False, "error": "No frames provided"}

        # Drop cameras whose current frame is flat/blank (dead feed) — analyzing a
        # static gray frame is useless and yields false "can't see anything" verdicts.
        live_windows = [
            w for w in windows if not is_blank_frame(w.frames[-1].frame_data)
        ]
        dead_cameras = [
            w.name for w in windows if is_blank_frame(w.frames[-1].frame_data)
        ]
        if not live_windows:
            logger.warning("All cameras are blank (no signal)", cameras=dead_cameras)
            return {
                "success": False,
                "error": "no live video: all cameras returned blank/flat frames "
                "(stream likely failed)",
            }
        if dead_cameras:
            logger.warning("Skipping blank cameras", cameras=dead_cameras)
        windows = live_windows

        try:
            parts = self._build_content(windows)
            images_sent = sum(1 for part in parts if isinstance(part, ImagePart))
            total_frames = sum(len(window.frames) for window in windows)
            raw_response = self._create_completion(parts)

            parsed_response = WatcherResponse.model_validate_json(raw_response)

            night_mode = all(
                is_night_frame(window.frames[-1].frame_data, window.night_threshold)
                for window in windows
            )

            return {
                "success": True,
                "should_alert": parsed_response.should_alert,
                "reasoning": parsed_response.reasoning,
                "recommended_awareness_level": parsed_response.recommended_awareness_level,
                "raw_response": raw_response,
                "images_sent": images_sent,
                "frames_skipped": total_frames - images_sent,
                "night_mode": night_mode,
            }

        except Exception as e:
            logger.error("Error processing frames", error=e)
            return {
                "success": False,
                "error": str(e),
            }
