import pytest
from ai_baby_monitor.watcher.base_prompt import get_instructions_prompt
from ai_baby_monitor.watcher import WatcherResponse, AwarenessLevel
from pydantic import ValidationError
from unittest.mock import MagicMock, patch
from ai_baby_monitor.watcher import Watcher
from ai_baby_monitor.stream import Frame
import datetime
from types import SimpleNamespace

import httpx
import cv2
import numpy as np
from openai import BadRequestError

def test_get_instructions_prompt_success():
    instructions = ["Instruction 1", "Instruction 2"]
    prompt = get_instructions_prompt(instructions, language="en")
    assert "* Instruction 1" in prompt
    assert "* Instruction 2" in prompt
    assert "should_alert (boolean)" in prompt
    assert "Always respond in English" in prompt

def test_get_instructions_prompt_empty_list():
    with pytest.raises(ValueError, match="Instructions must be a non-empty list"):
        get_instructions_prompt([])

def test_get_instructions_prompt_single_instruction():
    prompt = get_instructions_prompt(["Single instruction"], language="en")
    assert "* Single instruction" in prompt
    assert "Always respond in English" in prompt

def test_get_instructions_prompt_chinese():
    prompt = get_instructions_prompt(["宝宝不能做危险的动作。"], language="zh")
    assert "宝宝不能做危险的动作。" in prompt
    assert "始终使用中文回答" in prompt

def test_prompts_mention_night_vision():
    assert "night-vision" in get_instructions_prompt(["x"], language="en")
    assert "夜视" in get_instructions_prompt(["x"], language="zh")

def test_watcher_response_valid():
    data = {
        "should_alert": True,
        "reasoning": "Baby is crying",
        "recommended_awareness_level": "HIGH"
    }
    response = WatcherResponse(**data)
    assert response.should_alert is True
    assert response.reasoning == "Baby is crying"
    assert response.recommended_awareness_level == AwarenessLevel.HIGH

def test_watcher_response_invalid_awareness_level():
    data = {
        "should_alert": False,
        "reasoning": "Baby is sleeping",
        "recommended_awareness_level": "VERY_LOW"  # Invalid level
    }
    with pytest.raises(ValidationError):
        WatcherResponse(**data)

def test_watcher_response_missing_field():
    data = {
        "should_alert": True,
        "reasoning": "Baby is awake"
        # missing recommended_awareness_level
    }
    with pytest.raises(ValidationError):
        WatcherResponse(**data) 

# Tests for Watcher class
@patch('ai_baby_monitor.watcher.watcher.get_instructions_prompt')
def test_watcher_init(mock_get_instructions_prompt):
    mock_get_instructions_prompt.return_value = "Mocked Instructions Prompt"
    instructions = ["Don't cry"]

    watcher = Watcher(
        instructions=instructions,
        model_name="test_model",
        base_url="http://test_host:1234/v1",
        api_key="test_key",
    )

    mock_get_instructions_prompt.assert_called_once_with(instructions)
    assert watcher.instructions_prompt == "Mocked Instructions Prompt"
    # json_schema is the strict (OpenAI compatible) variant of the model schema.
    assert watcher.json_schema["additionalProperties"] is False
    assert set(watcher.json_schema["required"]) == {
        "should_alert",
        "reasoning",
        "recommended_awareness_level",
    }
    assert watcher.model_name == "test_model"
    assert watcher.base_url == "http://test_host:1234/v1"
    assert watcher.api_mode == "chat"
    assert str(watcher.client.base_url) == "http://test_host:1234/v1/"
    assert watcher.client.api_key == "test_key"


def test_watcher_init_invalid_api_mode():
    with pytest.raises(ValueError, match="api_mode must be 'chat' or 'responses'"):
        Watcher(instructions=["Test"], api_mode="rpc")


def test_make_strict_recurses_into_nested_objects():
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": "string"}},
            }
        },
    }
    strict = Watcher._make_strict(schema)
    assert strict["additionalProperties"] is False
    assert strict["required"] == ["outer"]
    assert strict["properties"]["outer"]["additionalProperties"] is False
    assert strict["properties"]["outer"]["required"] == ["inner"]

# Test for Watcher._calculate_fps
def test_watcher_calculate_fps_valid():
    watcher = Watcher(instructions=["Test"])
    now = datetime.datetime.now()
    frames = [
        Frame(frame_data=[], timestamp=now, frame_idx=0),
        Frame(frame_data=[], timestamp=now + datetime.timedelta(seconds=1), frame_idx=1),
        Frame(frame_data=[], timestamp=now + datetime.timedelta(seconds=2), frame_idx=2), # 3 frames, 2 seconds diff -> 1 FPS
    ]
    assert watcher._calculate_fps(frames) == 1

def test_watcher_calculate_fps_default_fps_param():
    watcher = Watcher(instructions=["Test"])
    now = datetime.datetime.now()
    frames = [
        Frame(frame_data=[], timestamp=now, frame_idx=0),
        Frame(frame_data=[], timestamp=now + datetime.timedelta(seconds=1), frame_idx=1),
        Frame(frame_data=[], timestamp=now + datetime.timedelta(seconds=2), frame_idx=2),
    ]
    assert watcher._calculate_fps(frames, default_fps=5) == 1 # Should still calculate correctly


def test_watcher_calculate_fps_too_few_frames():
    watcher = Watcher(instructions=["Test"])
    frames = [Frame(frame_data=[], timestamp=datetime.datetime.now(), frame_idx=0)]
    assert watcher._calculate_fps(frames, default_fps=3) == 3

def test_watcher_calculate_fps_zero_time_diff():
    watcher = Watcher(instructions=["Test"])
    now = datetime.datetime.now()
    frames = [
        Frame(frame_data=[], timestamp=now, frame_idx=0),
        Frame(frame_data=[], timestamp=now, frame_idx=1)
    ]
    assert watcher._calculate_fps(frames, default_fps=4) == 4
    
def test_watcher_calculate_fps_negative_time_diff(): # Should also use default
    watcher = Watcher(instructions=["Test"])
    now = datetime.datetime.now()
    frames = [
        Frame(frame_data=[], timestamp=now + datetime.timedelta(seconds=1), frame_idx=0),
        Frame(frame_data=[], timestamp=now, frame_idx=1)
    ]
    assert watcher._calculate_fps(frames, default_fps=4) == 4


def test_watcher_calculate_fps_unreasonable_low_fps():
    watcher = Watcher(instructions=["Test"])
    now = datetime.datetime.now()
    frames = [
        Frame(frame_data=[], timestamp=now, frame_idx=0),
        Frame(frame_data=[], timestamp=now + datetime.timedelta(seconds=1000), frame_idx=1), # (2-1)/1000 = 0.001 FPS
    ]
    assert watcher._calculate_fps(frames, default_fps=2) == 2

def test_watcher_calculate_fps_unreasonable_high_fps():
    watcher = Watcher(instructions=["Test"])
    now = datetime.datetime.now()
    frames = [
        Frame(frame_data=[], timestamp=now, frame_idx=0),
        Frame(frame_data=[], timestamp=now + datetime.timedelta(microseconds=100), frame_idx=1), # (2-1)/0.0001 = 10000 FPS
    ]
    assert watcher._calculate_fps(frames, default_fps=2) == 2
    
@patch('ai_baby_monitor.watcher.watcher.logger')
def test_watcher_calculate_fps_exception(mock_logger):
    watcher = Watcher(instructions=["Test"])
    frames = [
        Frame(frame_data=[], timestamp="not a datetime", frame_idx=0), # This will cause an error
        Frame(frame_data=[], timestamp="also not a datetime", frame_idx=1)
    ]
    assert watcher._calculate_fps(frames, default_fps=5) == 5
    called_with_correct_message = False
    for call_args in mock_logger.warning.call_args_list:
        args, kwargs = call_args
        if args[0] == "FPS calculation error" and isinstance(kwargs.get('error'), TypeError):
            called_with_correct_message = True
            break
    assert called_with_correct_message, "logger.warning not called with expected FPS calculation error"


# --- process_frames / structured-output negotiation ---

VALID_JSON = (
    '{"should_alert": true, "reasoning": "baby on couch", '
    '"recommended_awareness_level": "HIGH"}'
)


def _bad_request_error() -> BadRequestError:
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError("unsupported", response=response, body=None)


def _chat_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _sample_frame() -> Frame:
    return Frame(
        frame_data=np.array([1, 2, 3], dtype=np.uint8),
        timestamp=datetime.datetime.now(),
        frame_idx=0,
    )


def test_process_frames_chat_completions_uses_json_schema():
    watcher = Watcher(instructions=["Test"])
    watcher.client = MagicMock()
    watcher.client.chat.completions.create.return_value = _chat_response(VALID_JSON)

    result = watcher.process_frames([_sample_frame()])

    assert result["success"] is True
    assert result["should_alert"] is True
    assert result["recommended_awareness_level"] == AwarenessLevel.HIGH
    assert result["raw_response"] == VALID_JSON

    kwargs = watcher.client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    # Frames are sent as individual image parts, each preceded by a time annotation.
    user_content = kwargs["messages"][1]["content"]
    image_parts = [p for p in user_content if p["type"] == "image_url"]
    assert image_parts
    first_image_idx = next(i for i, p in enumerate(user_content) if p["type"] == "image_url")
    assert user_content[first_image_idx - 1]["type"] == "text"


def test_process_frames_falls_back_when_json_schema_rejected():
    watcher = Watcher(instructions=["Test"])
    watcher.client = MagicMock()

    def side_effect(**kwargs):
        if kwargs.get("response_format", {}).get("type") == "json_schema":
            raise _bad_request_error()
        return _chat_response(VALID_JSON)

    watcher.client.chat.completions.create.side_effect = side_effect

    result = watcher.process_frames([_sample_frame()])

    assert result["success"] is True
    # The working option is promoted so the fallback only costs one request.
    assert watcher._response_format_options[0] == "json_object"


def test_process_frames_responses_mode():
    watcher = Watcher(instructions=["Test"], api_mode="responses")
    watcher.client = MagicMock()
    # Responses mode streams; the text is reassembled from delta events.
    watcher.client.responses.create.return_value = [
        SimpleNamespace(type="response.output_text.delta", delta=VALID_JSON[:20]),
        SimpleNamespace(type="response.output_text.delta", delta=VALID_JSON[20:]),
    ]

    result = watcher.process_frames([_sample_frame()])

    assert result["success"] is True
    assert result["raw_response"] == VALID_JSON
    kwargs = watcher.client.responses.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["text"]["format"]["type"] == "json_schema"
    user_content = kwargs["input"][1]["content"]
    assert any(p["type"] == "input_image" for p in user_content)
    first_image_idx = next(i for i, p in enumerate(user_content) if p["type"] == "input_image")
    assert user_content[first_image_idx - 1]["type"] == "input_text"
    # Chat completions endpoint was not touched.
    watcher.client.chat.completions.create.assert_not_called()


def test_process_frames_no_frames():
    watcher = Watcher(instructions=["Test"])
    result = watcher.process_frames([])
    assert result["success"] is False
    assert "No frames provided" in result["error"]


# --- motion compression + time annotations ---

def _real_jpeg(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = (rng.random((90, 160, 3)) * 255).astype("uint8")
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer


def _frame_with(jpeg, second: int = 0) -> Frame:
    return Frame(
        frame_data=jpeg,
        timestamp=datetime.datetime(2026, 1, 1, 12, 0, second),
        frame_idx=second,
    )


def test_process_frames_compresses_unchanged_frames():
    watcher = Watcher(instructions=["Test"], motion_threshold=3.0)
    watcher.client = MagicMock()
    watcher.client.chat.completions.create.return_value = _chat_response(VALID_JSON)

    jpeg = _real_jpeg(1)
    frames = [_frame_with(jpeg.copy(), i) for i in range(4)]

    result = watcher.process_frames(frames)

    assert result["success"] is True
    assert result["images_sent"] == 1
    assert result["frames_skipped"] == 3

    content = watcher.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert len([p for p in content if p["type"] == "image_url"]) == 1
    texts = " ".join(p["text"] for p in content if p["type"] == "text")
    assert "3" in texts  # skip note mentions the 3 collapsed frames


def test_process_frames_threshold_zero_sends_every_frame():
    watcher = Watcher(instructions=["Test"], motion_threshold=0.0)
    watcher.client = MagicMock()
    watcher.client.chat.completions.create.return_value = _chat_response(VALID_JSON)

    jpeg = _real_jpeg(1)
    frames = [_frame_with(jpeg.copy(), i) for i in range(4)]

    result = watcher.process_frames(frames)

    assert result["images_sent"] == 4
    assert result["frames_skipped"] == 0


def test_frame_annotation_is_prepended_to_image():
    watcher = Watcher(instructions=["Test"], motion_threshold=0.0)
    watcher.client = MagicMock()
    watcher.client.chat.completions.create.return_value = _chat_response(VALID_JSON)

    watcher.process_frames([_frame_with(_real_jpeg(1))])

    content = watcher.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    # first part is the annotation, second is the image
    assert content[0]["type"] == "text"
    assert "2026-01-01" in content[0]["text"]
    assert content[1]["type"] == "image_url"


def test_skipped_annotation_chinese_by_default(monkeypatch):
    monkeypatch.delenv("APP_LANG", raising=False)
    from ai_baby_monitor.watcher import format_skipped_annotation

    text = format_skipped_annotation(
        3, datetime.datetime(2026, 1, 1, 12, 0, 0), datetime.datetime(2026, 1, 1, 12, 0, 1)
    )
    assert "跳过 3 帧" in text


def test_process_frames_reports_night_mode():
    watcher = Watcher(instructions=["Test"], motion_threshold=0.0)
    watcher.client = MagicMock()
    watcher.client.chat.completions.create.return_value = _chat_response(VALID_JSON)

    gray = cv2.imencode(".jpg", (np.random.default_rng(0).random((90, 160)) * 255).astype("uint8"))[1]
    result = watcher.process_frames([_frame_with(gray)])
    assert result["night_mode"] is True

    content = watcher.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    annotation = content[0]["text"]
    assert "夜视" in annotation or "night-vision" in annotation


# --- multi-camera (process_room) ---

def test_process_room_merges_cameras_with_labels():
    from ai_baby_monitor.watcher import CameraWindow

    watcher = Watcher(instructions=["Test"], motion_threshold=0.0)
    watcher.client = MagicMock()
    watcher.client.chat.completions.create.return_value = _chat_response(VALID_JSON)

    windows = [
        CameraWindow(frames=[_frame_with(_real_jpeg(1))], name="closeup", motion_threshold=0.0),
        CameraWindow(frames=[_frame_with(_real_jpeg(2))], name="wide", motion_threshold=0.0),
    ]
    result = watcher.process_room(windows)

    assert result["success"] is True
    assert result["images_sent"] == 2  # one changed frame per camera

    content = watcher.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    texts = " ".join(p["text"] for p in content if p["type"] == "text")
    assert "closeup" in texts
    assert "wide" in texts
    image_parts = [p for p in content if p["type"] == "image_url"]
    assert len(image_parts) == 2


def test_process_room_ignores_empty_windows():
    from ai_baby_monitor.watcher import CameraWindow

    watcher = Watcher(instructions=["Test"], motion_threshold=0.0)
    watcher.client = MagicMock()
    watcher.client.chat.completions.create.return_value = _chat_response(VALID_JSON)

    result = watcher.process_room(
        [
            CameraWindow(frames=[], name="closeup"),
            CameraWindow(frames=[_frame_with(_real_jpeg(1))], name="wide", motion_threshold=0.0),
        ]
    )
    assert result["success"] is True
    content = watcher.client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    texts = " ".join(p["text"] for p in content if p["type"] == "text")
    assert "closeup" not in texts  # empty camera contributes nothing
    assert "wide" in texts


def test_process_room_all_empty_returns_error():
    from ai_baby_monitor.watcher import CameraWindow

    watcher = Watcher(instructions=["Test"])
    result = watcher.process_room([CameraWindow(frames=[], name="a")])
    assert result["success"] is False
