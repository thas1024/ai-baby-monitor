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
import numpy as np
from openai import BadRequestError

def test_get_instructions_prompt_success():
    instructions = ["Instruction 1", "Instruction 2"]
    expected_prompt = """
You are given the following instructions:
* Instruction 1
* Instruction 2

If the instructions are violated, you should alert the user.
You should also recommend the awareness level based on the image.
Please generate a structured response in raw JSON format:
- should_alert (boolean)
- reasoning (string)
- recommended_awareness_level (Enum AwarenessLevel; one of: LOW, MEDIUM, HIGH)
Always respond in English, regardless of the content in the images.
        """
    assert get_instructions_prompt(instructions, language="en") == expected_prompt

def test_get_instructions_prompt_empty_list():
    with pytest.raises(ValueError, match="Instructions must be a non-empty list"):
        get_instructions_prompt([])

def test_get_instructions_prompt_single_instruction():
    instructions = ["Single instruction"]
    expected_prompt = """
You are given the following instructions:
* Single instruction

If the instructions are violated, you should alert the user.
You should also recommend the awareness level based on the image.
Please generate a structured response in raw JSON format:
- should_alert (boolean)
- reasoning (string)
- recommended_awareness_level (Enum AwarenessLevel; one of: LOW, MEDIUM, HIGH)
Always respond in English, regardless of the content in the images.
        """
    assert get_instructions_prompt(instructions, language="en") == expected_prompt

def test_get_instructions_prompt_chinese():
    prompt = get_instructions_prompt(["宝宝不能做危险的动作。"], language="zh")
    assert "宝宝不能做危险的动作。" in prompt
    assert "始终使用中文回答" in prompt

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
    # Frames are sent as individual images, not a vLLM video blob.
    user_content = kwargs["messages"][1]["content"]
    assert user_content[0]["type"] == "image_url"


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
    assert user_content[0]["type"] == "input_image"
    # Chat completions endpoint was not touched.
    watcher.client.chat.completions.create.assert_not_called()


def test_process_frames_no_frames():
    watcher = Watcher(instructions=["Test"])
    result = watcher.process_frames([])
    assert result["success"] is False
    assert "No frames provided" in result["error"]
