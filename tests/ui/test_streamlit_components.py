from unittest.mock import MagicMock, patch

from ai_baby_monitor.ui import streamlit_components as sc


def test_render_logs_handles_verdict_record():
    with patch.object(sc, "st") as mock_st:
        mock_st.chat_message.return_value = MagicMock()
        sc.render_logs(
            [
                {
                    "timestamp": "t",
                    "should_alert": "1",
                    "awareness_level": "HIGH",
                    "reasoning": "baby climbing",
                }
            ]
        )


def test_render_logs_handles_error_record_without_verdict_fields():
    # The watcher writes {timestamp, error} records on failure; rendering them
    # must not raise KeyError on the missing verdict fields.
    with patch.object(sc, "st") as mock_st:
        mock_st.chat_message.return_value = MagicMock()
        sc.render_logs([{"timestamp": "t", "error": "Request timed out."}])
