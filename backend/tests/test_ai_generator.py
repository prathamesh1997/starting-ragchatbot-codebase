"""
Tests for AIGenerator and MockAIGenerator in ai_generator.py.

Covers:
- MockAIGenerator always calls search_course_content tool
- MockAIGenerator formats results correctly
- MockAIGenerator handles empty results gracefully
- AIGenerator delegates to mock when API key is missing
- AIGenerator executes tool when stop_reason == "tool_use"
- AIGenerator makes exactly 2 API calls for the tool-use round-trip
- AIGenerator returns direct answer when stop_reason == "end_turn"
- AIGenerator falls back to mock on AuthenticationError
- Edge case: stop_reason == "tool_use" but tool_manager is None
"""

import pytest
from unittest.mock import MagicMock, patch, call
import anthropic

from ai_generator import AIGenerator, MockAIGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(tool_name: str, tool_input: dict, tool_id: str = "tu_001"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = tool_name
    block.input = tool_input
    block.id = tool_id
    return block


def _make_response(stop_reason: str, content: list):
    resp = MagicMock()
    resp.stop_reason = stop_reason
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# MockAIGenerator
# ---------------------------------------------------------------------------

class TestMockAIGenerator:
    def test_calls_search_course_content_tool(self):
        mgr = MagicMock()
        mgr.execute_tool.return_value = "[Python - Lesson 1]\nPython is great."

        gen = MockAIGenerator()
        gen.generate_response("What is Python?", tool_manager=mgr)

        mgr.execute_tool.assert_called_once_with(
            "search_course_content", query="What is Python?"
        )

    def test_returns_string(self):
        mgr = MagicMock()
        mgr.execute_tool.return_value = "[Python - Lesson 1]\nPython content here."

        gen = MockAIGenerator()
        result = gen.generate_response("What is Python?", tool_manager=mgr)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_tool_manager_returns_error_string(self):
        gen = MockAIGenerator()
        result = gen.generate_response("test", tool_manager=None)
        assert "No search tools available" in result

    def test_empty_results_returns_helpful_message(self):
        mgr = MagicMock()
        mgr.execute_tool.return_value = "No relevant content found."

        gen = MockAIGenerator()
        result = gen.generate_response("quantum physics dark matter", tool_manager=mgr)
        assert "No course content found" in result

    def test_formats_multi_block_results(self):
        mgr = MagicMock()
        mgr.execute_tool.return_value = (
            "[Python Fundamentals - Lesson 1]\nPython is a programming language.\n\n"
            "[Python Fundamentals - Lesson 2]\nVariables store data values."
        )

        gen = MockAIGenerator()
        result = gen.generate_response("Python basics", tool_manager=mgr)
        assert "Python Fundamentals" in result

    def test_offline_mode_notice_is_appended(self):
        mgr = MagicMock()
        mgr.execute_tool.return_value = "[Course]\nSome content."

        gen = MockAIGenerator()
        result = gen.generate_response("question", tool_manager=mgr)
        assert "offline" in result.lower() or "demo mode" in result.lower() or "ANTHROPIC_API_KEY" in result


# ---------------------------------------------------------------------------
# AIGenerator — initialization / mock delegation
# ---------------------------------------------------------------------------

class TestAIGeneratorInit:
    def test_empty_api_key_sets_use_mock_true(self):
        gen = AIGenerator(api_key="", model="claude-sonnet-4-6")
        assert gen._use_mock is True

    def test_dummy_api_key_sets_use_mock_true(self):
        gen = AIGenerator(api_key="sk-ant-dummy-anything", model="claude-sonnet-4-6")
        assert gen._use_mock is True

    def test_empty_api_key_delegates_to_mock(self):
        mgr = MagicMock()
        mgr.execute_tool.return_value = "[Course]\nContent."

        gen = AIGenerator(api_key="", model="claude-sonnet-4-6")
        result = gen.generate_response("test query", tool_manager=mgr)
        assert isinstance(result, str)
        mgr.execute_tool.assert_called_once()


# ---------------------------------------------------------------------------
# AIGenerator — Anthropic client interactions (mocked)
# ---------------------------------------------------------------------------

class TestAIGeneratorWithMockedClient:
    def _make_gen(self):
        """Return an AIGenerator with a fake API key and a mocked Anthropic client."""
        gen = AIGenerator(api_key="sk-ant-real-looking-key", model="claude-sonnet-4-6")
        gen._use_mock = False
        gen.client = MagicMock()
        return gen

    # --- end_turn path ---

    def test_end_turn_returns_direct_text(self):
        gen = self._make_gen()
        gen.client.messages.create.return_value = _make_response(
            "end_turn", [_make_text_block("Here is a direct answer.")]
        )

        result = gen.generate_response("What year did Python launch?")
        assert result == "Here is a direct answer."
        assert gen.client.messages.create.call_count == 1

    def test_end_turn_does_not_call_tool_manager(self):
        gen = self._make_gen()
        gen.client.messages.create.return_value = _make_response(
            "end_turn", [_make_text_block("Direct answer.")]
        )
        mgr = MagicMock()

        gen.generate_response("What year did Python launch?", tool_manager=mgr)
        mgr.execute_tool.assert_not_called()

    # --- tool_use path ---

    def test_tool_use_calls_tool_manager(self):
        gen = self._make_gen()
        tool_block = _make_tool_use_block(
            "search_course_content", {"query": "Python history"}
        )
        gen.client.messages.create.side_effect = [
            _make_response("tool_use", [tool_block]),
            _make_response("end_turn", [_make_text_block("Final answer.")]),
        ]
        mgr = MagicMock()
        mgr.execute_tool.return_value = "[Python - Lesson 1]\nPython content."

        gen.generate_response(
            "Tell me about Python",
            tools=[{"name": "search_course_content"}],
            tool_manager=mgr,
        )

        mgr.execute_tool.assert_called_once_with(
            "search_course_content", query="Python history"
        )

    def test_tool_use_makes_exactly_two_api_calls(self):
        gen = self._make_gen()
        tool_block = _make_tool_use_block(
            "search_course_content", {"query": "Python"}
        )
        gen.client.messages.create.side_effect = [
            _make_response("tool_use", [tool_block]),
            _make_response("end_turn", [_make_text_block("Answer.")]),
        ]
        mgr = MagicMock()
        mgr.execute_tool.return_value = "[Course]\nContent."

        gen.generate_response("Python question", tools=[{}], tool_manager=mgr)
        assert gen.client.messages.create.call_count == 2

    def test_tool_use_returns_final_response_text(self):
        gen = self._make_gen()
        tool_block = _make_tool_use_block(
            "search_course_content", {"query": "Python"}
        )
        gen.client.messages.create.side_effect = [
            _make_response("tool_use", [tool_block]),
            _make_response("end_turn", [_make_text_block("This is the final answer.")]),
        ]
        mgr = MagicMock()
        mgr.execute_tool.return_value = "[Course]\nContent."

        result = gen.generate_response("Python question", tools=[{}], tool_manager=mgr)
        assert result == "This is the final answer."

    def test_tool_use_second_call_does_not_include_tools(self):
        """Second API call (after tool result) must NOT include tools/tool_choice."""
        gen = self._make_gen()
        tool_block = _make_tool_use_block("search_course_content", {"query": "Python"})
        gen.client.messages.create.side_effect = [
            _make_response("tool_use", [tool_block]),
            _make_response("end_turn", [_make_text_block("Done.")]),
        ]
        mgr = MagicMock()
        mgr.execute_tool.return_value = "[Course]\nContent."

        gen.generate_response("Python?", tools=[{"name": "search_course_content"}], tool_manager=mgr)

        second_call_kwargs = gen.client.messages.create.call_args_list[1][1]
        assert "tools" not in second_call_kwargs
        assert "tool_choice" not in second_call_kwargs

    def test_tool_use_with_no_tool_manager_does_not_crash(self):
        """
        If stop_reason == 'tool_use' but tool_manager is None, the condition
        `response.stop_reason == "tool_use" and tool_manager` is False,
        so the code falls through to `return response.content[0].text`.
        A tool_use block has no .text — this should raise AttributeError.
        This test documents the behaviour so it can be fixed.
        """
        gen = self._make_gen()
        tool_block = _make_tool_use_block("search_course_content", {"query": "Python"})
        # tool_block has no .text attribute on a real Anthropic block
        del tool_block.text  # remove the MagicMock auto-attribute

        gen.client.messages.create.return_value = _make_response(
            "tool_use", [tool_block]
        )

        with pytest.raises(AttributeError):
            gen.generate_response("Python?", tool_manager=None)

    # --- auth error fallback ---

    def test_auth_error_falls_back_to_mock(self):
        gen = self._make_gen()
        gen.client.messages.create.side_effect = anthropic.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body={},
        )
        mgr = MagicMock()
        mgr.execute_tool.return_value = "[Course]\nContent."

        result = gen.generate_response("test", tool_manager=mgr)
        assert gen._use_mock is True
        assert isinstance(result, str)

    def test_after_auth_error_subsequent_calls_use_mock(self):
        gen = self._make_gen()
        gen.client.messages.create.side_effect = anthropic.AuthenticationError(
            message="Invalid", response=MagicMock(status_code=401), body={}
        )
        mgr = MagicMock()
        mgr.execute_tool.return_value = "[Course]\nContent."

        gen.generate_response("first call", tool_manager=mgr)
        gen.client.messages.create.reset_mock()

        gen.generate_response("second call", tool_manager=mgr)
        gen.client.messages.create.assert_not_called()

    # --- conversation history ---

    def test_conversation_history_appended_to_system_prompt(self):
        gen = self._make_gen()
        gen.client.messages.create.return_value = _make_response(
            "end_turn", [_make_text_block("Answer.")]
        )

        gen.generate_response("question", conversation_history="User: hi\nAssistant: hello")

        call_kwargs = gen.client.messages.create.call_args[1]
        assert "Previous conversation" in call_kwargs["system"]
        assert "User: hi" in call_kwargs["system"]
