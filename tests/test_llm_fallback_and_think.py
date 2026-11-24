"""Тесты fallback tool-call parser, <think> extraction и get_active_model."""

from __future__ import annotations

from types import SimpleNamespace

from llm import (
    FallbackToolCall,
    get_active_model,
    normalize_tool_calls,
    parse_fallback_tool_calls,
    split_think_content,
    strip_tool_call_markup,
)

KNOWN = {
    "write_file",
    "read_file",
    "list_files",
    "run_command",
}


class TestSplitThinkContent:
    def test_extracts_think_block(self):
        reasoning, visible = split_think_content(
            "<think>plan step one</think>\nI will call the tool now."
        )
        assert reasoning == "plan step one"
        assert "I will call the tool now." in visible
        assert "<think>" not in visible

    def test_thinking_alias_and_multiple(self):
        reasoning, visible = split_think_content(
            "<thinking>a</thinking> mid <think>b</think> end"
        )
        assert "a" in reasoning and "b" in reasoning
        assert "mid" in visible and "end" in visible

    def test_empty(self):
        assert split_think_content(None) == ("", "")
        assert split_think_content("") == ("", "")
        assert split_think_content("just text") == ("", "just text")


class TestFallbackToolCallParser:
    def test_tool_call_tag(self):
        content = (
            "I will write the file.\n"
            '<tool_call>{"name": "write_file", "arguments": '
            '{"path": "a.py", "content": "x = 1"}}</tool_call>'
        )
        calls = parse_fallback_tool_calls(content, known_names=KNOWN)
        assert len(calls) == 1
        assert isinstance(calls[0], FallbackToolCall)
        assert calls[0].function.name == "write_file"
        assert "a.py" in calls[0].function.arguments
        assert calls[0].id.startswith("fallback_")

    def test_json_fence(self):
        content = (
            "Calling tool:\n"
            "```json\n"
            '{"name": "read_file", "arguments": {"path": "b.py"}}\n'
            "```"
        )
        calls = parse_fallback_tool_calls(content, known_names=KNOWN)
        assert len(calls) == 1
        assert calls[0].function.name == "read_file"

    def test_raw_json_object(self):
        content = 'Next: {"name": "list_files", "arguments": {}}'
        calls = parse_fallback_tool_calls(content, known_names=KNOWN)
        assert len(calls) == 1
        assert calls[0].function.name == "list_files"

    def test_ignores_unknown_tool_names(self):
        content = '<tool_call>{"name": "rm_rf", "arguments": {}}</tool_call>'
        assert parse_fallback_tool_calls(content, known_names=KNOWN) == []

    def test_normalize_prefers_structured(self):
        structured = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="list_files", arguments="{}"),
        )
        msg = SimpleNamespace(
            tool_calls=[structured],
            content='<tool_call>{"name": "write_file", "arguments": {}}</tool_call>',
        )
        out = normalize_tool_calls(msg, known_names=KNOWN)
        assert len(out) == 1
        assert out[0].function.name == "list_files"

    def test_normalize_fallback_when_empty(self):
        msg = SimpleNamespace(
            tool_calls=[],
            content='<tool_call>{"name": "run_command", "arguments": '
            '{"cmd": "pytest -q"}}</tool_call>',
        )
        out = normalize_tool_calls(msg, known_names=KNOWN)
        assert len(out) == 1
        assert out[0].function.name == "run_command"

    def test_strip_markup(self):
        text = 'Hello <tool_call>{"name": "x"}</tool_call> world'
        assert "tool_call" not in strip_tool_call_markup(text)
        assert "Hello" in strip_tool_call_markup(text)
        assert "world" in strip_tool_call_markup(text)


class TestGetActiveModel:
    def test_prefers_loaded_state(self):
        client = SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: SimpleNamespace(
                    data=[
                        SimpleNamespace(id="text-embedding-nomic", state="loaded"),
                        SimpleNamespace(id="qwen-unloaded", state="not-loaded"),
                        SimpleNamespace(id="gemma-4b", state="loaded"),
                    ]
                )
            )
        )
        assert get_active_model(client) == "gemma-4b"

    def test_ignores_embed_even_if_loaded(self):
        client = SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: SimpleNamespace(
                    data=[
                        SimpleNamespace(id="nomic-embed-text", state="loaded"),
                        SimpleNamespace(id="chat-model", state="loaded"),
                    ]
                )
            )
        )
        assert get_active_model(client) == "chat-model"

    def test_fallback_without_state_attr(self):
        """OpenAI-compatible gateways without LM Studio `state` field."""
        client = SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: SimpleNamespace(
                    data=[
                        SimpleNamespace(id="openai/gpt-4o-mini"),
                        SimpleNamespace(id="text-embedding-3-small"),
                    ]
                )
            )
        )
        assert get_active_model(client) == "openai/gpt-4o-mini"

    def test_fallback_when_nothing_loaded(self):
        client = SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: SimpleNamespace(
                    data=[
                        SimpleNamespace(id="idle-a", state="not-loaded"),
                        SimpleNamespace(id="idle-b", state="not-loaded"),
                    ]
                )
            )
        )
        assert get_active_model(client) == "idle-a"

    def test_raises_when_empty(self):
        client = SimpleNamespace(
            models=SimpleNamespace(list=lambda: SimpleNamespace(data=[]))
        )
        try:
            get_active_model(client)
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "чат-модель" in str(exc).lower() or "model" in str(exc).lower()
