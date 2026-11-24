"""Tests for @-mention resolution and working-memory compaction."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent import resolve_mention_token, resolve_mentions, set_workspace, write_file
from llm import WORKING_MEMORY_MARKER, compact_messages


@pytest.fixture
def workspace(tmp_path: Path):
    set_workspace(tmp_path)
    yield tmp_path
    set_workspace(None)


class TestMentionResolution:
    def test_resolves_file_mention(self, workspace: Path):
        write_file("app/main.py", "print(1)\nprint(2)\n")
        out = resolve_mentions("Please review @app/main.py carefully")
        assert "--- injected @-context ---" in out
        assert "=== @app/main.py ===" in out
        assert "1 | print(1)" in out
        assert "2 | print(2)" in out
        assert "Please review @app/main.py carefully" in out

    def test_dedupes_repeated_mentions(self, workspace: Path):
        write_file("a.py", "x = 1\n")
        out = resolve_mentions("See @a.py and again @a.py please")
        assert out.count("=== @a.py ===") == 1

    def test_missing_file(self, workspace: Path):
        out = resolve_mention_token("nope.py")
        assert "не найден" in out.lower() or "ошибка" in out.lower()

    def test_no_at_passthrough(self, workspace: Path):
        assert resolve_mentions("plain task") == "plain task"


class TestWorkingMemoryCompactor:
    def _msgs_with_tools(self, n_tools: int) -> list[dict]:
        msgs: list[dict] = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "fix bugs"},
        ]
        for i in range(n_tools):
            tid = f"call_{i}"
            name = "write_file" if i % 3 == 0 else ("run_command" if i % 3 == 1 else "read_file")
            if name == "write_file":
                args = f'{{"path": "f{i}.py", "content": "x={i}"}}'
                result = f"OK: записано → f{i}.py" + ("Y" * 400)
            elif name == "run_command":
                args = '{"cmd": "pytest -q"}'
                result = (
                    '{"returncode": 0, "stdout": "2 passed", "stderr": "", '
                    f'"cmd": "pytest -q", "duration_ms": {i}}}'
                )
            else:
                args = f'{{"path": "f0.py"}}'
                result = ("line content " * 80) + f" tool-{i}"
            msgs.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tid,
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                    ],
                }
            )
            msgs.append({"role": "tool", "tool_call_id": tid, "content": result})
        return msgs

    def test_keeps_last_four_full_and_inserts_summary(self):
        msgs = self._msgs_with_tools(8)
        out = compact_messages(msgs, keep_recent=4, limit=250)
        tool_msgs = [m for m in out if m.get("role") == "tool"]
        assert len(tool_msgs) == 8
        long_old = [
            m for m in out
            if m.get("role") == "tool" and "сжато" in (m.get("content") or "")
        ]
        assert long_old, "ожидались сжатые старые tool-выводы"
        systems = [m for m in out if m.get("role") == "system"]
        assert any(
            WORKING_MEMORY_MARKER in (m.get("content") or "") for m in systems
        )
        wm = next(m for m in systems if WORKING_MEMORY_MARKER in m["content"])
        assert "files:" in wm["content"]
        assert "last_test:" in wm["content"]
        assert "PASS" in wm["content"] or "pytest" in wm["content"].lower()

    def test_no_compact_when_few_tools(self):
        msgs = self._msgs_with_tools(3)
        out = compact_messages(msgs, keep_recent=4, limit=250)
        assert out == msgs

    def test_tracks_write_file_paths(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "edit"},
        ]
        for i in range(5):
            tid = f"t{i}"
            msgs.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tid,
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": f'{{"path": "mod{i}.py", "content": "x"}}',
                            },
                        }
                    ],
                }
            )
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": "OK: записано → mod" + str(i) + ".py " + ("Z" * 300),
                }
            )
        out = compact_messages(msgs, keep_recent=4, limit=250)
        wm = next(
            m for m in out
            if m.get("role") == "system" and WORKING_MEMORY_MARKER in m.get("content", "")
        )
        assert "mod4.py" in wm["content"]
