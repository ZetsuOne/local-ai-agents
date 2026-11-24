"""Unit tests for workspace security and the 4 core tools in agent.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import (
    execute_tool,
    list_files,
    read_file,
    run_command,
    set_workspace,
    write_file,
)


@pytest.fixture
def workspace(tmp_path: Path):
    """Isolate each test in a fresh temporary workspace."""
    set_workspace(tmp_path)
    yield tmp_path
    set_workspace(None)


# ── Path traversal ───────────────────────────────────────────────────────────

class TestPathTraversal:
    def test_read_file_blocks_parent_path(self, workspace: Path):
        outside = workspace.parent / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        result = execute_tool("read_file", {"path": "../outside.txt"})
        assert "Ошибка" in result
        assert "запрещ" in result.lower() or "предел" in result.lower() or "вне" in result.lower()

    def test_write_file_blocks_parent_path(self, workspace: Path):
        target = workspace.parent / "outside_write.txt"
        result = execute_tool(
            "write_file",
            {"path": "../outside_write.txt", "content": "pwned"},
        )
        assert "Ошибка" in result
        assert not target.exists()

    def test_list_files_blocks_parent_path(self, workspace: Path):
        result = execute_tool("list_files", {"path": "../"})
        assert "Ошибка" in result

    def test_filename_alias_also_blocked(self, workspace: Path):
        result = execute_tool("read_file", {"filename": "../outside.txt"})
        assert "Ошибка" in result

    def test_read_file_raises_permission_error(self, workspace: Path):
        with pytest.raises(PermissionError):
            read_file("../outside.txt")

    def test_write_file_raises_permission_error(self, workspace: Path):
        with pytest.raises(PermissionError):
            write_file("../outside.txt", "pwned")


# ── Core tools ───────────────────────────────────────────────────────────────

class TestCoreTools:
    def test_write_read_list_roundtrip(self, workspace: Path):
        out = write_file("pkg/hello.py", "print('hi')\n")
        assert out.startswith("OK:")
        listed = list_files("pkg")
        assert "pkg/hello.py" in listed.replace("\\", "/")
        content = read_file("pkg/hello.py")
        assert "=== pkg/hello.py" in content.replace("\\", "/")
        assert "1 | print('hi')" in content

    def test_write_file_creates_backup(self, workspace: Path):
        write_file("note.txt", "v1")
        write_file("note.txt", "v2")

        backup_dir = workspace / ".agent_backup"
        assert backup_dir.is_dir()
        backups = list(backup_dir.iterdir())
        assert len(backups) >= 1
        assert any(b.name.endswith("_note.txt") for b in backups)
        assert (workspace / "note.txt").read_text(encoding="utf-8") == "v2"
        assert any(b.read_text(encoding="utf-8") == "v1" for b in backups)

    def test_write_file_reports_syntax_error(self, workspace: Path):
        bad = "def broken(\n"
        result = write_file("broken.py", bad)
        assert "SyntaxError" in result
        assert (workspace / "broken.py").read_text(encoding="utf-8") == bad

    def test_list_files_skips_hidden_dirs(self, workspace: Path):
        write_file("ok.py", "x = 1\n")
        (workspace / ".venv").mkdir()
        (workspace / ".venv" / "secret.py").write_text("pass\n", encoding="utf-8")
        (workspace / "__pycache__").mkdir()
        listed = list_files()
        assert "ok.py" in listed
        assert ".venv" not in listed
        assert "__pycache__" not in listed

    def test_execute_tool_unknown(self, workspace: Path):
        result = execute_tool("apply_patch", {"path": "x.py", "patch_text": "x"})
        assert "Ошибка" in result
        assert "неизвестный" in result.lower()


# ── run_command allowlist ────────────────────────────────────────────────────

class TestRunCommand:
    def test_allows_pytest(self, workspace: Path):
        write_file("test_ok.py", "def test_truth():\n    assert True\n")
        result = run_command("pytest -q", timeout=60)
        data = json.loads(result)
        assert "allowlist" not in (data.get("error") or "").lower()
        assert "shell" not in (data.get("error") or "").lower()
        assert data.get("returncode") == 0, data
        assert "duration_ms" in data

    def test_allows_python_m_pytest(self, workspace: Path):
        write_file("test_ok2.py", "def test_a():\n    assert 1 == 1\n")
        result = run_command("python -m pytest -q", timeout=60)
        data = json.loads(result)
        assert data["returncode"] == 0

    def test_allows_python_workspace_script(self, workspace: Path):
        write_file("hello.py", "print('ok')\n")
        result = run_command("python hello.py")
        data = json.loads(result)
        assert data.get("returncode") == 0, data
        assert "ok" in (data.get("stdout") or "")

    def test_blocks_shell_chaining(self, workspace: Path):
        for bad in (
            "pytest; rm -rf /",
            "pytest && echo pwned",
            "pytest || echo x",
            "pytest | cat",
            "pytest > out.txt",
            "pytest < in.txt",
            "pytest `id`",
            "pytest $(whoami)",
        ):
            result = run_command(bad)
            data = json.loads(result)
            assert "error" in data, bad
            assert "shell" in data["error"].lower() or "отклон" in data["error"].lower()

    def test_blocks_disallowed_commands(self, workspace: Path):
        for bad in ("rm -rf .", "curl http://x", "bash -c ls"):
            result = run_command(bad)
            data = json.loads(result)
            assert "error" in data
            assert "allowlist" in data["error"].lower()

    def test_allows_git_status(self, workspace: Path):
        result = run_command("git status")
        data = json.loads(result)
        assert "allowlist" not in (data.get("error") or "").lower()
        assert "shell" not in (data.get("error") or "").lower()
