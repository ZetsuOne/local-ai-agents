"""Unit tests for workspace security and critical tools."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from tools import (
    edit_file,
    execute_tool,
    read_file,
    set_workspace,
    sqlite_query,
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
        result = execute_tool("read_file", {"filename": "../outside.txt"})
        assert "Ошибка" in result
        assert "запрещ" in result.lower() or "предел" in result.lower()

    def test_write_file_blocks_parent_path(self, workspace: Path):
        target = workspace.parent / "outside_write.txt"
        result = execute_tool(
            "write_file",
            {"filename": "../outside_write.txt", "content": "pwned"},
        )
        assert "Ошибка" in result
        assert "запрещ" in result.lower() or "предел" in result.lower()
        assert not target.exists()

    def test_read_file_raises_permission_error(self, workspace: Path):
        with pytest.raises(PermissionError):
            read_file("../outside.txt")

    def test_write_file_raises_permission_error(self, workspace: Path):
        with pytest.raises(PermissionError):
            write_file("../outside.txt", "pwned")


# ── Auto-backups & syntax validation ─────────────────────────────────────────

class TestBackupsAndSyntax:
    def test_write_file_creates_backup(self, workspace: Path):
        write_file("note.txt", "v1")
        write_file("note.txt", "v2")

        backup_dir = workspace / ".agent_backup"
        assert backup_dir.is_dir()
        backups = list(backup_dir.iterdir())
        assert len(backups) >= 1
        assert any(b.name.endswith("_note.txt") for b in backups)
        assert (workspace / "note.txt").read_text(encoding="utf-8") == "v2"
        # Original content preserved in at least one backup
        assert any(b.read_text(encoding="utf-8") == "v1" for b in backups)

    def test_edit_file_creates_backup(self, workspace: Path):
        write_file("code.py", "x = 1\n")
        result = edit_file("code.py", "x = 1", "x = 2")
        assert result.startswith("OK:")

        backup_dir = workspace / ".agent_backup"
        assert backup_dir.is_dir()
        backups = [b for b in backup_dir.iterdir() if b.name.endswith("_code.py")]
        assert len(backups) >= 1
        assert any("x = 1" in b.read_text(encoding="utf-8") for b in backups)

    def test_write_file_reports_syntax_error(self, workspace: Path):
        bad = "def broken(\n"
        result = write_file("broken.py", bad)
        assert "SyntaxError" in result
        assert (workspace / "broken.py").read_text(encoding="utf-8") == bad

    def test_edit_file_reports_syntax_error(self, workspace: Path):
        write_file("ok.py", "def ok():\n    return 1\n")
        result = edit_file("ok.py", "return 1", "return (\n")
        assert "SyntaxError" in result


# ── sqlite_query security ────────────────────────────────────────────────────

class TestSqliteQuery:
    @pytest.fixture
    def sample_db(self, workspace: Path) -> str:
        db_path = workspace / "app.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO items (name) VALUES ('alpha'), ('beta')")
            conn.commit()
        finally:
            conn.close()
        return "app.db"

    def test_select_works_readonly(self, sample_db: str):
        result = sqlite_query(sample_db, "SELECT name FROM items ORDER BY id")
        assert "Ошибка" not in result
        data = json.loads(result)
        assert data["count"] == 2
        assert data["rows"][0]["name"] == "alpha"
        assert data["rows"][1]["name"] == "beta"

    def test_rejects_drop(self, sample_db: str):
        result = sqlite_query(sample_db, "DROP TABLE items")
        assert "Ошибка" in result
        assert "SELECT" in result or "PRAGMA" in result or "DROP" in result

    def test_rejects_insert(self, sample_db: str):
        result = sqlite_query(sample_db, "INSERT INTO items (name) VALUES ('gamma')")
        assert "Ошибка" in result

    def test_rejects_update(self, sample_db: str):
        result = sqlite_query(sample_db, "UPDATE items SET name='x' WHERE id=1")
        assert "Ошибка" in result

    def test_rejects_multiple_statements(self, sample_db: str):
        result = sqlite_query(
            sample_db,
            "SELECT name FROM items; DROP TABLE items",
        )
        assert "Ошибка" in result
        assert ";" in result or "цепочк" in result.lower()

    def test_rejects_insert_hidden_in_select(self, sample_db: str):
        # Token blacklist should catch dangerous keywords even inside a SELECT
        result = sqlite_query(
            sample_db,
            "SELECT * FROM items WHERE name = 'INSERT'",
        )
        # 'INSERT' as a string literal token still matches the keyword filter
        assert "Ошибка" in result
        assert "INSERT" in result
