"""Tests for SQLite session persistence."""
from __future__ import annotations

from pathlib import Path

import pytest

import sessions as session_store


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "test_sessions.db"
    monkeypatch.setattr(session_store, "_DB_PATH", path)
    session_store.init_db(path)
    yield path


class TestSessionStore:
    def test_create_and_get(self, db: Path):
        created = session_store.create_session(
            title="Hello world",
            mode="agent",
            model="test-model",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        )
        assert created["session_id"]
        assert created["title"] == "Hello world"
        assert created["mode"] == "agent"
        assert created["model"] == "test-model"
        assert created["message_count"] == 2
        assert created["created_at"]
        assert created["updated_at"]

        loaded = session_store.get_session(created["session_id"])
        assert loaded is not None
        assert loaded["messages"][1]["content"] == "hi"
        assert db.is_file()

    def test_list_and_update(self, db: Path):
        a = session_store.create_session(title="A", messages=[{"role": "user", "content": "1"}])
        b = session_store.create_session(title="B", messages=[{"role": "user", "content": "2"}])
        listed = session_store.list_sessions()
        assert len(listed) == 2
        # fresh first
        assert listed[0]["session_id"] == b["session_id"]
        assert "messages" not in listed[0]
        assert listed[0]["message_count"] == 1

        updated = session_store.update_session(
            a["session_id"],
            title="A2",
            messages=[
                {"role": "user", "content": "1"},
                {"role": "assistant", "content": "ok"},
            ],
        )
        assert updated is not None
        assert updated["title"] == "A2"
        assert updated["message_count"] == 2

    def test_delete(self, db: Path):
        s = session_store.create_session(title="gone")
        assert session_store.delete_session(s["session_id"]) is True
        assert session_store.get_session(s["session_id"]) is None
        assert session_store.delete_session(s["session_id"]) is False

    def test_title_from_task(self, db: Path):
        assert session_store.title_from_task("short") == "short"
        long = "x" * 100
        t = session_store.title_from_task(long, limit=20)
        assert len(t) <= 20
        assert t.endswith("…")

    def test_checkpoints_and_summary(self, db: Path):
        s = session_store.create_session(title="cps")
        session_store.append_checkpoint(
            s["session_id"],
            {
                "filename": "a.py",
                "action": "write_file",
                "backup": "20260101_000000_a.py",
                "added": 3,
                "removed": 1,
            },
        )
        session_store.append_checkpoint(
            s["session_id"],
            {
                "filename": "a.py",
                "action": "write_file",
                "backup": "20260101_000001_a.py",
                "added": 2,
                "removed": 2,
            },
        )
        loaded = session_store.get_session(s["session_id"])
        assert loaded is not None
        assert len(loaded["checkpoints"]) == 2
        changes = loaded["changes"]
        assert changes["file_count"] == 1
        assert changes["added"] == 5
        assert changes["removed"] == 3
        assert changes["files"][0]["filename"] == "a.py"
