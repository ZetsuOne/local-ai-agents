"""Лёгкое SQLite-хранилище чат/агент-сессий (переживает рестарт UI)."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DB_PATH = Path(__file__).resolve().parent / ".agent_sessions.db"
_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def db_path() -> Path:
    return _DB_PATH


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(path: Path | None = None) -> Path:
    """Создать таблицу sessions при необходимости. Возвращает путь к БД."""
    global _DB_PATH
    if path is not None:
        _DB_PATH = Path(path)
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    mode TEXT NOT NULL DEFAULT 'agent',
                    model TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    messages TEXT NOT NULL DEFAULT '[]',
                    checkpoints TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            # Миграция для старых БД без checkpoints
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "checkpoints" not in cols:
                conn.execute(
                    "ALTER TABLE sessions ADD COLUMN checkpoints TEXT NOT NULL DEFAULT '[]'"
                )
            conn.commit()
        finally:
            conn.close()
    return _DB_PATH


def _parse_checkpoints(raw: str | None) -> list[dict[str, Any]]:
    try:
        data = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def _row_to_dict(row: sqlite3.Row, *, include_messages: bool = True) -> dict[str, Any]:
    keys = set(row.keys())
    data = {
        "session_id": row["session_id"],
        "title": row["title"],
        "mode": row["mode"],
        "model": row["model"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    checkpoints = _parse_checkpoints(row["checkpoints"] if "checkpoints" in keys else "[]")
    data["checkpoints"] = checkpoints
    data["changes"] = summarize_checkpoints(checkpoints)

    if include_messages:
        try:
            msgs = json.loads(row["messages"] or "[]")
        except json.JSONDecodeError:
            msgs = []
        data["messages"] = msgs if isinstance(msgs, list) else []
        data["message_count"] = len(data["messages"])
    else:
        try:
            msgs = json.loads(row["messages"] or "[]")
            data["message_count"] = len(msgs) if isinstance(msgs, list) else 0
        except json.JSONDecodeError:
            data["message_count"] = 0
    return data


def summarize_checkpoints(checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    """Сводка файлов сессии: уникальные файлы + суммарные +/− строки."""
    files: dict[str, dict[str, Any]] = {}
    total_added = 0
    total_removed = 0
    backups: list[str] = []
    for cp in checkpoints:
        fname = (cp.get("filename") or "").strip()
        if not fname:
            continue
        added = int(cp.get("added") or 0)
        removed = int(cp.get("removed") or 0)
        total_added += added
        total_removed += removed
        entry = files.setdefault(
            fname,
            {"filename": fname, "added": 0, "removed": 0, "backups": [], "actions": []},
        )
        entry["added"] += added
        entry["removed"] += removed
        bak = cp.get("backup")
        if bak:
            entry["backups"].append(bak)
            backups.append(bak)
        action = cp.get("action")
        if action:
            entry["actions"].append(action)
    return {
        "files": list(files.values()),
        "file_count": len(files),
        "added": total_added,
        "removed": total_removed,
        "backups": backups,
        "checkpoint_count": len(checkpoints),
    }


def create_session(
    *,
    title: str = "",
    mode: str = "agent",
    model: str = "",
    messages: list[dict[str, Any]] | None = None,
    checkpoints: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Создать новую сессию и вернуть её как dict."""
    init_db()
    sid = (session_id or "").strip() or str(uuid.uuid4())
    now = _utc_now()
    payload = json.dumps(messages or [], ensure_ascii=False, default=str)
    cps = json.dumps(checkpoints or [], ensure_ascii=False, default=str)
    title = (title or "").strip() or "Untitled"
    mode = (mode or "agent").strip() or "agent"
    model = (model or "").strip()
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO sessions
                    (session_id, title, mode, model, created_at, updated_at, messages, checkpoints)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (sid, title, mode, model, now, now, payload, cps),
            )
            conn.commit()
        finally:
            conn.close()
    return get_session(sid) or {
        "session_id": sid,
        "title": title,
        "mode": mode,
        "model": model,
        "created_at": now,
        "updated_at": now,
        "messages": messages or [],
        "message_count": len(messages or []),
        "checkpoints": checkpoints or [],
        "changes": summarize_checkpoints(checkpoints or []),
    }


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """Список сессий (без полного messages), свежие сверху."""
    init_db()
    limit = max(1, min(int(limit or 50), 200))
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT session_id, title, mode, model, created_at, updated_at,
                       messages, checkpoints
                FROM sessions
                ORDER BY updated_at DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()
    return [_row_to_dict(r, include_messages=False) for r in rows]


def get_session(session_id: str) -> dict[str, Any] | None:
    """Полная сессия с messages, или None."""
    init_db()
    sid = (session_id or "").strip()
    if not sid:
        return None
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (sid,),
            ).fetchone()
        finally:
            conn.close()
    if row is None:
        return None
    return _row_to_dict(row, include_messages=True)


def update_session(
    session_id: str,
    *,
    title: str | None = None,
    mode: str | None = None,
    model: str | None = None,
    messages: list[dict[str, Any]] | None = None,
    checkpoints: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Обновить поля сессии. Возвращает обновлённую запись или None."""
    init_db()
    sid = (session_id or "").strip()
    if not sid:
        return None
    current = get_session(sid)
    if current is None:
        return None

    new_title = current["title"] if title is None else (title.strip() or current["title"])
    new_mode = current["mode"] if mode is None else (mode.strip() or current["mode"])
    new_model = current["model"] if model is None else (model or "").strip()
    new_messages = current["messages"] if messages is None else messages
    new_checkpoints = current["checkpoints"] if checkpoints is None else checkpoints
    now = _utc_now()
    payload = json.dumps(new_messages, ensure_ascii=False, default=str)
    cps = json.dumps(new_checkpoints, ensure_ascii=False, default=str)

    with _lock:
        conn = _connect()
        try:
            conn.execute(
                """
                UPDATE sessions
                SET title = ?, mode = ?, model = ?, updated_at = ?,
                    messages = ?, checkpoints = ?
                WHERE session_id = ?
                """,
                (new_title, new_mode, new_model, now, payload, cps, sid),
            )
            conn.commit()
        finally:
            conn.close()
    return get_session(sid)


def append_checkpoint(session_id: str, checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    """Добавить checkpoint (file_change / backup) к сессии."""
    sid = (session_id or "").strip()
    if not sid or not checkpoint:
        return None
    current = get_session(sid)
    if current is None:
        return None
    cps = list(current.get("checkpoints") or [])
    entry = {
        "ts": checkpoint.get("ts") or _utc_now(),
        "filename": checkpoint.get("filename") or "",
        "action": checkpoint.get("action") or "",
        "backup": checkpoint.get("backup"),
        "added": int(checkpoint.get("added") or 0),
        "removed": int(checkpoint.get("removed") or 0),
        "diff": checkpoint.get("diff") or "",
    }
    cps.append(entry)
    return update_session(sid, checkpoints=cps)


def clear_checkpoints(session_id: str) -> dict[str, Any] | None:
    return update_session(session_id, checkpoints=[])


def session_file_diff(session_id: str, filename: str) -> dict[str, Any]:
    """Сравнить первый бэкап сессии для файла с текущим содержимым workspace.

    ``before`` — содержимое первого backup (или ``""`` если файл был создан
    без бэкапа); ``after`` — текущий файл; ``diff`` — unified diff.
    """
    from agent import _safe_path, _unified_diff, read_backup_text

    sid = (session_id or "").strip()
    fname = (filename or "").strip().replace("\\", "/")
    if not sid:
        return {"ok": False, "error": "недопустимый session_id"}
    if not fname:
        return {"ok": False, "error": "недопустимый filename"}
    current = get_session(sid)
    if current is None:
        return {"ok": False, "error": "сессия не найдена"}

    cps = list(current.get("checkpoints") or [])
    first_backup: str | None = None
    for cp in cps:
        if (cp.get("filename") or "").strip().replace("\\", "/") != fname:
            continue
        bak = cp.get("backup")
        if bak and first_backup is None:
            first_backup = bak

    before = ""
    if first_backup:
        bak_data = read_backup_text(first_backup)
        if not bak_data.get("ok"):
            return {
                "ok": False,
                "error": bak_data.get("error") or "не удалось прочитать бэкап",
                "filename": fname,
                "backup": first_backup,
            }
        before = bak_data.get("content") or ""

    try:
        path = _safe_path(fname)
    except (ValueError, PermissionError) as exc:
        return {"ok": False, "error": str(exc), "filename": fname}

    after = ""
    exists = path.is_file()
    if exists:
        try:
            after = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"ok": False, "error": str(exc), "filename": fname}

    diff = _unified_diff(fname, before, after)
    return {
        "ok": True,
        "filename": fname,
        "backup": first_backup,
        "before": before,
        "after": after,
        "diff": diff,
        "exists": exists,
        "created_in_session": first_backup is None and exists,
    }


def revert_session_changes(session_id: str) -> dict[str, Any]:
    """Откатить все файлы сессии к первому бэкапу каждого файла.

    Для каждого filename берётся самый ранний backup из checkpoints и
    восстанавливается через agent.rollback_backup. После успеха checkpoints очищаются.
    """
    from agent import rollback_backup

    sid = (session_id or "").strip()
    if not sid:
        return {"ok": False, "error": "недопустимый session_id"}
    current = get_session(sid)
    if current is None:
        return {"ok": False, "error": "сессия не найдена"}

    cps = list(current.get("checkpoints") or [])
    if not cps:
        return {"ok": True, "reverted": [], "message": "нет изменений для отката"}

    # Первый бэкап по каждому файлу (хронологический порядок checkpoints)
    first_backup: dict[str, str] = {}
    for cp in cps:
        fname = (cp.get("filename") or "").strip()
        bak = cp.get("backup")
        if fname and bak and fname not in first_backup:
            first_backup[fname] = bak

    reverted: list[dict[str, Any]] = []
    errors: list[str] = []
    for fname, bak in first_backup.items():
        result = rollback_backup(bak)
        if result.get("ok"):
            reverted.append({"filename": fname, "backup": bak})
        else:
            errors.append(f"{fname}: {result.get('error') or result}")

    clear_checkpoints(sid)
    ok = not errors
    return {
        "ok": ok,
        "reverted": reverted,
        "errors": errors,
        "message": (
            f"OK: откат {len(reverted)} файл(ов)"
            if ok
            else f"Частичный откат: {len(reverted)} ок, {len(errors)} ошибок"
        ),
    }


def delete_session(session_id: str) -> bool:
    """Удалить сессию. True если удалена."""
    init_db()
    sid = (session_id or "").strip()
    if not sid:
        return False
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (sid,),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


def title_from_task(task: str, limit: int = 60) -> str:
    """Короткий title из первой строки задачи."""
    text = (task or "").strip().replace("\n", " ")
    if not text:
        return "Untitled"
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
