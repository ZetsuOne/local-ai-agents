"""Локальный Web-UI на http.server + SSE (ThreadingHTTPServer, без внешних CDN)."""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import events
import sessions as session_store
from agent import run_agent
from llm import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_CONTEXT_LIMIT,
    get_active_model,
    make_client,
    resolve_model,
)
from multi_agent import run_pipeline
from agent import (
    apply_cli_workspace,
    get_workspace,
    list_backups,
    list_mention_targets,
    list_workspace_tree,
    rollback_backup,
    set_workspace,
    workspace_has_tests,
)
from workflow import run_workflow

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DEFAULT_PORT = 8080

_run_lock = threading.Lock()
_run_thread: threading.Thread | None = None
_running = False
_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()
_token_totals = {"prompt": 0, "completion": 0}
_last_tok_s: float | None = None
# История messages активного Agent-сеанса (follow-up в Web-UI)
_session_lock = threading.Lock()
_session_messages: list[dict[str, Any]] | None = None
_session_id: str | None = None

session_store.init_db()


def _broadcast(event: dict[str, Any]) -> None:
    payload = json.dumps(event, ensure_ascii=False, default=str)
    with _sse_lock:
        dead: list[queue.Queue] = []
        for q in _sse_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            try:
                _sse_clients.remove(q)
            except ValueError:
                pass


def _on_event(event: dict[str, Any]) -> None:
    global _token_totals, _last_tok_s
    if event.get("type") == "tokens":
        _token_totals["prompt"] += int(event.get("prompt_tokens") or 0)
        _token_totals["completion"] += int(event.get("completion_tokens") or 0)
        tps = event.get("tokens_per_sec")
        if tps is not None:
            try:
                _last_tok_s = float(tps)
            except (TypeError, ValueError):
                pass
        event = {
            **event,
            "total_prompt": _token_totals["prompt"],
            "total_completion": _token_totals["completion"],
            "tokens_per_sec": _last_tok_s if tps is None else tps,
        }
    _broadcast(event)


events.subscribe(_on_event)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _send_json(handler: BaseHTTPRequestHandler, code: int, data: Any) -> None:
    body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _send_file(handler: BaseHTTPRequestHandler, path: Path, content_type: str) -> None:
    if not path.is_file():
        handler.send_error(404, "Not Found")
        return
    data = path.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    handler.wfile.write(data)


def _safe_workspace_file(rel: str) -> Path | None:
    if not rel or ".." in Path(rel).parts:
        return None
    ws = get_workspace()
    candidate = (ws / rel).resolve()
    try:
        candidate.relative_to(ws.resolve())
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    # Не отдаём бэкапы через UI по умолчанию — можно, но дерево их не показывает
    return candidate


def _session_info() -> dict[str, Any]:
    with _session_lock:
        msgs = _session_messages
        count = len(msgs) if msgs else 0
        active = bool(msgs)
        sid = _session_id
    return {
        "active": active,
        "messages": count,
        "session_id": sid,
        "tokens_per_sec": _last_tok_s,
    }


def _reset_session() -> None:
    global _session_messages, _session_id
    with _session_lock:
        _session_messages = None
        _session_id = None


def _persist_agent_session(
    *,
    task: str,
    mode: str,
    model: str,
    messages: list[dict[str, Any]],
) -> str | None:
    """Создать или обновить SQLite-сессию; вернуть session_id."""
    global _session_id
    with _session_lock:
        sid = _session_id
    title = session_store.title_from_task(task)
    try:
        if sid and session_store.get_session(sid):
            session_store.update_session(
                sid, title=title, mode=mode, model=model, messages=messages
            )
            return sid
        created = session_store.create_session(
            title=title, mode=mode, model=model, messages=messages
        )
        sid = created["session_id"]
        with _session_lock:
            _session_id = sid
        return sid
    except Exception as exc:  # noqa: BLE001
        events.emit("error", message=f"session persist: {exc}")
        return sid


def _ensure_session_for_run(*, task: str, mode: str, model: str) -> str | None:
    """Гарантировать session_id до старта, чтобы checkpoints писались в run."""
    global _session_id
    with _session_lock:
        sid = _session_id
    if sid and session_store.get_session(sid):
        return sid
    try:
        created = session_store.create_session(
            title=session_store.title_from_task(task),
            mode=mode,
            model=model,
            messages=[],
        )
        sid = created["session_id"]
        with _session_lock:
            _session_id = sid
        return sid
    except Exception as exc:  # noqa: BLE001
        events.emit("error", message=f"session create: {exc}")
        return None


def _on_file_change_checkpoint(event: dict[str, Any]) -> None:
    """Писать file_change / backup_created в checkpoints активной сессии."""
    et = event.get("type")
    if et not in ("file_change", "backup_created"):
        return
    with _session_lock:
        sid = _session_id
    if not sid:
        return
    try:
        if et == "file_change":
            session_store.append_checkpoint(
                sid,
                {
                    "ts": event.get("ts"),
                    "filename": event.get("filename") or "",
                    "action": event.get("action") or "",
                    "backup": event.get("backup"),
                    "added": event.get("added") or 0,
                    "removed": event.get("removed") or 0,
                    "diff": event.get("diff") or "",
                },
            )
        elif et == "backup_created" and event.get("backup"):
            # если file_change уже записал backup — не дублируем без filename stats
            pass
        data = session_store.get_session(sid)
        if data:
            events.emit(
                "session_changes",
                session_id=sid,
                changes=data.get("changes") or {},
            )
    except Exception:  # noqa: BLE001
        pass


events.subscribe(_on_file_change_checkpoint)


def _load_persisted_session(session_id: str) -> dict[str, Any] | None:
    global _session_messages, _session_id
    data = session_store.get_session(session_id)
    if data is None:
        return None
    with _session_lock:
        _session_id = data["session_id"]
        _session_messages = list(data.get("messages") or [])
    return data


def _worker(payload: dict[str, Any]) -> None:
    global _running, _token_totals, _session_messages, _last_tok_s
    _token_totals = {"prompt": 0, "completion": 0}
    _last_tok_s = None
    events.clear_stop()
    mode = (payload.get("mode") or "agent").strip().lower()
    task = (payload.get("task") or "").strip()
    workspace = (payload.get("workspace") or "./sandbox").strip()
    clean = bool(payload.get("clean"))
    max_steps = int(payload.get("max_steps") or 12)
    base_url = (payload.get("base_url") or DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    api_key = (payload.get("api_key") or DEFAULT_API_KEY).strip() or DEFAULT_API_KEY
    model_arg = payload.get("model")
    new_session = bool(payload.get("new_session"))
    model_architect = (payload.get("model_architect") or "").strip() or None
    model_coder = (payload.get("model_coder") or "").strip() or None
    model_reviewer = (payload.get("model_reviewer") or "").strip() or None
    auto_test_raw = payload.get("auto_test")
    if auto_test_raw is None:
        auto_test: bool | None = None
    else:
        auto_test = bool(auto_test_raw)

    code = 1
    try:
        apply_cli_workspace(workspace, clean)
        client = make_client(base_url=base_url, api_key=api_key)
        model = resolve_model(client, model_arg)
        events.emit("model", id=model)

        if mode in ("agent", "react"):
            if new_session:
                _reset_session()
            _ensure_session_for_run(task=task, mode="agent", model=model)
            with _session_lock:
                prior = list(_session_messages) if _session_messages else None
            code, msgs = run_agent(
                task,
                model,
                base_url,
                api_key,
                max_steps,
                messages=prior,
                auto_test=auto_test,
            )
            with _session_lock:
                _session_messages = msgs
            _persist_agent_session(
                task=task, mode="agent", model=model, messages=msgs
            )
            events.emit("session", **_session_info())
            events.emit("sessions_changed")
        elif mode in ("workflow", "wf"):
            code = run_workflow(task, model, base_url, api_key)
        elif mode in ("multi", "multi_agent", "multi-agent"):
            code = run_pipeline(
                task,
                model,
                base_url,
                api_key,
                max_steps,
                model_architect=model_architect,
                model_coder=model_coder,
                model_reviewer=model_reviewer,
            )
            # Мультиагент: сохраняем краткую сессию без полного ReAct-history
            try:
                session_store.create_session(
                    title=session_store.title_from_task(task),
                    mode="multi_agent",
                    model=model,
                    messages=[
                        {"role": "user", "content": task},
                        {
                            "role": "assistant",
                            "content": f"(multi-agent finished code={code})",
                        },
                    ],
                )
                events.emit("sessions_changed")
            except Exception:  # noqa: BLE001
                pass
        else:
            events.emit("error", message=f"Неизвестный режим: {mode}")
            code = 2
    except ValueError as exc:
        events.emit("error", message=str(exc))
        code = 2
    except Exception as exc:  # noqa: BLE001
        events.emit("error", message=str(exc), traceback=traceback.format_exc())
        code = 2
    finally:
        _running = False
        events.emit(
            "status",
            running=False,
            code=code,
            tokens=_token_totals,
            tokens_per_sec=_last_tok_s,
            session=_session_info(),
        )


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalAIAgentsUI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Тише в консоли: только ошибки
        if args and str(args[0]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            _send_file(self, STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return

        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            file_path = (STATIC_DIR / rel).resolve()
            try:
                file_path.relative_to(STATIC_DIR.resolve())
            except ValueError:
                self.send_error(403)
                return
            ctype = "application/octet-stream"
            if file_path.suffix == ".css":
                ctype = "text/css; charset=utf-8"
            elif file_path.suffix == ".js":
                ctype = "application/javascript; charset=utf-8"
            elif file_path.suffix == ".html":
                ctype = "text/html; charset=utf-8"
            _send_file(self, file_path, ctype)
            return

        if path == "/api/model":
            qs = parse_qs(parsed.query)
            base_url = (qs.get("base_url") or [None])[0] or DEFAULT_BASE_URL
            api_key = (qs.get("api_key") or [None])[0] or DEFAULT_API_KEY
            try:
                client = make_client(base_url=base_url, api_key=api_key)
                model = get_active_model(client)
                _send_json(
                    self,
                    200,
                    {"ok": True, "model": model, "base_url": base_url},
                )
            except Exception as exc:  # noqa: BLE001
                _send_json(
                    self,
                    200,
                    {
                        "ok": False,
                        "model": None,
                        "base_url": base_url,
                        "error": str(exc),
                    },
                )
            return

        if path == "/api/status":
            _send_json(
                self,
                200,
                {
                    "running": _running,
                    "workspace": str(get_workspace()),
                    "tokens": _token_totals,
                    "tokens_per_sec": _last_tok_s,
                    "context_limit": DEFAULT_CONTEXT_LIMIT,
                    "session": _session_info(),
                },
            )
            return

        if path == "/api/session":
            _send_json(self, 200, {"ok": True, **_session_info()})
            return

        if path == "/api/sessions":
            try:
                items = session_store.list_sessions()
                _send_json(self, 200, {"ok": True, "sessions": items})
            except Exception as exc:  # noqa: BLE001
                _send_json(self, 500, {"ok": False, "error": str(exc)})
            return

        if path.startswith("/api/sessions/") and path.rstrip("/").endswith("/diff"):
            # GET /api/sessions/<id>/diff?path=file.py
            rest = path[len("/api/sessions/") :].strip("/")
            if rest.endswith("/diff"):
                rest = rest[: -len("/diff")].rstrip("/")
            sid = rest
            if not sid or "/" in sid:
                _send_json(self, 400, {"ok": False, "error": "недопустимый session_id"})
                return
            qs = parse_qs(parsed.query)
            fname = (qs.get("path") or qs.get("file") or [""])[0]
            try:
                result = session_store.session_file_diff(sid, fname)
                code = 200 if result.get("ok") else 404
                _send_json(self, code, result)
            except Exception as exc:  # noqa: BLE001
                _send_json(self, 500, {"ok": False, "error": str(exc)})
            return

        if path.startswith("/api/sessions/"):
            sid = path[len("/api/sessions/") :].strip("/")
            if not sid or "/" in sid:
                _send_json(self, 400, {"ok": False, "error": "недопустимый session_id"})
                return
            data = session_store.get_session(sid)
            if data is None:
                _send_json(self, 404, {"ok": False, "error": "сессия не найдена"})
                return
            _send_json(self, 200, {"ok": True, "session": data})
            return

        if path == "/api/files":
            try:
                # Не вызывать set_workspace() здесь: он шлёт SSE "workspace",
                # а UI на это снова дергает /api/files → бесконечный цикл.
                ws = get_workspace()
                ws.mkdir(parents=True, exist_ok=True)
                _send_json(
                    self,
                    200,
                    {
                        "workspace": str(ws),
                        "files": list_workspace_tree(),
                        "mentions": list_mention_targets(),
                        "has_tests": workspace_has_tests(),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                _send_json(self, 500, {"error": str(exc)})
            return

        if path == "/api/file":
            qs = parse_qs(parsed.query)
            rel = (qs.get("path") or [""])[0]
            file_path = _safe_workspace_file(rel)
            if file_path is None:
                _send_json(self, 404, {"error": "файл не найден"})
                return
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                _send_json(
                    self,
                    200,
                    {"path": rel, "content": text, "size": file_path.stat().st_size},
                )
            except OSError as exc:
                _send_json(self, 500, {"error": str(exc)})
            return

        if path == "/api/backups":
            try:
                _send_json(
                    self,
                    200,
                    {
                        "workspace": str(get_workspace()),
                        "backups": list_backups(),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                _send_json(self, 500, {"error": str(exc)})
            return

        if path == "/api/events":
            self._sse_stream()
            return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:  # noqa: N802
        global _running, _run_thread
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/run":
            payload = _read_json(self)
            task = (payload.get("task") or "").strip()
            if not task:
                _send_json(self, 400, {"ok": False, "error": "task обязателен"})
                return
            with _run_lock:
                if _running:
                    _send_json(self, 409, {"ok": False, "error": "Уже выполняется"})
                    return
                _running = True
                events.emit("status", running=True)
                _run_thread = threading.Thread(
                    target=_worker,
                    args=(payload,),
                    daemon=True,
                    name="agent-run",
                )
                _run_thread.start()
            _send_json(self, 200, {"ok": True, "started": True})
            return

        if path == "/api/stop":
            events.request_stop()
            _send_json(self, 200, {"ok": True, "stopping": True})
            return

        if path == "/api/session/reset" or path == "/api/session/new":
            _reset_session()
            events.emit("session", **_session_info())
            _send_json(self, 200, {"ok": True, **_session_info()})
            return

        if path.startswith("/api/sessions/") and path.endswith("/load"):
            # POST /api/sessions/<id>/load — активировать сохранённую сессию
            sid = path[len("/api/sessions/") : -len("/load")].strip("/")
            if not sid or "/" in sid:
                _send_json(self, 400, {"ok": False, "error": "недопустимый session_id"})
                return
            data = _load_persisted_session(sid)
            if data is None:
                _send_json(self, 404, {"ok": False, "error": "сессия не найдена"})
                return
            events.emit("session", **_session_info())
            events.emit(
                "session_changes",
                session_id=sid,
                changes=data.get("changes") or {},
            )
            _send_json(
                self,
                200,
                {
                    "ok": True,
                    **_session_info(),
                    "title": data.get("title"),
                    "mode": data.get("mode"),
                    "model": data.get("model"),
                    "changes": data.get("changes") or {},
                },
            )
            return

        if path.startswith("/api/sessions/") and path.endswith("/revert"):
            sid = path[len("/api/sessions/") : -len("/revert")].strip("/")
            if not sid or "/" in sid:
                _send_json(self, 400, {"ok": False, "error": "недопустимый session_id"})
                return
            try:
                result = session_store.revert_session_changes(sid)
                code = 200 if result.get("ok") else 400
                if result.get("ok") or result.get("reverted"):
                    events.emit(
                        "session_changes",
                        session_id=sid,
                        changes={"files": [], "file_count": 0, "added": 0, "removed": 0},
                    )
                    events.emit("files_changed", workspace=str(get_workspace()))
                _send_json(self, code, result)
            except Exception as exc:  # noqa: BLE001
                _send_json(self, 500, {"ok": False, "error": str(exc)})
            return

        if path == "/api/rollback":
            payload = _read_json(self)
            name = (payload.get("backup") or payload.get("name") or "").strip()
            if not name:
                _send_json(self, 400, {"ok": False, "error": "backup обязателен"})
                return
            try:
                result = rollback_backup(name)
                code = 200 if result.get("ok") else 400
                _send_json(self, code, result)
            except Exception as exc:  # noqa: BLE001
                _send_json(self, 500, {"ok": False, "error": str(exc)})
            return

        if path == "/api/workspace":
            payload = _read_json(self)
            ws = payload.get("workspace") or "./sandbox"
            try:
                path_set = set_workspace(ws)
                _send_json(self, 200, {"ok": True, "workspace": str(path_set)})
            except Exception as exc:  # noqa: BLE001
                _send_json(self, 400, {"ok": False, "error": str(exc)})
            return

        self.send_error(404, "Not Found")

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/sessions/"):
            sid = path[len("/api/sessions/") :].strip("/")
            if not sid or "/" in sid:
                _send_json(self, 400, {"ok": False, "error": "недопустимый session_id"})
                return
            deleted = session_store.delete_session(sid)
            if not deleted:
                _send_json(self, 404, {"ok": False, "error": "сессия не найдена"})
                return
            with _session_lock:
                active = _session_id
            if active == sid:
                _reset_session()
                events.emit("session", **_session_info())
            events.emit("sessions_changed")
            _send_json(self, 200, {"ok": True, "deleted": sid})
            return
        self.send_error(404, "Not Found")

    def _sse_stream(self) -> None:
        q: queue.Queue = queue.Queue(maxsize=256)
        with _sse_lock:
            _sse_clients.append(q)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        # hello + текущий статус
        hello = json.dumps(
            {
                "type": "hello",
                "ts": time.time(),
                "running": _running,
                "workspace": str(get_workspace()),
            },
            ensure_ascii=False,
        )
        try:
            self.wfile.write(f"data: {hello}\n\n".encode("utf-8"))
            self.wfile.flush()
            while True:
                try:
                    item = q.get(timeout=15.0)
                    self.wfile.write(f"data: {item}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    # keep-alive comment
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            with _sse_lock:
                try:
                    _sse_clients.remove(q)
                except ValueError:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Web-UI для local-ai-agents")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Не открывать браузер автоматически",
    )
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not (STATIC_DIR / "index.html").is_file():
        print(f"Ошибка: не найден {STATIC_DIR / 'index.html'}")
        raise SystemExit(1)

    set_workspace("./sandbox")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Local AI Agents UI → {url}")
    print(f"  workspace={get_workspace()}")
    print("  Ctrl+C для остановки")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановка UI…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
