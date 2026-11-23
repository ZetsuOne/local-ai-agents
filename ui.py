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
from agent import run_agent
from llm import DEFAULT_API_KEY, DEFAULT_BASE_URL, get_active_model, make_client, resolve_model
from multi_agent import run_pipeline
from tools import (
    apply_cli_workspace,
    get_workspace,
    list_backups,
    list_workspace_tree,
    rollback_backup,
    set_workspace,
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
# История messages активного Agent-сеанса (follow-up в Web-UI)
_session_lock = threading.Lock()
_session_messages: list[dict[str, Any]] | None = None


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
    global _token_totals
    if event.get("type") == "tokens":
        _token_totals["prompt"] += int(event.get("prompt_tokens") or 0)
        _token_totals["completion"] += int(event.get("completion_tokens") or 0)
        event = {
            **event,
            "total_prompt": _token_totals["prompt"],
            "total_completion": _token_totals["completion"],
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
    return {"active": active, "messages": count}


def _reset_session() -> None:
    global _session_messages
    with _session_lock:
        _session_messages = None


def _worker(payload: dict[str, Any]) -> None:
    global _running, _token_totals, _session_messages
    _token_totals = {"prompt": 0, "completion": 0}
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

    code = 1
    try:
        apply_cli_workspace(workspace, clean)
        client = make_client(base_url=base_url, api_key=api_key)
        model = resolve_model(client, model_arg)
        events.emit("model", id=model)

        if mode in ("agent", "react"):
            if new_session:
                _reset_session()
            with _session_lock:
                prior = list(_session_messages) if _session_messages else None
            code, msgs = run_agent(
                task, model, base_url, api_key, max_steps, messages=prior
            )
            with _session_lock:
                _session_messages = msgs
            events.emit("session", **_session_info())
        elif mode in ("workflow", "wf"):
            code = run_workflow(task, model, base_url, api_key)
        elif mode in ("multi", "multi_agent", "multi-agent"):
            code = run_pipeline(task, model, base_url, api_key, max_steps)
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
                    "session": _session_info(),
                },
            )
            return

        if path == "/api/session":
            _send_json(self, 200, {"ok": True, **_session_info()})
            return

        if path == "/api/files":
            try:
                set_workspace(get_workspace())
                _send_json(
                    self,
                    200,
                    {
                        "workspace": str(get_workspace()),
                        "files": list_workspace_tree(),
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
