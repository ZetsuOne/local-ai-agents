"""Автономный ReAct-агент + 4 workspace-инструмента (оптимизация под 4B–12B)."""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import events
from llm import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    apply_loop_guard,
    chat_completion,
    compact_messages,
    log_token_usage,
    make_client,
    normalize_tool_calls,
    resolve_model,
    split_think_content,
    strip_tool_call_markup,
)

# ── Workspace ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = (ROOT / "sandbox").resolve()
BACKUP_DIR_NAME = ".agent_backup"
SKIP_DIR_NAMES = frozenset({"__pycache__", ".git", ".venv", BACKUP_DIR_NAME, "node_modules"})
SKIP_SUFFIXES = (".pyc",)

WORKSPACE_DIR: Path = DEFAULT_WORKSPACE

BASIC_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the workspace (or a subdirectory).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative subdirectory (default: workspace root)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the workspace (with line numbers).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run an allowlisted command in the workspace "
                "(pytest, python -m pytest, python <script.py>, unittest, "
                "ruff/flake8/mypy, git status/diff/log)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Command to run"},
                },
                "required": ["cmd"],
            },
        },
    },
]

CORE_TOOL_NAMES = frozenset(
    t["function"]["name"] for t in BASIC_TOOLS_SCHEMA if t.get("function", {}).get("name")
)


def get_workspace() -> Path:
    return WORKSPACE_DIR


def set_workspace(path: str | Path | None) -> Path:
    """Установить активный workspace. None → ./sandbox."""
    global WORKSPACE_DIR
    if path is None or str(path).strip() == "":
        target = DEFAULT_WORKSPACE
    else:
        target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR = target
    try:
        events.emit("workspace", path=str(WORKSPACE_DIR))
    except Exception:  # noqa: BLE001
        pass
    return WORKSPACE_DIR


def apply_cli_workspace(workspace: str, clean: bool = False) -> Path:
    """CLI/UI: установить workspace; --clean только для дефолтного ./sandbox."""
    ws_arg = (workspace or "./sandbox").strip() or "./sandbox"
    target = Path(ws_arg).expanduser()
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    else:
        target = target.resolve()

    default_sandbox = DEFAULT_WORKSPACE
    if clean:
        if target != default_sandbox:
            raise ValueError(
                "Ошибка: --clean разрешён только для дефолтного ./sandbox, "
                f"получено: {target}"
            )
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
    return set_workspace(target)


def _safe_path(rel: str) -> Path:
    """Resolve relative path strictly inside WORKSPACE_DIR (blocks ../ traversal)."""
    raw = (rel or "").strip().replace("\\", "/")
    if not raw or raw in (".", "./"):
        return WORKSPACE_DIR.resolve()
    if Path(raw).is_absolute():
        raise PermissionError(f"Ошибка: абсолютные пути запрещены: {rel!r}")
    parts = Path(raw).parts
    if ".." in parts:
        raise PermissionError(
            f"Ошибка: выход за пределы workspace запрещён: {rel!r}"
        )
    candidate = (WORKSPACE_DIR / raw).resolve()
    try:
        candidate.relative_to(WORKSPACE_DIR.resolve())
    except ValueError as exc:
        raise PermissionError(
            f"Ошибка: путь вне workspace: {rel!r}"
        ) from exc
    return candidate


def _rel_display(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE_DIR.resolve()).as_posix()
    except ValueError:
        return path.name


def _should_skip(path: Path) -> bool:
    if path.name in SKIP_DIR_NAMES:
        return True
    if path.suffix in SKIP_SUFFIXES:
        return True
    return False


def _unified_diff(filename: str, before: str, after: str) -> str:
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
    )
    return "\n".join(diff)


def _backup_file(path: Path) -> str | None:
    """Copy existing file into .agent_backup/; return backup basename or None."""
    if not path.is_file():
        return None
    bak_dir = WORKSPACE_DIR / BACKUP_DIR_NAME
    bak_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{path.name}"
    dest = bak_dir / name
    # avoid collisions within the same second
    n = 0
    while dest.exists():
        n += 1
        name = f"{stamp}_{n}_{path.name}"
        dest = bak_dir / name
    shutil.copy2(path, dest)
    try:
        events.emit("backup_created", backup=name, filename=_rel_display(path))
    except Exception:  # noqa: BLE001
        pass
    return name


def workspace_has_tests() -> bool:
    ws = WORKSPACE_DIR
    if not ws.is_dir():
        return False
    if (ws / "tests").is_dir():
        return True
    if (ws / "pytest.ini").is_file() or (ws / "pyproject.toml").is_file():
        return True
    for p in ws.rglob("test_*.py"):
        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        return True
    for p in ws.rglob("*_test.py"):
        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        return True
    return False


def list_workspace_tree(max_entries: int = 500) -> list[dict[str, Any]]:
    """Flat list of relative files for the UI tree."""
    ws = WORKSPACE_DIR.resolve()
    out: list[dict[str, Any]] = []
    if not ws.is_dir():
        return out
    for root, dirs, files in os.walk(ws):
        dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
        root_path = Path(root)
        for name in sorted(files):
            if name.endswith(SKIP_SUFFIXES):
                continue
            path = root_path / name
            try:
                rel = path.resolve().relative_to(ws).as_posix()
            except ValueError:
                continue
            out.append(
                {
                    "path": rel,
                    "name": name,
                    "size": path.stat().st_size if path.is_file() else 0,
                }
            )
            if len(out) >= max_entries:
                return out
    return out


def list_mention_targets(max_entries: int = 200) -> list[str]:
    """Autocomplete targets for @mentions in the UI."""
    files = [e["path"] for e in list_workspace_tree(max_entries=max_entries)]
    return files


def list_backups() -> list[dict[str, Any]]:
    bak_dir = WORKSPACE_DIR / BACKUP_DIR_NAME
    if not bak_dir.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for p in sorted(bak_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not p.is_file():
            continue
        # stamp_originalname or stamp_n_originalname
        orig = p.name
        m = re.match(r"^\d{8}_\d{6}(?:_\d+)?_(.+)$", p.name)
        if m:
            orig = m.group(1)
        items.append(
            {
                "name": p.name,
                "original": orig,
                "size": p.stat().st_size,
                "mtime": p.stat().st_mtime,
            }
        )
    return items


def read_backup_text(backup_name: str) -> dict[str, Any]:
    name = (backup_name or "").strip()
    if not name or "/" in name or "\\" in name or ".." in name:
        return {"ok": False, "error": "недопустимое имя бэкапа"}
    path = (WORKSPACE_DIR / BACKUP_DIR_NAME / name).resolve()
    try:
        path.relative_to((WORKSPACE_DIR / BACKUP_DIR_NAME).resolve())
    except ValueError:
        return {"ok": False, "error": "бэкап вне .agent_backup"}
    if not path.is_file():
        return {"ok": False, "error": "бэкап не найден"}
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "content": content, "name": name}


def rollback_backup(backup_name: str) -> dict[str, Any]:
    name = (backup_name or "").strip()
    data = read_backup_text(name)
    if not data.get("ok"):
        return data
    m = re.match(r"^\d{8}_\d{6}(?:_\d+)?_(.+)$", name)
    if not m:
        return {"ok": False, "error": "не удалось извлечь имя файла из бэкапа"}
    orig = m.group(1)
    try:
        dest = _safe_path(orig)
    except PermissionError as exc:
        return {"ok": False, "error": str(exc)}
    # backup current before restore
    _backup_file(dest) if dest.is_file() else None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(data["content"], encoding="utf-8")
    try:
        events.emit(
            "file_change",
            filename=orig.replace("\\", "/"),
            action="rollback",
            backup=name,
            added=0,
            removed=0,
            diff="",
        )
        events.emit("files_changed", workspace=str(WORKSPACE_DIR))
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "filename": orig, "backup": name, "message": f"OK: откат → {orig}"}


_MENTION_RE = re.compile(r"(?<!\w)@([A-Za-z0-9_.\-/\\]+)")


def resolve_mention_token(token: str) -> str:
    tok = (token or "").strip().lstrip("@").replace("\\", "/")
    if not tok:
        return "Ошибка: пустой @-mention"
    try:
        path = _safe_path(tok)
    except PermissionError as exc:
        return str(exc)
    if not path.is_file():
        return f"Ошибка: файл @{tok} не найден"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Ошибка чтения @{tok}: {exc}"
    lines = text.splitlines()
    numbered = "\n".join(f"{i} | {line}" for i, line in enumerate(lines, 1))
    return f"=== @{tok} ===\n{numbered}"


def resolve_mentions(task: str) -> str:
    """Inject @file contents into the user task (deduped)."""
    text = task or ""
    matches = list(_MENTION_RE.finditer(text))
    if not matches:
        return text
    seen: set[str] = set()
    blocks: list[str] = []
    for m in matches:
        tok = m.group(1).replace("\\", "/")
        if tok in seen:
            continue
        seen.add(tok)
        blocks.append(resolve_mention_token(tok))
    if not blocks:
        return text
    return text + "\n\n--- injected @-context ---\n" + "\n\n".join(blocks)


# ── Four core tools ───────────────────────────────────────────────────────────

def list_files(path: str = "") -> str:
    try:
        base = _safe_path(path) if (path or "").strip() else WORKSPACE_DIR.resolve()
    except PermissionError as exc:
        return str(exc)
    if not base.exists():
        return f"Ошибка: путь не найден: {path or '.'}"
    if base.is_file():
        return _rel_display(base)

    entries: list[str] = []
    try:
        children = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        return f"Ошибка: {exc}"
    for child in children:
        if _should_skip(child):
            continue
        rel = _rel_display(child)
        entries.append(f"{rel}/" if child.is_dir() else rel)
    if not entries:
        return "(пусто)"
    return "\n".join(entries)


def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    target = _safe_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Ошибка: файл не найден: {path}")
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    total = len(lines)
    start = 1 if start_line is None else max(1, int(start_line))
    end = total if end_line is None else min(total, int(end_line))
    if start > total:
        start = total if total else 1
    if end < start:
        end = start
    chunk = lines[start - 1 : end]
    numbered = "\n".join(f"{i} | {line}" for i, line in enumerate(chunk, start))
    rel = _rel_display(target)
    header = f"=== {rel} (строки {start}-{end} из {total}) ==="
    return header + "\n" + numbered


def write_file(path: str, content: str) -> str:
    target = _safe_path(path)
    before = ""
    existed = target.is_file()
    if existed:
        before = target.read_text(encoding="utf-8", errors="replace")
    backup = _backup_file(target) if existed else None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content if content is not None else "", encoding="utf-8")

    syntax_note = ""
    if target.suffix == ".py":
        try:
            ast.parse(content or "")
        except SyntaxError as exc:
            syntax_note = f" ⚠ SyntaxError: {exc.msg} (line {exc.lineno})"

    after = content or ""
    added = removed = 0
    for line in difflib.ndiff(before.splitlines(), after.splitlines()):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            removed += 1
    rel = _rel_display(target)
    diff = _unified_diff(rel, before, after)
    try:
        events.emit(
            "file_change",
            filename=rel,
            action="write_file",
            backup=backup,
            added=added,
            removed=removed,
            diff=diff,
        )
        events.emit("files_changed", workspace=str(WORKSPACE_DIR))
    except Exception:  # noqa: BLE001
        pass
    bak_msg = f" (backup={backup})" if backup else ""
    return f"OK: записано → {rel}{bak_msg}{syntax_note}"


_SHELL_META_RE = re.compile(r"[;&|><`]|\$\(")
_ALLOWED_GIT = frozenset({"status", "diff", "log"})


def _allowlist_argv(argv: list[str]) -> str | None:
    """Return error message if argv is not allowlisted; else None."""
    if not argv:
        return "пустая команда"
    head = argv[0].lower()
    # strip .exe / path for matching
    head_base = Path(head).name.lower().removesuffix(".exe")

    if head_base in {"pytest", "py.test"}:
        return None
    if head_base in {"ruff", "flake8", "mypy", "unittest"}:
        return None
    if head_base == "git":
        if len(argv) >= 2 and argv[1].lower() in _ALLOWED_GIT:
            return None
        return "allowlist: разрешены только git status/diff/log"
    if head_base in {"python", "python3", "py"}:
        # python -m pytest / python -m unittest ...
        if len(argv) >= 3 and argv[1] == "-m" and argv[2].lower() in {
            "pytest",
            "unittest",
            "ruff",
            "flake8",
            "mypy",
        }:
            return None
        # python script.py [args…] — script must live in workspace
        if len(argv) >= 2 and not argv[1].startswith("-"):
            script = argv[1]
            if not script.lower().endswith(".py"):
                return "allowlist: python разрешён только для .py в workspace или -m pytest/…"
            try:
                _safe_path(script)
            except PermissionError:
                return "allowlist: скрипт вне workspace"
            return None
        return "allowlist: python разрешён только для .py в workspace или -m pytest/…"
    return f"allowlist: команда {argv[0]!r} запрещена"


def run_command(cmd: str, timeout: int = 60) -> str:
    raw = (cmd or "").strip()
    if not raw:
        return json.dumps({"error": "пустая команда", "cmd": cmd}, ensure_ascii=False)
    if _SHELL_META_RE.search(raw):
        return json.dumps(
            {
                "error": "Отклонено: shell-метасимволы (; && || | > < ` $()) запрещены",
                "cmd": cmd,
            },
            ensure_ascii=False,
        )
    try:
        argv = shlex.split(raw, posix=(os.name != "nt"))
    except ValueError as exc:
        return json.dumps({"error": f"не удалось разобрать команду: {exc}", "cmd": cmd}, ensure_ascii=False)

    deny = _allowlist_argv(argv)
    if deny:
        return json.dumps({"error": deny, "cmd": cmd}, ensure_ascii=False)

    # Prefer the interpreter that runs this process (venv), not bare PATH names
    head_base = Path(argv[0]).name.lower().removesuffix(".exe")
    if head_base in {"python", "python3", "py"}:
        argv = [sys.executable, *argv[1:]]
    elif head_base in {"pytest", "py.test"}:
        argv = [sys.executable, "-m", "pytest", *argv[1:]]

    def _emit_and_dumps(payload: dict[str, Any]) -> str:
        try:
            events.emit("run_command", cmd=cmd, result=payload)
        except Exception:  # noqa: BLE001
            pass
        return json.dumps(payload, ensure_ascii=False)

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1, int(timeout)),
            shell=False,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        return _emit_and_dumps(
            {
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "")[-8000:],
                "stderr": (proc.stderr or "")[-4000:],
                "cmd": cmd,
                "duration_ms": duration_ms,
            }
        )
    except subprocess.TimeoutExpired:
        return _emit_and_dumps(
            {"error": f"timeout {timeout}s", "cmd": cmd, "returncode": -1}
        )
    except FileNotFoundError:
        return _emit_and_dumps(
            {"error": f"исполняемый файл не найден: {argv[0]}", "cmd": cmd}
        )
    except OSError as exc:
        return _emit_and_dumps({"error": str(exc), "cmd": cmd})


def filter_tools_schema(allowed: set[str] | frozenset[str]) -> list[dict[str, Any]]:
    return [
        t
        for t in BASIC_TOOLS_SCHEMA
        if (t.get("function") or {}).get("name") in allowed
    ]


def execute_tool(name: str, args: dict[str, Any] | None = None) -> str:
    """Dispatch one of the 4 core tools; normalize path/filename aliases."""
    args = dict(args or {})
    # aliases for older prompts / fallback JSON
    if "path" not in args and "filename" in args:
        args["path"] = args["filename"]
    if "cmd" not in args and "command" in args:
        args["cmd"] = args["command"]

    try:
        if name == "list_files":
            return list_files(str(args.get("path") or args.get("directory") or ""))
        if name == "read_file":
            path = str(args.get("path") or "")
            if not path:
                return "Ошибка: нужен path"
            return read_file(
                path,
                start_line=args.get("start_line"),
                end_line=args.get("end_line"),
            )
        if name == "write_file":
            path = str(args.get("path") or "")
            if not path:
                return "Ошибка: нужен path"
            return write_file(path, str(args.get("content") if args.get("content") is not None else ""))
        if name == "run_command":
            return run_command(str(args.get("cmd") or ""), timeout=int(args.get("timeout") or 60))
        return f"Ошибка: неизвестный инструмент {name!r} (доступны: {', '.join(sorted(CORE_TOOL_NAMES))})"
    except PermissionError as exc:
        return str(exc) if str(exc).startswith("Ошибка") else f"Ошибка: {exc}"
    except FileNotFoundError as exc:
        return str(exc) if str(exc).startswith("Ошибка") else f"Ошибка: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Ошибка: {exc}"


# ── ReAct agent ───────────────────────────────────────────────────────────────

DEFAULT_MAX_STEPS = 12
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BLUE = "\033[34m"

SYSTEM_PROMPT = """Ты локальный coding-агент. Работай строго внутри workspace.

Правила:
1. Если файлы уже прикреплены в запросе (@file) — НЕ вызывай read_file для них, сразу вноси изменения.
2. Для просмотра файлов используй list_files и read_file. Не читай один и тот же файл повторно, если не менял его.
3. Для правки файлов (до 400 строк) перезаписывай файл целиком через write_file — это надёжнее частичных патчей.
4. Проверяй изменения через run_command (pytest) только если в проекте есть Python-код или тесты. Сделал правки — сразу дай краткий финальный ответ без лишних вызовов."""

_WRITE_TOOLS = frozenset({"write_file"})


def _short(text: str, limit: int = 400) -> str:
    text = text.replace("\r\n", "\n")
    if len(text) <= limit:
        return text
    return text[:limit] + f"… (+{len(text) - limit} символов)"


def _log_step(step: int, max_steps: int) -> None:
    print(f"\n{BOLD}{BLUE}════════ Шаг {step}/{max_steps} ════════{RESET}")
    events.emit("step", step=step, max_steps=max_steps)


def _emit_reasoning(reasoning: str, *, active: bool = False, agent: str | None = None) -> None:
    if not reasoning:
        return
    print(f"{DIM}🧠 Reasoning:{RESET} {_short(reasoning, 400)}")
    payload: dict[str, Any] = {"content": reasoning, "active": active}
    if agent:
        payload["agent"] = agent
    events.emit("reasoning", **payload)


def _log_thought(content: str | None) -> None:
    reasoning, visible = split_think_content(content)
    if reasoning:
        _emit_reasoning(reasoning, active=True)
    visible = strip_tool_call_markup(visible) if visible else ""
    if visible:
        print(f"{CYAN}💭 Thought:{RESET} {_short(visible, 500)}")
        events.emit("thought", content=visible)
    elif content and content.strip() and not reasoning:
        cleaned = strip_tool_call_markup(content.strip())
        if cleaned:
            print(f"{CYAN}💭 Thought:{RESET} {_short(cleaned, 500)}")
            events.emit("thought", content=cleaned)


def _log_tool_call(name: str, args: dict[str, Any]) -> None:
    pretty = json.dumps(args, ensure_ascii=False)
    print(f"{YELLOW}🔧 Tool:{RESET} {BOLD}{name}{RESET}")
    print(f"{DIM}   args:{RESET} {_short(pretty, 300)}")
    events.emit("tool_call", name=name, args=args)


def _log_tool_result(result: str) -> None:
    print(f"{GREEN}✅ Result:{RESET} {_short(result, 500)}")
    events.emit("tool_result", result=result)


def _log_final(content: str) -> None:
    reasoning, visible = split_think_content(content)
    if reasoning:
        _emit_reasoning(reasoning, active=False)
    text = visible or (content or "").strip() or "(пустой ответ модели)"
    print(f"\n{BOLD}{MAGENTA}🏁 Final answer:{RESET}\n{text}")
    events.emit("final", content=text)


def _parse_tool_args(raw: str | None) -> dict[str, Any]:
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _is_pytest_cmd(name: str, args: dict[str, Any]) -> bool:
    if name != "run_command":
        return False
    cmd = str(args.get("cmd") or "").lower()
    return "pytest" in cmd


def _pytest_failed(result: str) -> bool:
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return True
    if not isinstance(data, dict):
        return True
    if data.get("error"):
        return True
    return data.get("returncode", 1) not in (0, None)


def _final_report_without_tools(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    messages = compact_messages(messages)
    resp = chat_completion(
        client,
        model=model,
        messages=messages,
        temperature=0.2,
    )
    log_token_usage(resp)
    msg = resp.choices[0].message
    raw = (msg.content or "").strip() or "(пустой ответ модели)"
    _reasoning, visible = split_think_content(raw)
    final = visible or raw
    messages.append({"role": "assistant", "content": final})
    return raw, messages


def _inject_auto_test(
    messages: list[dict[str, Any]],
    pytest_result: str,
) -> list[dict[str, Any]]:
    failed = _pytest_failed(pytest_result)
    if failed:
        nudge = (
            "⚠ AUTO-TEST: pytest был запущен автоматически после твоих правок .py "
            "и УПАЛ. Исправь код через write_file и снова прогони "
            f"pytest через run_command. Результат:\n{pytest_result}"
        )
    else:
        nudge = (
            "✓ AUTO-TEST: pytest прошёл успешно после правок. "
            "Можешь дать финальный ответ.\n"
            f"{pytest_result}"
        )
    messages.append({"role": "user", "content": nudge})
    events.emit("auto_test", failed=failed, result=pytest_result)
    return messages


def run_agent(
    task: str,
    model: str,
    base_url: str,
    api_key: str,
    max_steps: int,
    messages: list[dict[str, Any]] | None = None,
    *,
    auto_test: bool | None = None,
    resolve_at_mentions: bool = True,
) -> tuple[int, list[dict[str, Any]]]:
    """Запустить ReAct-цикл."""
    client = make_client(base_url=base_url, api_key=api_key)
    follow_up = bool(messages)

    if auto_test is None:
        auto_test = workspace_has_tests()

    task_text = resolve_mentions(task) if resolve_at_mentions else task

    if messages:
        messages = list(messages)
        messages.append({"role": "user", "content": task_text})
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_text},
        ]
    last_call: tuple[str, str] | None = None
    last_was_tool = False
    py_modified = False
    pytest_since_edit = False
    auto_heal_used = False
    ws = get_workspace()

    print(f"{BOLD}ReAct Agent{RESET}")
    print(f"  model={model}")
    print(f"  max_steps={max_steps}")
    print(f"  workspace={ws}")
    print(f"  auto_test={auto_test}")
    print(f"  task={task}")
    if follow_up:
        print(f"  follow_up=yes (история {len(messages) - 1} сообщений до задачи)")
    events.emit(
        "run_start",
        mode="agent",
        model=model,
        workspace=str(ws),
        task=task,
        max_steps=max_steps,
        follow_up=follow_up,
        auto_test=auto_test,
    )
    step = 0
    while step < max_steps:
        step += 1
        if events.should_stop():
            print(f"\n{YELLOW}⏹ Остановлено пользователем{RESET}")
            events.emit("run_end", status="stopped", code=130)
            return 130, messages
        _log_step(step, max_steps)
        messages = compact_messages(messages)
        try:
            resp = chat_completion(
                client,
                model=model,
                messages=messages,
                tools=BASIC_TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{RED}API error:{RESET} {exc}")
            events.emit("error", message=str(exc))
            events.emit("run_end", status="error", code=2)
            return 2, messages

        log_token_usage(resp)
        msg = resp.choices[0].message
        tool_calls = normalize_tool_calls(msg, known_names=CORE_TOOL_NAMES)
        if not tool_calls:
            raw_final = (msg.content or "").strip() or "(пустой ответ модели)"
            _reasoning, visible_final = split_think_content(raw_final)
            final = visible_final or raw_final

            if (
                auto_test
                and py_modified
                and not pytest_since_edit
                and not auto_heal_used
                and not events.should_stop()
            ):
                print(
                    f"\n{YELLOW}🧪 AUTO-TEST: .py изменены без pytest — "
                    f"запускаю pytest -q…{RESET}"
                )
                events.emit("auto_test_start", cmd="pytest -q")
                pytest_result = run_command("pytest -q", timeout=60)
                pytest_since_edit = True
                _log_tool_result(pytest_result)
                messages.append({"role": "assistant", "content": final})
                messages = _inject_auto_test(messages, pytest_result)
                auto_heal_used = True
                if _pytest_failed(pytest_result):
                    last_was_tool = True
                    continue
                _log_final(raw_final)
                events.emit("run_end", status="ok", code=0)
                return 0, messages

            messages.append({"role": "assistant", "content": final})
            _log_final(raw_final)
            events.emit("run_end", status="ok", code=0)
            return 0, messages
        _log_thought(msg.content)
        last_was_tool = True
        _r, thought_visible = split_think_content(msg.content)
        assistant_content = thought_visible if thought_visible else (msg.content or "")
        if not (msg.tool_calls or []):
            assistant_content = strip_tool_call_markup(assistant_content)

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in tool_calls
            ],
        }
        messages.append(assistant_msg)
        for tc in tool_calls:
            if events.should_stop():
                print(f"\n{YELLOW}⏹ Остановлено пользователем{RESET}")
                events.emit("run_end", status="stopped", code=130)
                return 130, messages
            name = tc.function.name
            args = _parse_tool_args(tc.function.arguments)
            _log_tool_call(name, args)
            result = execute_tool(name, args)
            result, last_call = apply_loop_guard(name, args, result, last_call)
            _log_tool_result(result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )
            if name in _WRITE_TOOLS:
                target = str(args.get("path") or args.get("filename") or "")
                if target.lower().endswith(".py") or (
                    "OK:" in result and ".py" in result
                ):
                    py_modified = True
                    pytest_since_edit = False
            if _is_pytest_cmd(name, args):
                pytest_since_edit = True

    if last_was_tool and not events.should_stop():
        print(
            f"\n{YELLOW}⚠ Лимит шагов ({max_steps}) исчерпан — "
            f"запрашиваю итоговый отчёт без инструментов…{RESET}"
        )
        try:
            final, messages = _final_report_without_tools(client, model, messages)
            _log_final(final)
        except Exception as exc:  # noqa: BLE001
            print(f"{RED}API error (финальный отчёт):{RESET} {exc}")
            events.emit("error", message=str(exc))
            events.emit("run_end", status="max_steps", code=1)
            return 1, messages
        events.emit("run_end", status="max_steps", code=1)
        return 1, messages

    print(f"\n{RED}✗ Лимит шагов ({max_steps}) исчерпан без финального ответа{RESET}")
    events.emit("run_end", status="max_steps", code=1)
    return 1, messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Автономный ReAct-агент для локального LM Studio",
    )
    parser.add_argument("--task", required=True, help="Задача для агента")
    parser.add_argument(
        "--model",
        default=None,
        help="Имя модели в LM Studio (по умолчанию — первая загруженная чат-модель)",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument(
        "--workspace",
        default="./sandbox",
        help="Рабочая директория агента (по умолчанию ./sandbox)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Очистить ./sandbox перед запуском (только для дефолтной папки)",
    )
    parser.add_argument(
        "--auto-test",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "После правок .py автоматически гонять pytest, если модель "
            "завершилась без проверки (по умолчанию: вкл. при наличии tests/)"
        ),
    )
    args = parser.parse_args()
    try:
        apply_cli_workspace(args.workspace, args.clean)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    client = make_client(base_url=args.base_url, api_key=args.api_key)
    try:
        model = resolve_model(client, args.model)
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка определения модели: {exc}", file=sys.stderr)
        sys.exit(2)

    code, _messages = run_agent(
        task=args.task,
        model=model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_steps=args.max_steps,
        auto_test=args.auto_test,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
