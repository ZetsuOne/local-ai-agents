"""Изолированная рабочая директория и боевые инструменты для агентов."""
from __future__ import annotations
import ast
import difflib
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import events
DEFAULT_WORKSPACE = (Path(__file__).resolve().parent / "sandbox").resolve()
WORKSPACE_DIR: Path = DEFAULT_WORKSPACE
# Обратная совместимость: модули, читающие SANDBOX_DIR через tools.SANDBOX_DIR
SANDBOX_DIR: Path = WORKSPACE_DIR
_BACKUP_DIR_NAME = ".agent_backup"
_MAX_OUTPUT = 4000
_PACKAGE_RE = re.compile(r"^[a-zA-Z0-9._-]+$")
_SKIP_DIR_NAMES = frozenset({"__pycache__", ".git", ".venv", _BACKUP_DIR_NAME})
_BACKUP_NAME_RE = re.compile(r"^(\d{8}_\d{6})(?:_(\d+))?_(.+)$")
_TEXT_EXTENSIONS = {
    ".py", ".txt", ".md", ".json", ".csv", ".sql", ".yml", ".yaml",
    ".toml", ".ini", ".cfg", ".xml", ".html", ".css", ".js", ".ts",
    ".log", ".sh", ".bat", ".ps1",
}
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

def get_workspace() -> Path:
    """Активная рабочая директория."""
    return WORKSPACE_DIR

def set_workspace(path: str | Path | None = None) -> Path:
    """Установить рабочую папку (по умолчанию ./sandbox). Создаёт каталог при необходимости."""
    global WORKSPACE_DIR, SANDBOX_DIR
    target = Path(path) if path else DEFAULT_WORKSPACE
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    else:
        target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR = target
    SANDBOX_DIR = WORKSPACE_DIR
    events.emit("workspace", path=str(WORKSPACE_DIR))
    return WORKSPACE_DIR

def is_default_workspace(path: str | Path | None = None) -> bool:
    """True, если путь совпадает с дефолтным ./sandbox."""
    candidate = Path(path).resolve() if path else WORKSPACE_DIR
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return candidate.resolve() == DEFAULT_WORKSPACE.resolve()

def clean_workspace() -> str:
    """Очистить содержимое активной WORKSPACE_DIR (файлы и подпапки)."""
    ws = WORKSPACE_DIR
    removed = 0
    if ws.is_dir():
        for child in list(ws.iterdir()):
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
                removed += 1
            elif child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
    ws.mkdir(parents=True, exist_ok=True)
    msg = f"OK: очищено {removed} объектов в {ws}"
    events.emit("workspace_cleaned", path=str(ws), removed=removed)
    return msg

def apply_cli_workspace(workspace: str | Path | None, clean: bool) -> Path:
    """Установить workspace; --clean разрешён только для дефолтного ./sandbox."""
    ws = set_workspace(workspace or str(DEFAULT_WORKSPACE))
    if clean:
        if not is_default_workspace(ws):
            raise ValueError(
                "Ошибка: --clean работает ТОЛЬКО для дефолтной ./sandbox"
            )
        print(clean_workspace())
    return ws

def _safe_path(filename: str) -> Path:
    """Разрешить путь строго внутри WORKSPACE_DIR (защита от path traversal)."""
    if not filename or filename.strip() == "":
        raise ValueError("filename не может быть пустым")
    candidate = (WORKSPACE_DIR / filename).resolve()
    try:
        candidate.relative_to(WORKSPACE_DIR)
    except ValueError as exc:
        raise PermissionError(
            f"Доступ запрещён: путь выходит за пределы workspace ({filename!r})"
        ) from exc
    return candidate

def _is_backup_path(path: Path) -> bool:
    """Путь внутри .agent_backup (игнорируется в list/search)."""
    try:
        rel = path.resolve().relative_to(WORKSPACE_DIR.resolve())
    except ValueError:
        return False
    return bool(rel.parts) and rel.parts[0] == _BACKUP_DIR_NAME

def _should_skip_path(path: Path) -> bool:
    """Игнорировать служебные каталоги и *.pyc в list/search/tree."""
    try:
        rel = path.resolve().relative_to(WORKSPACE_DIR.resolve())
    except ValueError:
        return True
    if any(part in _SKIP_DIR_NAMES for part in rel.parts):
        return True
    if path.is_file() and path.suffix.lower() == ".pyc":
        return True
    return False

def _parse_backup_filename(name: str) -> tuple[str, str] | None:
    """Из имени бэкапа извлечь (stamp, original_filename)."""
    match = _BACKUP_NAME_RE.match(name)
    if not match:
        return None
    return match.group(1), match.group(3)

def _truncate(text: str, limit: int = _MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [обрезано, всего {len(text)} символов]"

def _backup_existing(path: Path) -> Path | None:
    """Сохранить копию существующего файла в <workspace>/.agent_backup/."""
    if not path.is_file():
        return None
    backup_root = WORKSPACE_DIR / _BACKUP_DIR_NAME
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_root / f"{stamp}_{path.name}"
    # Уникальность при коллизии в ту же секунду
    if dest.exists():
        n = 1
        while True:
            candidate = backup_root / f"{stamp}_{n}_{path.name}"
            if not candidate.exists():
                dest = candidate
                break
            n += 1
    shutil.copy2(path, dest)
    return dest

def _check_python_syntax(filename: str, content: str) -> str | None:
    """Проверить .py через ast.parse. None = OK, иначе предупреждение для модели."""
    if not filename.lower().endswith(".py"):
        return None
    try:
        ast.parse(content)
        return None
    except SyntaxError as exc:
        line = exc.lineno or "?"
        text = exc.msg or str(exc)
        return (
            f"⚠ SyntaxError в {filename} на строке {line}: {text}. "
            f"Исправь синтаксис немедленно (write_file / replace_lines / edit_file), "
            f"не трать шаг на run_python."
        )

def _unified_diff(filename: str, before: str | None, after: str) -> str:
    """Классический построчный unified diff (stdlib difflib)."""
    old_lines = (before or "").splitlines()
    new_lines = (after or "").splitlines()
    diff_iter = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
        n=3,
    )
    return "\n".join(diff_iter)

def _emit_file_change(
    filename: str,
    action: str,
    before: str | None,
    after: str,
) -> None:
    diff = _unified_diff(filename, before, after)
    events.emit(
        "file_change",
        filename=filename,
        action=action,
        diff=diff,
        before=before,
        after=after,
    )
    events.emit("files_changed", workspace=str(WORKSPACE_DIR))

def list_workspace_tree() -> list[dict[str, Any]]:
    """Дерево файлов workspace для Web-UI (без служебных каталогов)."""
    nodes: list[dict[str, Any]] = []
    if not WORKSPACE_DIR.is_dir():
        return nodes
    for path in sorted(WORKSPACE_DIR.rglob("*")):
        if _should_skip_path(path):
            continue
        try:
            rel = path.relative_to(WORKSPACE_DIR).as_posix()
        except ValueError:
            continue
        nodes.append(
            {
                "path": rel,
                "name": path.name,
                "is_dir": path.is_dir(),
                "size": path.stat().st_size if path.is_file() else 0,
            }
        )
    return nodes

def list_backups() -> list[dict[str, Any]]:
    """Список файлов в <workspace>/.agent_backup/ для Web-UI."""
    backup_root = WORKSPACE_DIR / _BACKUP_DIR_NAME
    if not backup_root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(backup_root.iterdir(), key=lambda p: p.name, reverse=True):
        if not path.is_file():
            continue
        parsed = _parse_backup_filename(path.name)
        try:
            st = path.stat()
            size = st.st_size
            mtime = st.st_mtime
        except OSError:
            size, mtime = 0, 0.0
        items.append(
            {
                "name": path.name,
                "original": parsed[1] if parsed else path.name,
                "stamp": parsed[0] if parsed else "",
                "size": size,
                "mtime": mtime,
            }
        )
    return items

def rollback_backup(backup_name: str) -> dict[str, Any]:
    """Восстановить бэкап поверх текущего файла (текущая версия сначала бэкапится)."""
    name = (backup_name or "").strip()
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        return {"ok": False, "error": "недопустимое имя бэкапа"}
    if Path(name).name != name:
        return {"ok": False, "error": "недопустимое имя бэкапа"}

    backup_root = (WORKSPACE_DIR / _BACKUP_DIR_NAME).resolve()
    backup_path = (backup_root / name).resolve()
    try:
        backup_path.relative_to(backup_root)
    except ValueError:
        return {"ok": False, "error": "путь выходит за пределы .agent_backup"}
    if not backup_path.is_file():
        return {"ok": False, "error": f"бэкап не найден: {name}"}

    parsed = _parse_backup_filename(name)
    original = parsed[1] if parsed else name
    try:
        target = _safe_path(original)
    except (ValueError, PermissionError) as exc:
        return {"ok": False, "error": str(exc)}

    before: str | None = None
    if target.is_file():
        try:
            before = target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            before = None
        _backup_existing(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_path, target)
    try:
        after = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        after = ""
    _emit_file_change(original, "rollback", before, after)
    return {
        "ok": True,
        "backup": name,
        "restored": original,
        "message": f"OK: восстановлен {original} из {name}",
    }

# ── Инструменты ──────────────────────────────────────────────────────────────

def list_files() -> str:
    """Список файлов в workspace (без __pycache__/.git/.venv/.agent_backup/*.pyc)."""
    entries: list[str] = []
    for path in sorted(WORKSPACE_DIR.rglob("*")):
        if not path.is_file():
            continue
        if _should_skip_path(path):
            continue
        rel = path.relative_to(WORKSPACE_DIR).as_posix()
        size = path.stat().st_size
        entries.append(f"{rel} ({size} bytes)")
    if not entries:
        return "(workspace пуст)"
    return "\n".join(entries)

def read_file(filename: str, start_line: int = 1, end_line: int = 250) -> str:
    """Чтение файла с нумерацией строк (`1 | ...`)."""
    path = _safe_path(filename)
    if not path.is_file():
        return f"Ошибка: файл не найден: {filename}"
    start_line = max(1, int(start_line))
    end_line = max(start_line, int(end_line))
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)
    chunk = lines[start_line - 1 : end_line]
    if not chunk:
        return (
            f"=== {filename} (строки {start_line}-{end_line} из {total}) ===\n"
            f"(файл пуст или диапазон вне границ)"
        )
    end_actual = start_line + len(chunk) - 1
    numbered = [f"{i} | {line}" for i, line in enumerate(chunk, start=start_line)]
    header = f"=== {filename} (строки {start_line}-{end_actual} из {total}) ==="
    return header + "\n" + "\n".join(numbered)

def get_file_outline(filename: str) -> str:
    """Компактный outline .py файла: классы, функции и номера строк (через ast)."""
    path = _safe_path(filename)
    if not path.is_file():
        return f"Ошибка: файл не найден: {filename}"
    if path.suffix.lower() != ".py":
        return "Ошибка: get_file_outline поддерживает только .py файлы"

    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        line = exc.lineno or "?"
        return f"Ошибка синтаксиса в {filename} на строке {line}: {exc.msg or exc}"

    rows: list[str] = []

    def _fmt_func(node: ast.AST, indent: int = 0) -> None:
        pad = "  " * indent
        if isinstance(node, ast.AsyncFunctionDef):
            rows.append(f"{pad}async def {node.name}  L{node.lineno}")
        elif isinstance(node, ast.FunctionDef):
            rows.append(f"{pad}def {node.name}  L{node.lineno}")

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            rows.append(f"class {node.name}  L{node.lineno}")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _fmt_func(item, indent=1)
                elif isinstance(item, ast.ClassDef):
                    rows.append(f"  class {item.name}  L{item.lineno}")
                    for nested in item.body:
                        if isinstance(nested, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            _fmt_func(nested, indent=2)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _fmt_func(node, indent=0)

    if not rows:
        return f"=== outline: {filename} ===\n(нет классов/функций)"
    return f"=== outline: {filename} ===\n" + "\n".join(rows)

def write_file(filename: str, content: str) -> str:
    """Создание/перезапись файла в workspace (с бэкапом и проверкой синтаксиса .py)."""
    path = _safe_path(filename)
    before: str | None = None
    if path.is_file():
        before = path.read_text(encoding="utf-8", errors="replace")
        _backup_existing(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _emit_file_change(filename, "write_file", before, content)
    msg = f"OK: записано {len(content)} символов → {filename}"
    syntax_warn = _check_python_syntax(filename, content)
    if syntax_warn:
        msg = f"{msg}\n{syntax_warn}"
    return msg

def edit_file(filename: str, old_text: str, new_text: str) -> str:
    """Точечная замена уникального фрагмента текста."""
    path = _safe_path(filename)
    if not path.is_file():
        return f"Ошибка: файл не найден: {filename}"
    content = path.read_text(encoding="utf-8", errors="replace")
    count = content.count(old_text)
    if count == 0:
        return "Ошибка: фрагмент old_text не найден в файле"
    if count > 1:
        return f"Ошибка: фрагмент old_text встречается {count} раз (нужна уникальность)"
    _backup_existing(path)
    updated = content.replace(old_text, new_text, 1)
    path.write_text(updated, encoding="utf-8")
    _emit_file_change(filename, "edit_file", content, updated)
    msg = f"OK: замена выполнена в {filename}"
    syntax_warn = _check_python_syntax(filename, updated)
    if syntax_warn:
        msg = f"{msg}\n{syntax_warn}"
    return msg

def replace_lines(filename: str, start_line: int, end_line: int, new_text: str) -> str:
    """Замена диапазона строк [start_line, end_line] (1-based, включительно) на new_text."""
    path = _safe_path(filename)
    if not path.is_file():
        return f"Ошибка: файл не найден: {filename}"
    start_line = int(start_line)
    end_line = int(end_line)
    if start_line < 1:
        return "Ошибка: start_line должен быть >= 1"
    if end_line < start_line:
        return "Ошибка: end_line должен быть >= start_line"

    original = path.read_text(encoding="utf-8", errors="replace")
    lines = original.splitlines(keepends=True)
    total = len(lines)
    if start_line > total + 1:
        return f"Ошибка: start_line={start_line} за пределами файла ({total} строк)"
    if end_line > total:
        end_line = total
    if start_line == total + 1:
        end_line = total

    replacement = new_text
    if replacement and not replacement.endswith("\n"):
        replacement = replacement + "\n"
    new_lines = replacement.splitlines(keepends=True) if replacement else []

    before = lines[: start_line - 1]
    after = lines[end_line:] if end_line <= total else []
    updated = "".join(before + new_lines + after)
    _backup_existing(path)
    path.write_text(updated, encoding="utf-8")
    _emit_file_change(filename, "replace_lines", original, updated)
    msg = (
        f"OK: заменены строки {start_line}-{end_line} в {filename} "
        f"({len(new_lines)} новых строк)"
    )
    syntax_warn = _check_python_syntax(filename, updated)
    if syntax_warn:
        msg = f"{msg}\n{syntax_warn}"
    return msg

def search_in_files(query: str) -> str:
    """Поиск по текстовым файлам (без __pycache__/.git/.venv/.agent_backup/*.pyc)."""
    if not query:
        return "Ошибка: query пуст"
    hits: list[str] = []
    for path in sorted(WORKSPACE_DIR.rglob("*")):
        if not path.is_file():
            continue
        if _should_skip_path(path):
            continue
        if path.suffix.lower() not in _TEXT_EXTENSIONS and path.suffix:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(WORKSPACE_DIR).as_posix()
        for i, line in enumerate(text.splitlines(), start=1):
            if query in line:
                hits.append(f"{rel}:{i}: {line}")
    if not hits:
        return f"(совпадений не найдено для {query!r})"
    return _truncate("\n".join(hits))

def run_python(filename: str, args: str = "") -> str:
    """Запуск .py файла через sys.executable с cwd=WORKSPACE_DIR, таймаут 15 сек."""
    path = _safe_path(filename)
    if not path.is_file():
        return json.dumps({"error": f"файл не найден: {filename}"}, ensure_ascii=False)
    if path.suffix.lower() != ".py":
        return json.dumps({"error": "разрешены только .py файлы"}, ensure_ascii=False)

    cmd = [sys.executable, str(path)]
    if args and args.strip():
        cmd.extend(args.split())

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(WORKSPACE_DIR),
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
        result = {
            "returncode": proc.returncode,
            "stdout": _truncate(proc.stdout or ""),
            "stderr": _truncate(proc.stderr or ""),
        }
    except subprocess.TimeoutExpired:
        result = {
            "returncode": -1,
            "stdout": "",
            "stderr": "TimeoutExpired: превышен лимит 15 секунд",
        }
    events.emit("run_python", filename=filename, result=result)
    return json.dumps(result, ensure_ascii=False, indent=2)

def pip_install(package: str) -> str:
    """Безопасная установка пакета через pip текущего интерпретатора."""
    package = (package or "").strip()
    if not _PACKAGE_RE.match(package):
        return (
            "Ошибка: недопустимое имя пакета. "
            "Разрешены только буквы, цифры, ., _, -"
        )
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
        return json.dumps(
            {
                "returncode": proc.returncode,
                "stdout": _truncate(proc.stdout or ""),
                "stderr": _truncate(proc.stderr or ""),
            },
            ensure_ascii=False,
            indent=2,
        )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {"returncode": -1, "stdout": "", "stderr": "TimeoutExpired: pip > 120 сек"},
            ensure_ascii=False,
        )

def sqlite_query(db_filename: str, sql: str) -> str:
    """Только read-only запросы (SELECT / PRAGMA) к SQLite в workspace."""
    raw_sql = sql or ""
    if ";" in raw_sql:
        return "Ошибка: символ ';' запрещён (только один SELECT/PRAGMA без цепочки)"

    cleaned = raw_sql.strip()
    if not cleaned:
        return "Ошибка: пустой SQL"
    upper = cleaned.upper()
    if not (upper.startswith("SELECT") or upper.startswith("PRAGMA")):
        return "Ошибка: разрешены только запросы, начинающиеся с SELECT или PRAGMA"

    # Запрет опасных конструкций даже внутри SELECT
    forbidden = (
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
        "ATTACH", "DETACH", "REPLACE", "VACUUM", "REINDEX",
    )
    tokens = re.findall(r"[A-Za-z_]+", upper)
    for tok in tokens:
        if tok in forbidden:
            return f"Ошибка: запрещённая операция {tok} в SQL"

    path = _safe_path(db_filename)
    if not path.is_file():
        return f"Ошибка: база не найдена: {db_filename}"

    try:
        uri = path.as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(cleaned)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
            data = [dict(zip(cols, row)) for row in rows]
            return json.dumps(
                {"columns": cols, "rows": data, "count": len(data)},
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return f"Ошибка SQLite: {exc}"

# ── Схема и диспетчер ────────────────────────────────────────────────────────

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Список всех файлов в workspace с размерами.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Прочитать файл из workspace с нумерацией строк.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Относительный путь файла",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Начальная строка (с 1)",
                        "default": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Конечная строка",
                        "default": 250,
                    },
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_outline",
            "description": (
                "Компактный outline .py файла через ast: классы, функции "
                "и номера строк. Для быстрого изучения больших файлов без полного read_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Относительный путь .py файла",
                    },
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Создать или перезаписать файл в workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Точечная замена уникального фрагмента текста в файле.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["filename", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_lines",
            "description": (
                "Заменить диапазон строк по номерам (1-based, включительно). "
                "Удобнее edit_file для локальных моделей после read_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "start_line": {
                        "type": "integer",
                        "description": "Первая строка диапазона (с 1)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Последняя строка диапазона (включительно)",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Новый текст вместо указанного диапазона",
                    },
                },
                "required": ["filename", "start_line", "end_line", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_files",
            "description": "Поиск подстроки по всем текстовым файлам workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Запустить .py файл в workspace (таймаут 15 сек). "
                "Возвращает JSON: returncode, stdout, stderr."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "args": {
                        "type": "string",
                        "description": "Аргументы командной строки",
                        "default": "",
                    },
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pip_install",
            "description": "Установить пакет через pip текущего .venv.",
            "parameters": {
                "type": "object",
                "properties": {
                    "package": {
                        "type": "string",
                        "description": "Имя пакета (буквы, цифры, ., _, -)",
                    },
                },
                "required": ["package"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sqlite_query",
            "description": (
                "Read-only SQL (один SELECT/PRAGMA без ';') к SQLite-базе в workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "db_filename": {"type": "string"},
                    "sql": {"type": "string"},
                },
                "required": ["db_filename", "sql"],
            },
        },
    },
]

_DISPATCH: dict[str, Callable[..., str]] = {
    "list_files": list_files,
    "read_file": read_file,
    "get_file_outline": get_file_outline,
    "write_file": write_file,
    "edit_file": edit_file,
    "replace_lines": replace_lines,
    "search_in_files": search_in_files,
    "run_python": run_python,
    "pip_install": pip_install,
    "sqlite_query": sqlite_query,
}

def filter_tools_schema(allowed: set[str] | list[str]) -> list[dict[str, Any]]:
    """Подмножество TOOLS_SCHEMA по именам инструментов."""
    allowed_set = set(allowed)
    return [
        t for t in TOOLS_SCHEMA
        if t.get("function", {}).get("name") in allowed_set
    ]

def execute_tool(name: str, args_json: str | dict[str, Any]) -> str:
    """Диспетчер: вызвать инструмент по имени с JSON-аргументами."""
    if name not in _DISPATCH:
        return (
            f"Ошибка: неизвестный инструмент {name!r}. "
            f"Доступны: {', '.join(_DISPATCH)}"
        )
    if isinstance(args_json, str):
        try:
            args = json.loads(args_json) if args_json.strip() else {}
        except json.JSONDecodeError as exc:
            return f"Ошибка разбора аргументов JSON: {exc}"
    else:
        args = dict(args_json or {})
    try:
        return _DISPATCH[name](**args)
    except TypeError as exc:
        return f"Ошибка аргументов для {name}: {exc}"
    except (ValueError, PermissionError, OSError) as exc:
        return f"Ошибка {name}: {exc}"
