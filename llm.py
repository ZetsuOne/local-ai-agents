"""Общие хелперы LLM: клиент, шлюз, retry, логирование, сжатие контекста."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "lm-studio")
DEFAULT_MODEL = os.environ.get("LLM_MODEL") or None
DEFAULT_TIMEOUT = 300.0

try:
    DEFAULT_CONTEXT_LIMIT = max(1024, int(os.environ.get("LLM_CONTEXT_LIMIT", "32768")))
except (TypeError, ValueError):
    DEFAULT_CONTEXT_LIMIT = 32768

TOOL_COMPACT_LIMIT = 250
TOOL_KEEP_RECENT = 4
WORKING_MEMORY_MARKER = "[FILES MODIFIED & LAST TEST STATUS]"

_THINK_RE = re.compile(
    r"<think(?:ing)?>(.*?)</think(?:ing)?>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_JSON_FENCE_RE = re.compile(
    r"```(?:json|tool_call|tool)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)
_BALANCED_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)

_RETRY_DELAYS = (1.0, 2.0, 4.0)
_RETRY_STATUS = {429, 502, 503}

_LOG_FILE = Path(__file__).resolve().parent / "agent_runs.log"
_logger: logging.Logger | None = None


def setup_logging() -> logging.Logger:
    """Стандартный logging: консоль + файл agent_runs.log."""
    global _logger
    if _logger is not None:
        return _logger

    logger = logging.getLogger("local_ai_agents")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    _logger = logger
    return _logger


def get_logger() -> logging.Logger:
    return setup_logging()


# Инициализация при импорте + подписка на шину событий
setup_logging()

def _attach_event_logging() -> None:
    try:
        import events

        def _on_event(event: dict[str, Any]) -> None:
            et = event.get("type", "?")
            parts = [f"{k}={v!r}" for k, v in event.items() if k not in ("type", "ts")]
            msg = f"event:{et}"
            if parts:
                clipped: list[str] = []
                for p in parts:
                    if len(p) > 400:
                        clipped.append(p[:400] + "…")
                    else:
                        clipped.append(p)
                msg += " " + " ".join(clipped)
            get_logger().info(msg)

        events.subscribe(_on_event)
    except Exception:  # noqa: BLE001
        pass


_attach_event_logging()


def make_client(
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> OpenAI:
    """OpenAI-совместимый клиент (LM Studio / OpenRouter / любой шлюз)."""
    return OpenAI(
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=api_key or DEFAULT_API_KEY,
        timeout=timeout,
    )


def get_active_model(client: OpenAI) -> str:
    """Первая чат-модель со state=loaded (LM Studio); иначе fallback без embed."""
    data = list(client.models.list().data)
    chat = [m for m in data if "embed" not in (m.id or "").lower()]
    loaded = [m.id for m in chat if getattr(m, "state", None) == "loaded"]
    if loaded:
        return loaded[0]
    # Non-LM Studio OpenAI-compatible gateways usually omit `state`
    if chat:
        return chat[0].id
    raise RuntimeError("В шлюзе не найдена ни одна чат-модель (models.list пуст)")


def resolve_model(client: OpenAI, model: str | None = None) -> str:
    """Приоритет: явный аргумент → LLM_MODEL → автоопределение через API."""
    if model and str(model).strip():
        return str(model).strip()
    if DEFAULT_MODEL and str(DEFAULT_MODEL).strip():
        return str(DEFAULT_MODEL).strip()
    return get_active_model(client)


def split_think_content(content: str | None) -> tuple[str, str]:
    """Отделить ``<think>…</think>`` / ``<thinking>`` от видимого текста."""
    text = content or ""
    if not text:
        return "", ""
    chunks: list[str] = []
    for match in _THINK_RE.finditer(text):
        chunk = (match.group(1) or "").strip()
        if chunk:
            chunks.append(chunk)
    visible = _THINK_RE.sub("", text).strip()
    return "\n\n".join(chunks), visible


class _FallbackFn:
    __slots__ = ("name", "arguments")

    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FallbackToolCall:
    """OpenAI-совместимый объект tool_call, синтезированный из content."""

    __slots__ = ("id", "type", "function")

    def __init__(self, *, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.type = "function"
        self.function = _FallbackFn(name, arguments)


def _known_tool_names(known_names: set[str] | list[str] | None = None) -> set[str]:
    if known_names is not None:
        return {str(n) for n in known_names if n}
    try:
        from agent import BASIC_TOOLS_SCHEMA

        return {
            str(t.get("function", {}).get("name") or "")
            for t in BASIC_TOOLS_SCHEMA
            if t.get("function", {}).get("name")
        }
    except Exception:  # noqa: BLE001
        return {"list_files", "read_file", "write_file", "run_command"}


def _coerce_tool_args(raw: Any) -> str | None:
    """Нормализовать arguments в JSON-строку объекта; None если непригодно."""
    if raw is None:
        return "{}"
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return "{}"
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)
        return None
    return None


def _tool_call_from_obj(
    obj: Any,
    known: set[str],
    *,
    seen: set[tuple[str, str]],
) -> FallbackToolCall | None:
    if not isinstance(obj, dict):
        return None
    # распространённые формы локальных моделей
    name: Any = obj.get("name") or obj.get("tool")
    fn = obj.get("function")
    if not name and isinstance(fn, dict):
        name = fn.get("name")
    elif not name and isinstance(fn, str):
        name = fn
    name = str(name or "").strip()
    if not name or name not in known:
        return None

    args_raw = obj.get("arguments", obj.get("parameters", obj.get("args")))
    if args_raw is None and isinstance(obj.get("function"), dict):
        args_raw = obj["function"].get("arguments")
    # иногда вся полезная нагрузка — сам объект без arguments
    if args_raw is None:
        skip = {"name", "tool", "function", "id", "type"}
        residual = {k: v for k, v in obj.items() if k not in skip}
        args_raw = residual if residual else {}

    args_json = _coerce_tool_args(args_raw)
    if args_json is None:
        return None
    key = (name, args_json)
    if key in seen:
        return None
    seen.add(key)
    call_id = str(obj.get("id") or f"fallback_{uuid.uuid4().hex[:12]}")
    return FallbackToolCall(call_id=call_id, name=name, arguments=args_json)


def _iter_json_candidates(content: str) -> list[str]:
    """Кандидаты JSON из тегов, fence-блоков и сырых объектов."""
    chunks: list[str] = []
    for match in _TOOL_CALL_TAG_RE.finditer(content):
        chunks.append(match.group(1).strip())
    for match in _JSON_FENCE_RE.finditer(content):
        chunks.append(match.group(1).strip())
    # сырые объекты — только если похожи на tool call (есть "name")
    for match in _BALANCED_OBJECT_RE.finditer(content):
        blob = match.group(0).strip()
        if '"name"' in blob or "'name'" in blob or '"tool"' in blob:
            chunks.append(blob)
    return chunks


def parse_fallback_tool_calls(
    content: str | None,
    known_names: set[str] | list[str] | None = None,
) -> list[FallbackToolCall]:
    """Извлечь tool calls из ``message.content``, если structured tool_calls пуст."""
    text = (content or "").strip()
    if not text:
        return []
    known = _known_tool_names(known_names)
    if not known:
        return []

    found: list[FallbackToolCall] = []
    seen: set[tuple[str, str]] = set()

    for chunk in _iter_json_candidates(text):
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        items: list[Any]
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # {"tool_calls":[…]} или одиночный объект
            nested = data.get("tool_calls")
            if isinstance(nested, list):
                items = nested
            else:
                items = [data]
        else:
            continue
        for item in items:
            tc = _tool_call_from_obj(item, known, seen=seen)
            if tc is not None:
                found.append(tc)

    return found


def normalize_tool_calls(
    message: Any,
    known_names: set[str] | list[str] | None = None,
) -> list[Any]:
    """Вернуть structured tool_calls либо fallback-парсинг из content."""
    structured = getattr(message, "tool_calls", None) or []
    if structured:
        return list(structured)

    content = getattr(message, "content", None) or ""
    fallback = parse_fallback_tool_calls(content, known_names=known_names)
    if fallback:
        names = [tc.function.name for tc in fallback]
        get_logger().info(
            "fallback tool-call parser: extracted %s from message.content (%s)",
            len(fallback),
            ", ".join(names),
        )
        print(
            f"  ⚠ fallback tool-call parser: {len(fallback)} call(s) "
            f"from content → {', '.join(names)}"
        )
    return fallback


def strip_tool_call_markup(content: str | None) -> str:
    """Убрать ``<tool_call>`` / json-fence tool payload из assistant content."""
    text = content or ""
    if not text:
        return ""
    text = _TOOL_CALL_TAG_RE.sub("", text)
    text = _JSON_FENCE_RE.sub("", text)
    return text.strip()


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return status in _RETRY_STATUS
    # сетевые сбои без обёртки openai
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return False


def chat_completion(client: OpenAI, **kwargs: Any) -> Any:
    """Обёртка chat.completions.create с retry (до 3 повторов: 1с, 2с, 4с)."""
    log = get_logger()
    last_exc: BaseException | None = None
    attempts = 1 + len(_RETRY_DELAYS)

    for attempt in range(attempts):
        started = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - started
            try:
                setattr(resp, "_elapsed_sec", elapsed)
            except Exception:  # noqa: BLE001
                pass
            return resp
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= len(_RETRY_DELAYS) or not _is_retryable(exc):
                log.error("LLM request failed (attempt %s/%s): %s", attempt + 1, attempts, exc)
                raise
            delay = _RETRY_DELAYS[attempt]
            log.warning(
                "LLM retryable error (attempt %s/%s): %s — sleep %.0fs",
                attempt + 1,
                attempts,
                exc,
                delay,
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


def _extract_working_memory(messages: list[dict[str, Any]]) -> str:
    """Детерминированный summary: изменённые файлы + последний статус тестов."""
    import json
    import re

    write_tools = ("write_file",)
    files: list[str] = []
    call_meta: dict[str, tuple[str, dict[str, Any]]] = {}  # id → (name, args)
    last_test = "(тесты ещё не запускались)"

    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = (
                        json.loads(raw_args)
                        if isinstance(raw_args, str)
                        else dict(raw_args or {})
                    )
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tid = tc.get("id") or ""
                if tid:
                    call_meta[tid] = (name, args)
                if name in write_tools:
                    fname = (args.get("path") or args.get("filename") or "").strip()
                    if not fname:
                        continue
                    if fname in files:
                        files.remove(fname)
                    files.append(fname)

        if msg.get("role") != "tool":
            continue
        tid = msg.get("tool_call_id") or ""
        name, args = call_meta.get(tid, ("", {}))
        content = msg.get("content") or ""
        if name != "run_command":
            continue
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict) or "returncode" not in data:
            continue
        cmd = str(data.get("cmd") or args.get("cmd") or "").lower()
        if "pytest" not in cmd and "unittest" not in cmd:
            continue
        rc = data.get("returncode")
        ok = rc in (0, None)
        tail = (data.get("stdout") or data.get("stderr") or "")
        tail = re.sub(r"\s+", " ", str(tail)).strip()[:180]
        last_test = (
            f"{'PASS' if ok else 'FAIL'} rc={rc} cmd={data.get('cmd') or args.get('cmd')}"
            + (f" — {tail}" if tail else "")
        )

    files_line = ", ".join(files) if files else "(нет)"
    return (
        f"{WORKING_MEMORY_MARKER}\n"
        f"files: {files_line}\n"
        f"last_test: {last_test}"
    )


def compact_messages(
    messages: list[dict[str, Any]],
    keep_recent: int = TOOL_KEEP_RECENT,
    limit: int = TOOL_COMPACT_LIMIT,
) -> list[dict[str, Any]]:
    """Сжать старые role=tool, оставив keep_recent целиком + working-memory summary."""
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= keep_recent:
        return messages

    summary = _extract_working_memory(messages)
    keep = set(tool_indices[-keep_recent:])
    compacted: list[dict[str, Any]] = []
    summary_inserted = False

    for i, msg in enumerate(messages):
        # Вставить summary сразу после system (или в начало, если system нет)
        if not summary_inserted:
            if msg.get("role") == "system":
                compacted.append(msg)
                compacted.append({"role": "system", "content": summary})
                summary_inserted = True
                continue
            compacted.append({"role": "system", "content": summary})
            summary_inserted = True

        # Пропускаем предыдущие working-memory блоки
        if (
            msg.get("role") == "system"
            and isinstance(msg.get("content"), str)
            and msg["content"].startswith(WORKING_MEMORY_MARKER)
        ):
            continue

        if msg.get("role") != "tool" or i in keep:
            compacted.append(msg)
            continue
        content = msg.get("content") or ""
        if len(content) <= limit:
            compacted.append(msg)
            continue
        new_msg = dict(msg)
        new_msg["content"] = content[:limit] + f"… [сжато, было {len(content)} символов]"
        compacted.append(new_msg)

    if not summary_inserted:
        compacted.insert(0, {"role": "system", "content": summary})

    return compacted


def log_token_usage(resp: Any, prefix: str = "  tokens") -> dict[str, Any]:
    """Печать prompt/completion tokens + tok/s; эмит SSE ``tokens``."""
    usage = getattr(resp, "usage", None)
    prompt = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion = getattr(usage, "completion_tokens", None) if usage is not None else None
    elapsed = getattr(resp, "_elapsed_sec", None)
    try:
        elapsed_f = float(elapsed) if elapsed is not None else None
    except (TypeError, ValueError):
        elapsed_f = None

    tokens_per_sec: float | None = None
    if (
        completion is not None
        and elapsed_f is not None
        and elapsed_f > 0
        and int(completion) > 0
    ):
        tokens_per_sec = round(int(completion) / elapsed_f, 2)

    context_limit = DEFAULT_CONTEXT_LIMIT
    context_pct: float | None = None
    if prompt is not None and context_limit > 0:
        context_pct = round(100.0 * int(prompt) / context_limit, 1)

    metrics: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "elapsed_sec": round(elapsed_f, 3) if elapsed_f is not None else None,
        "tokens_per_sec": tokens_per_sec,
        "context_limit": context_limit,
        "context_pct": context_pct,
    }

    if prompt is None and completion is None and elapsed_f is None:
        return metrics

    parts = []
    if prompt is not None or completion is not None:
        parts.append(f"prompt={prompt} completion={completion}")
    if context_pct is not None:
        parts.append(f"ctx={prompt}/{context_limit} ({context_pct}%)")
    if tokens_per_sec is not None:
        parts.append(f"tok/s={tokens_per_sec}")
    elif elapsed_f is not None:
        parts.append(f"elapsed={elapsed_f:.2f}s")
    line = f"{prefix}: " + " ".join(parts) if parts else prefix
    print(line)
    get_logger().info(
        "tokens prompt=%s completion=%s ctx=%s/%s (%.1f%%) elapsed_sec=%s tok/s=%s",
        prompt,
        completion,
        prompt,
        context_limit,
        context_pct if context_pct is not None else -1.0,
        metrics["elapsed_sec"],
        tokens_per_sec,
    )
    try:
        import events

        events.emit(
            "tokens",
            prompt_tokens=prompt,
            completion_tokens=completion,
            elapsed_sec=metrics["elapsed_sec"],
            tokens_per_sec=tokens_per_sec,
            context_limit=context_limit,
            context_pct=context_pct,
        )
    except Exception:  # noqa: BLE001
        pass
    return metrics


def is_tool_error(result: str) -> bool:
    """Эвристика: ответ инструмента — ошибка."""
    if not result:
        return False
    lower = result.lower()
    if result.startswith("Ошибка") or result.startswith("Error"):
        return True
    if '"error"' in lower or "'error'" in lower:
        return True
    if '"returncode"' in lower:
        try:

            data = json.loads(result)
            if isinstance(data, dict) and data.get("returncode", 0) not in (0, None):
                return True
        except (json.JSONDecodeError, TypeError):
            pass
    return False


LOOP_WARNING = (
    "\n\n⚠ ПРЕДУПРЕЖДЕНИЕ: тот же инструмент с теми же аргументами уже вызывался "
    "и снова вернул ошибку. Смени подход: перечитай файл (read_file) "
    "или перезапиши целиком через write_file."
)

def apply_loop_guard(
    name: str,
    args: dict[str, Any],
    result: str,
    last_call: tuple[str, str] | None,
) -> tuple[str, tuple[str, str]]:
    """Если повторный идентичный вызов снова ошибка — дописать предупреждение."""
    import json

    args_key = json.dumps(args, ensure_ascii=False, sort_keys=True)
    current = (name, args_key)
    if (
        last_call is not None
        and last_call == current
        and is_tool_error(result)
    ):
        result = result + LOOP_WARNING
    return result, current
