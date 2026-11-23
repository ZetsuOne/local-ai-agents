"""Общие хелперы LLM: клиент, шлюз, retry, логирование, сжатие контекста."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "lm-studio")
DEFAULT_MODEL = os.environ.get("LLM_MODEL") or None
DEFAULT_TIMEOUT = 300.0

TOOL_COMPACT_LIMIT = 250
TOOL_KEEP_RECENT = 3

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
        # Файл — основной канал; консоль/SSE уже дают живой вывод агентов
        fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # Retry/ошибки LLM — ещё и в консоль
        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    _logger = logger
    return logger


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
                # не раздувать лог огромными diff/result
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
    """Первая загруженная чат-модель (без embedding)."""
    models = [
        m.id
        for m in client.models.list().data
        if "embed" not in m.id.lower()
    ]
    if not models:
        raise RuntimeError("В шлюзе не найдена ни одна чат-модель (models.list пуст)")
    return models[0]


def resolve_model(client: OpenAI, model: str | None = None) -> str:
    """Приоритет: явный аргумент → LLM_MODEL → автоопределение через API."""
    if model and str(model).strip():
        return str(model).strip()
    if DEFAULT_MODEL and str(DEFAULT_MODEL).strip():
        return str(DEFAULT_MODEL).strip()
    return get_active_model(client)


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
    """Обёртка chat.completions.create с retry (до 3 повторов: 1с, 2с, 4с).

    Повторяет при сетевых сбоях и HTTP 429/502/503.
    """
    log = get_logger()
    last_exc: BaseException | None = None
    attempts = 1 + len(_RETRY_DELAYS)

    for attempt in range(attempts):
        try:
            return client.chat.completions.create(**kwargs)
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


def compact_messages(
    messages: list[dict[str, Any]],
    keep_recent: int = TOOL_KEEP_RECENT,
    limit: int = TOOL_COMPACT_LIMIT,
) -> list[dict[str, Any]]:
    """Обрезать старые role=tool сообщения, оставив keep_recent свежих целиком."""
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if len(tool_indices) <= keep_recent:
        return messages

    keep = set(tool_indices[-keep_recent:])
    compacted: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
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
    return compacted


def log_token_usage(resp: Any, prefix: str = "  tokens") -> None:
    """Печать prompt_tokens / completion_tokens, если шлюз вернул usage."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    if prompt is None and completion is None:
        return
    print(f"{prefix}: prompt={prompt} completion={completion}")
    get_logger().info("tokens prompt=%s completion=%s", prompt, completion)
    try:
        import events

        events.emit(
            "tokens",
            prompt_tokens=prompt,
            completion_tokens=completion,
        )
    except Exception:  # noqa: BLE001
        pass


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
            import json

            data = json.loads(result)
            if isinstance(data, dict) and data.get("returncode", 0) not in (0, None):
                return True
        except (json.JSONDecodeError, TypeError):
            pass
    return False


LOOP_WARNING = (
    "\n\n⚠ ПРЕДУПРЕЖДЕНИЕ: тот же инструмент с теми же аргументами уже вызывался "
    "и снова вернул ошибку. Смени подход: перечитай файл (read_file), "
    "используй replace_lines по номерам строк или другой инструмент."
)


def apply_loop_guard(
    name: str,
    args: dict[str, Any],
    result: str,
    last_call: tuple[str, str] | None,
) -> tuple[str, tuple[str, str]]:
    """Если повторный идентичный вызов снова ошибка — дописать предупреждение.

    Возвращает (возможно дополненный result, новый last_call=(name, args_key)).
    """
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
