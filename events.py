"""Простая шина событий для CLI-логов и SSE Web-UI (только stdlib)."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

EventHandler = Callable[[dict[str, Any]], None]

_lock = threading.Lock()
_subscribers: list[EventHandler] = []
_stop_flag = threading.Event()

# Дедуп Thought: одинаковый content подряд (часто Thought == Final) не шлём дважды
_last_thought_key: str | None = None
_thought_lock = threading.Lock()


def subscribe(handler: EventHandler) -> EventHandler:
    with _lock:
        if handler not in _subscribers:
            _subscribers.append(handler)
    return handler


def unsubscribe(handler: EventHandler) -> None:
    with _lock:
        try:
            _subscribers.remove(handler)
        except ValueError:
            pass


def clear_subscribers() -> None:
    with _lock:
        _subscribers.clear()


def emit(event_type: str, **payload: Any) -> None:
    global _last_thought_key

    if event_type == "run_start":
        with _thought_lock:
            _last_thought_key = None

    if event_type == "thought":
        content = (payload.get("content") or "").strip()
        agent = payload.get("agent") or ""
        key = f"{agent}::{content}"
        with _thought_lock:
            if content and key == _last_thought_key:
                return
            _last_thought_key = key

    event: dict[str, Any] = {
        "type": event_type,
        "ts": time.time(),
        **payload,
    }
    with _lock:
        handlers = list(_subscribers)
    for handler in handlers:
        try:
            handler(event)
        except Exception:  # noqa: BLE001 — UI/логгер не должны ронять агента
            pass


def request_stop() -> None:
    _stop_flag.set()
    emit("stopped", message="Остановка запрошена")


def clear_stop() -> None:
    _stop_flag.clear()


def should_stop() -> bool:
    return _stop_flag.is_set()
