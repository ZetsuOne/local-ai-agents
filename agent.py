"""Автономный ReAct-агент c Tool Calling через LM Studio."""

from __future__ import annotations

import argparse
import json
import sys
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
    resolve_model,
)
from tools import TOOLS_SCHEMA, apply_cli_workspace, execute_tool, get_workspace

DEFAULT_MAX_STEPS = 12

# ANSI-цвета для консоли (Windows 10+ / современные терминалы)
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BLUE = "\033[34m"

SYSTEM_PROMPT = """Ты автономный coding-агент в режиме ReAct (Reason + Act).
У тебя есть инструменты для работы строго внутри активной рабочей директории (workspace).

Правила поведения:
1. Сначала исследуй окружение: list_files / search_in_files / get_file_outline / read_file — не гадай.
2. Для больших .py сначала get_file_outline, затем read_file нужных фрагментов.
3. Пиши новый код через write_file; правь существующий через replace_lines (по номерам строк после read_file) или edit_file (уникальный фрагмент).
4. Если инструмент записи вернул ⚠ SyntaxError — сразу исправь синтаксис, не вызывай run_python.
5. Перед финальным ответом ВСЕГДА проверяй код запуском run_python. Без успешного прогона не завершай задачу.
6. Для зависимостей используй pip_install (только безопасные имена пакетов).
7. Для данных в SQLite — только sqlite_query (один SELECT/PRAGMA без ';').
8. Думай кратко, действуй инструментами. Когда задача решена и проверена — дай финальный текстовый ответ без вызова тулов.
"""


def _short(text: str, limit: int = 400) -> str:
    text = text.replace("\r\n", "\n")
    if len(text) <= limit:
        return text
    return text[:limit] + f"… (+{len(text) - limit} символов)"


def _log_step(step: int, max_steps: int) -> None:
    print(f"\n{BOLD}{BLUE}════════ Шаг {step}/{max_steps} ════════{RESET}")
    events.emit("step", step=step, max_steps=max_steps)


def _log_thought(content: str | None) -> None:
    if content and content.strip():
        print(f"{CYAN}💭 Thought:{RESET} {_short(content.strip(), 500)}")
        events.emit("thought", content=content.strip())


def _log_tool_call(name: str, args: dict[str, Any]) -> None:
    pretty = json.dumps(args, ensure_ascii=False)
    print(f"{YELLOW}🔧 Tool:{RESET} {BOLD}{name}{RESET}")
    print(f"{DIM}   args:{RESET} {_short(pretty, 300)}")
    events.emit("tool_call", name=name, args=args)


def _log_tool_result(result: str) -> None:
    print(f"{GREEN}✅ Result:{RESET} {_short(result, 500)}")
    events.emit("tool_result", result=result)


def _log_final(content: str) -> None:
    print(f"\n{BOLD}{MAGENTA}🏁 Final answer:{RESET}\n{content}")
    events.emit("final", content=content)


def _parse_tool_args(raw: str | None) -> dict[str, Any]:
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _final_report_without_tools(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Один завершающий chat_completion без tools — итоговый отчёт пользователю."""
    messages = compact_messages(messages)
    resp = chat_completion(
        client,
        model=model,
        messages=messages,
        temperature=0.2,
    )
    log_token_usage(resp)
    msg = resp.choices[0].message
    final = (msg.content or "").strip() or "(пустой ответ модели)"
    messages.append({"role": "assistant", "content": final})
    return final, messages


def run_agent(
    task: str,
    model: str,
    base_url: str,
    api_key: str,
    max_steps: int,
    messages: list[dict[str, Any]] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Запустить ReAct-цикл.

    Если передан ``messages`` (история сеанса), новая задача добавляется как
    follow-up user-сообщение. Возвращает ``(exit_code, messages)``.
    """
    client = make_client(base_url=base_url, api_key=api_key)
    follow_up = bool(messages)
    if messages:
        messages = list(messages)
        messages.append({"role": "user", "content": task})
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
    last_call: tuple[str, str] | None = None
    last_was_tool = False
    ws = get_workspace()

    print(f"{BOLD}ReAct Agent{RESET}")
    print(f"  model={model}")
    print(f"  max_steps={max_steps}")
    print(f"  workspace={ws}")
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
    )

    for step in range(1, max_steps + 1):
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
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001 — учебно-боевой CLI
            print(f"{RED}API error:{RESET} {exc}")
            events.emit("error", message=str(exc))
            events.emit("run_end", status="error", code=2)
            return 2, messages

        log_token_usage(resp)
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            # Не дублируем Thought + Final с одним текстом
            final = (msg.content or "").strip() or "(пустой ответ модели)"
            messages.append({"role": "assistant", "content": final})
            _log_final(final)
            events.emit("run_end", status="ok", code=0)
            return 0, messages

        _log_thought(msg.content)
        last_was_tool = True

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
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

    # Лимит шагов: если последний шаг был tool-call — один финальный ответ без tools
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
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
