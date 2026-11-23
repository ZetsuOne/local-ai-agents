"""Мультиагентный пайплайн без фреймворков: Архитектор → Кодер ↔ Ревьюер."""

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
from tools import (
    apply_cli_workspace,
    execute_tool,
    filter_tools_schema,
    get_workspace,
)

DEFAULT_MAX_STEPS = 10
MAX_REVIEW_CYCLES = 2
PLAN_FILE = "plan.md"
SOLUTION_FILE = "solution.py"
TEST_FILE = "test_solution.py"

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BLUE = "\033[34m"

ARCHITECT_TOOLS = {
    "list_files",
    "read_file",
    "get_file_outline",
    "write_file",
    "replace_lines",
}
CODER_TOOLS = {
    "list_files",
    "read_file",
    "get_file_outline",
    "write_file",
    "replace_lines",
    "edit_file",
    "run_python",
    "pip_install",
}
REVIEWER_TOOLS = {
    "list_files",
    "read_file",
    "get_file_outline",
    "write_file",
    "replace_lines",
    "run_python",
}

ARCHITECT_PROMPT = f"""Ты Архитектор в мультиагентной системе.
Работай ТОЛЬКО в активном workspace. Твои инструменты: list_files, read_file, write_file, replace_lines.

Задача: проанализировать запрос пользователя и записать техническое задание в файл {PLAN_FILE}.
В плане укажи:
- цель и ограничения;
- структуру файлов (минимум {SOLUTION_FILE});
- публичный API / ожидаемый stdout;
- крайние случаи для тестов.

Не пиши production-код решения — только план/ТЗ в {PLAN_FILE}.
Когда план готов и записан — дай краткий финальный ответ без вызова тулов.
"""

CODER_PROMPT = f"""Ты Кодер в мультиагентной системе.
Работай ТОЛЬКО в активном workspace. Инструменты: list_files, read_file, write_file, replace_lines, edit_file, run_python, pip_install.

Правила:
1. Сначала прочитай {PLAN_FILE}.
2. Реализуй решение в {SOLUTION_FILE} (и доп. файлы по плану при необходимости).
3. Для правок предпочитай replace_lines по номерам строк после read_file.
4. Если инструмент записи вернул ⚠ SyntaxError — сразу исправь синтаксис.
5. Обязательно прогони run_python на {SOLUTION_FILE}.
6. Если пришёл баг-репорт от Ревьюера — исправь код и снова запусти.
Когда код готов и проверен запуском — финальный ответ без тулов.
"""

REVIEWER_PROMPT = f"""Ты Ревьюер-тестировщик в мультиагентной системе.
Работай ТОЛЬКО в активном workspace. Инструменты: list_files, read_file, write_file, replace_lines, run_python.

Правила:
1. Прочитай {PLAN_FILE} и {SOLUTION_FILE}.
2. Напиши отдельный тестовый скрипт {TEST_FILE} с проверками крайних случаев
   (assert / явные raise при провале). Тесты должны импортировать или вызывать решение.
3. Запусти {TEST_FILE} через run_python.
4. Если тесты упали — в ФИНАЛЬНОМ ответе начни строкой BUG_REPORT: и опиши баги для Кодера.
5. Если тесты прошли — финальный ответ начни строкой PASS: и кратко резюмируй.
Не правь {SOLUTION_FILE} сам — только тесты и отчёт.
"""


def _short(text: str, limit: int = 400) -> str:
    text = text.replace("\r\n", "\n")
    if len(text) <= limit:
        return text
    return text[:limit] + f"… (+{len(text) - limit} символов)"


def _parse_tool_args(raw: str | None) -> dict[str, Any]:
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def run_specialist(
    *,
    name: str,
    system_prompt: str,
    user_message: str,
    allowed_tools: set[str],
    client: Any,
    model: str,
    max_steps: int,
) -> str:
    """Один специализированный ReAct-агент с ограниченным набором тулов."""
    schema = filter_tools_schema(allowed_tools)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    last_call: tuple[str, str] | None = None
    final_text = ""

    print(f"\n{BOLD}{MAGENTA}▶ Агент: {name}{RESET}")
    print(f"{DIM}  tools={sorted(allowed_tools)}{RESET}")
    events.emit("agent_role", name=name, tools=sorted(allowed_tools))

    for step in range(1, max_steps + 1):
        if events.should_stop():
            return "STOPPED"
        print(f"\n{BOLD}{BLUE}── {name} шаг {step}/{max_steps} ──{RESET}")
        events.emit("step", step=step, max_steps=max_steps, agent=name)
        messages = compact_messages(messages)
        try:
            resp = chat_completion(
                client,
                model=model,
                messages=messages,
                tools=schema,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{RED}API error ({name}):{RESET} {exc}")
            events.emit("error", message=str(exc), agent=name)
            return f"ERROR: {exc}"

        log_token_usage(resp)
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []
        if not tool_calls:
            # Не дублируем Thought + Final
            final_text = (msg.content or "").strip() or "(пустой ответ)"
            print(f"{GREEN}🏁 {name} done:{RESET} {_short(final_text, 500)}")
            events.emit("final", content=final_text, agent=name)
            return final_text

        if msg.content and msg.content.strip():
            print(f"{CYAN}💭 Thought:{RESET} {_short(msg.content.strip(), 400)}")
            events.emit("thought", content=msg.content.strip(), agent=name)

        messages.append(
            {
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
        )

        for tc in tool_calls:
            if events.should_stop():
                return "STOPPED"
            tname = tc.function.name
            if tname not in allowed_tools:
                result = f"Ошибка: инструмент {tname!r} запрещён для роли {name}"
                print(f"{YELLOW}🔧 Tool:{RESET} {tname} (запрещён)")
                events.emit("tool_call", name=tname, args={}, agent=name, denied=True)
            else:
                args = _parse_tool_args(tc.function.arguments)
                print(f"{YELLOW}🔧 Tool:{RESET} {BOLD}{tname}{RESET}")
                print(
                    f"{DIM}   args:{RESET} "
                    f"{_short(json.dumps(args, ensure_ascii=False), 280)}"
                )
                events.emit("tool_call", name=tname, args=args, agent=name)
                result = execute_tool(tname, args)
                result, last_call = apply_loop_guard(tname, args, result, last_call)
            print(f"{GREEN}✅ Result:{RESET} {_short(result, 400)}")
            events.emit("tool_result", result=result, agent=name)
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )

    print(f"{RED}✗ {name}: лимит шагов исчерпан{RESET}")
    return final_text or f"(лимит шагов {max_steps} для {name})"


def _is_pass(review_text: str) -> bool:
    upper = review_text.strip().upper()
    return upper.startswith("PASS:") or upper.startswith("PASS ")


def _is_bug_report(review_text: str) -> bool:
    upper = review_text.strip().upper()
    return upper.startswith("BUG_REPORT:") or "BUG_REPORT" in upper[:40]


def run_pipeline(
    task: str,
    model: str,
    base_url: str,
    api_key: str,
    max_steps: int,
) -> int:
    client = make_client(base_url=base_url, api_key=api_key)
    ws = get_workspace()
    print(f"{BOLD}Multi-Agent Pipeline{RESET}")
    print(f"  model={model}")
    print(f"  max_steps/agent={max_steps}")
    print(f"  review_cycles≤{MAX_REVIEW_CYCLES}")
    print(f"  task={task}")
    print(f"  workspace={ws}")
    events.emit(
        "run_start",
        mode="multi_agent",
        model=model,
        workspace=str(ws),
        task=task,
        max_steps=max_steps,
    )

    # 1) Архитектор
    arch_out = run_specialist(
        name="Архитектор",
        system_prompt=ARCHITECT_PROMPT,
        user_message=f"Задача пользователя:\n{task}",
        allowed_tools=ARCHITECT_TOOLS,
        client=client,
        model=model,
        max_steps=max_steps,
    )
    if events.should_stop() or arch_out == "STOPPED":
        events.emit("run_end", status="stopped", code=130)
        return 130

    plan_path = ws / PLAN_FILE
    if not plan_path.is_file():
        print(f"{RED}✗ Архитектор не создал {PLAN_FILE}{RESET}")
        print(f"  Ответ: {_short(arch_out, 300)}")
        events.emit("run_end", status="failed", code=1)
        return 1

    bug_report: str | None = None
    review_out = ""

    for cycle in range(1, MAX_REVIEW_CYCLES + 1):
        if events.should_stop():
            events.emit("run_end", status="stopped", code=130)
            return 130
        print(f"\n{BOLD}═══ Цикл разработка/ревью {cycle}/{MAX_REVIEW_CYCLES} ═══{RESET}")

        # 2) Кодер
        coder_msg = (
            f"Исходная задача:\n{task}\n\n"
            f"Следуй плану в {PLAN_FILE}. Реализуй {SOLUTION_FILE}."
        )
        if bug_report:
            coder_msg += (
                f"\n\nБаг-репорт от Ревьюера (исправь и перепроверь):\n{bug_report}"
            )

        coder_out = run_specialist(
            name="Кодер",
            system_prompt=CODER_PROMPT,
            user_message=coder_msg,
            allowed_tools=CODER_TOOLS,
            client=client,
            model=model,
            max_steps=max_steps,
        )
        if events.should_stop() or coder_out == "STOPPED":
            events.emit("run_end", status="stopped", code=130)
            return 130

        if not (ws / SOLUTION_FILE).is_file():
            print(f"{RED}✗ Кодер не создал {SOLUTION_FILE}{RESET}")
            events.emit("run_end", status="failed", code=1)
            return 1

        # 3) Ревьюер
        review_out = run_specialist(
            name="Ревьюер",
            system_prompt=REVIEWER_PROMPT,
            user_message=(
                f"Исходная задача:\n{task}\n\n"
                f"Проверь {SOLUTION_FILE} по {PLAN_FILE}. "
                f"Напиши и запусти {TEST_FILE}."
            ),
            allowed_tools=REVIEWER_TOOLS,
            client=client,
            model=model,
            max_steps=max_steps,
        )
        if events.should_stop() or review_out == "STOPPED":
            events.emit("run_end", status="stopped", code=130)
            return 130

        if _is_pass(review_out):
            print(f"\n{BOLD}{GREEN}✓ Пайплайн успешен (цикл {cycle}){RESET}")
            print(f"  plan: {plan_path}")
            print(f"  solution: {ws / SOLUTION_FILE}")
            print(f"  tests: {ws / TEST_FILE}")
            events.emit("run_end", status="ok", code=0)
            return 0

        if _is_bug_report(review_out):
            bug_report = review_out
            print(f"{YELLOW}↻ Баг-репорт → Кодер (цикл {cycle}){RESET}")
            continue

        # Неясный вердикт — считаем как баг-репорт, чтобы дать шанс доработке
        bug_report = f"BUG_REPORT: неясный вердикт ревьюера:\n{review_out}"
        print(f"{YELLOW}↻ Неясный вердикт → ещё одна итерация{RESET}")

    print(f"\n{RED}✗ Тесты не прошли за {MAX_REVIEW_CYCLES} циклов{RESET}")
    print(f"  Последний отчёт: {_short(review_out, 600)}")
    events.emit("run_end", status="failed", code=1)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Мультиагентный пайплайн Архитектор→Кодер↔Ревьюер (LM Studio)",
    )
    parser.add_argument("--task", required=True, help="Задача для пайплайна")
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
        help="Рабочая директория (по умолчанию ./sandbox)",
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

    sys.exit(
        run_pipeline(
            task=args.task,
            model=model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_steps=args.max_steps,
        )
    )


if __name__ == "__main__":
    main()
