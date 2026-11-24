"""Детерминированный Workflow: Evaluator-Optimizer (генератор → запуск → критик)."""

from __future__ import annotations

import argparse
import json
import re
import sys

import events
from agent import apply_cli_workspace, get_workspace, run_command, write_file
from llm import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    chat_completion,
    log_token_usage,
    make_client,
    resolve_model,
)

SCRIPT_NAME = "solution.py"
MAX_FIX_CYCLES = 3


def _chat(client, model: str, messages: list[dict], temperature: float = 0.2) -> str:
    resp = chat_completion(
        client,
        model=model,
        messages=messages,
        temperature=temperature,
    )
    log_token_usage(resp)
    content = (resp.choices[0].message.content or "").strip()
    if content:
        events.emit("thought", content=content)
    return content


def _extract_code(text: str) -> str:
    """Достать Python-код из ответа модели (```python ... ``` или сырой текст)."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip() + "\n"
    return text.strip() + "\n"


def _extract_json(text: str) -> dict:
    """Достать JSON-объект из ответа критика."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"ok": False, "reason": "критик не вернул JSON", "fixed_code": ""}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"ok": False, "reason": "невалидный JSON от критика", "fixed_code": ""}


def step_generate(client, model: str, task: str) -> str:
    print("\n=== Шаг 1: Генератор ===")
    events.emit("step", step=1, label="Генератор")
    messages = [
        {
            "role": "system",
            "content": (
                "Ты пишешь чистый, рабочий Python-скрипт по задаче. "
                "Верни ТОЛЬКО код (можно в ```python блоке). "
                "Без объяснений. Скрипт должен быть самодостаточным и печатать результат в stdout."
            ),
        },
        {"role": "user", "content": f"Задача:\n{task}"},
    ]
    raw = _chat(client, model, messages)
    code = _extract_code(raw)
    result = write_file(SCRIPT_NAME, code)
    print(f"  → {result}")
    print(f"  → файл: {get_workspace() / SCRIPT_NAME}")
    events.emit("tool_result", result=result)
    return code


def step_run() -> dict:
    print("\n=== Шаг 2: Запуск ===")
    events.emit("step", step=2, label="Запуск")
    raw = run_command(f"python {SCRIPT_NAME}")
    data = json.loads(raw)
    print(f"  returncode: {data.get('returncode')}")
    if data.get("stdout"):
        print(f"  stdout:\n{data['stdout'][:800]}")
    if data.get("stderr"):
        print(f"  stderr:\n{data['stderr'][:800]}")
    if data.get("error"):
        print(f"  error: {data['error']}")
    return data


def step_critique(client, model: str, task: str, code: str, run_result: dict) -> dict:
    print("\n=== Шаг 3: Критик ===")
    events.emit("step", step=3, label="Критик")
    messages = [
        {
            "role": "system",
            "content": (
                "Ты критик кода в цикле Evaluator-Optimizer. "
                "Проанализируй код и результат запуска относительно задачи. "
                "Ответь СТРОГО JSON-объектом без markdown:\n"
                '{"ok": true/false, "reason": "...", "fixed_code": "..."}\n'
                "Если ok=true — багов нет, fixed_code пустой. "
                "Если ok=false — верни полный исправленный Python-код в fixed_code."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Задача:\n{task}\n\n"
                f"Код ({SCRIPT_NAME}):\n```python\n{code}\n```\n\n"
                f"Результат запуска:\n{json.dumps(run_result, ensure_ascii=False, indent=2)}"
            ),
        },
    ]
    raw = _chat(client, model, messages, temperature=0.1)
    verdict = _extract_json(raw)
    print(f"  ok={verdict.get('ok')} | {verdict.get('reason', '')[:200]}")
    return verdict


def _apply_fix(verdict: dict, code: str, client, model: str, task: str) -> str:
    """Применить правку: fixed_code из критика → иначе полная перегенерация."""
    fixed = (verdict.get("fixed_code") or "").strip()
    if fixed:
        code = _extract_code(fixed) if "```" in fixed else (fixed if fixed.endswith("\n") else fixed + "\n")
        result = write_file(SCRIPT_NAME, code)
        print(f"  write_file: {result}")
        events.emit("tool_result", result=result)
        return code

    print("  → критик не дал fixed_code — перегенерация")
    messages = [
        {
            "role": "system",
            "content": (
                "Исправь Python-скрипт по замечанию критика. "
                "Верни ТОЛЬКО полный исправленный код."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Задача:\n{task}\n\nКод:\n```python\n{code}\n```\n\n"
                f"Замечание: {verdict.get('reason')}"
            ),
        },
    ]
    code = _extract_code(_chat(client, model, messages))
    write_file(SCRIPT_NAME, code)
    return code


def run_workflow(task: str, model: str, base_url: str, api_key: str) -> int:
    client = make_client(base_url=base_url, api_key=api_key)
    ws = get_workspace()
    print("Workflow Evaluator-Optimizer")
    print(f"  model={model}")
    print(f"  task={task}")
    print(f"  workspace={ws}")
    events.emit(
        "run_start",
        mode="workflow",
        model=model,
        workspace=str(ws),
        task=task,
        max_steps=MAX_FIX_CYCLES + 2,
    )

    code = step_generate(client, model, task)
    if events.should_stop():
        events.emit("run_end", status="stopped", code=130)
        return 130
    run_result = step_run()

    for cycle in range(1, MAX_FIX_CYCLES + 1):
        if events.should_stop():
            events.emit("run_end", status="stopped", code=130)
            return 130
        verdict = step_critique(client, model, task, code, run_result)
        if verdict.get("ok"):
            print(f"\n✓ Готово за цикл критики #{cycle - 1 if cycle > 1 else 0} (принято)")
            print(f"  Итоговый скрипт: {ws / SCRIPT_NAME}")
            events.emit("final", content=f"Принято. Скрипт: {ws / SCRIPT_NAME}")
            events.emit("run_end", status="ok", code=0)
            return 0

        print(f"\n--- Правка цикл {cycle}/{MAX_FIX_CYCLES} ---")
        code = _apply_fix(verdict, code, client, model, task)
        run_result = step_run()

    final = step_critique(client, model, task, code, run_result)
    if final.get("ok") and run_result.get("returncode") == 0:
        print("\n✓ Принято после максимального числа правок")
        events.emit("final", content="Принято после максимального числа правок")
        events.emit("run_end", status="ok", code=0)
        return 0

    print(f"\n✗ Не удалось полностью исправить за {MAX_FIX_CYCLES} циклов")
    print(f"  Последняя причина: {final.get('reason', '')}")
    events.emit("run_end", status="failed", code=1)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Workflow Evaluator-Optimizer для локального LM Studio",
    )
    parser.add_argument("--task", required=True, help="Текстовая задача для генератора")
    parser.add_argument(
        "--model",
        default=None,
        help="Имя модели в LM Studio (по умолчанию — первая загруженная чат-модель)",
    )
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

    sys.exit(run_workflow(args.task, model, args.base_url, args.api_key))


if __name__ == "__main__":
    main()
