# local-ai-agents

Учебно-боевой стенд на чистом Python + `openai` для локальных моделей **12B–30B** в LM Studio и любых OpenAI-совместимых шлюзов (`LLM_BASE_URL`).

Три режима без LangChain / LlamaIndex / CrewAI + локальный Web-UI:

| Режим | Файл | Паттерн |
|-------|------|---------|
| Workflow | `workflow.py` | Evaluator-Optimizer (генератор → запуск → критик) |
| Agent | `agent.py` | Автономный ReAct + Tool Calling |
| Multi-Agent | `multi_agent.py` | Архитектор → Кодер ↔ Ревьюер |
| Web-UI | `ui.py` + `static/index.html` | Тёмный офлайн-интерфейс (SSE) |

## Установка

```powershell
cd C:\Projects\local-ai-agents
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

В LM Studio запусти локальный сервер (порт `1234`) и загрузи **одну** чат-модель (embedding-модели игнорируются).

## Быстрый старт (Web-UI)

Двойной клик по `start_ui.bat` — или:

```powershell
.\.venv\Scripts\python.exe ui.py
```

Откроется `http://localhost:8080` (флаг `--no-browser` отключает автозапуск браузера). В шапке: индикатор модели, режим (`Agent` / `Workflow` / `Multi-Agent`), `Workspace`, необязательное поле **API Base URL**, `Max steps`, чекбокс очистки sandbox, индикатор **Шаг X / max_steps**, кнопки **Запустить** / **Остановить** / **Новый сеанс**.

Центральная лента получает шаги по SSE (Thought, tool calls, unified diff `+`/`−`, карточка `run_python` со stdout/stderr без дубля сырого JSON, токены). Над полем задачи — три пресета шаблонов («Скрипт + тесты», «Рефакторинг файла», «Анализ SQLite»). В режиме **Agent** история `messages` сохраняется между запусками: после завершения можно отправить уточнение (follow-up) в том же контексте; **Новый сеанс** очищает историю.

Справа — дерево файлов workspace, просмотрщик с нумерацией строк и блок **Бэкапы** (список версий из `.agent_backup/` + кнопка **Откатить**). Ход работы также пишется в `agent_runs.log`.

## ИИ-шлюзы / OpenRouter

Клиент OpenAI-совместимый. Провайдер переключается переменными окружения (или полем **API Base URL** в UI / флагами `--base-url` / `--api-key`):

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `LLM_BASE_URL` | `http://localhost:1234/v1` | Базовый URL шлюза |
| `LLM_API_KEY` | `lm-studio` | API-ключ (для локального LM Studio достаточно заглушки) |
| `LLM_MODEL` | *(пусто)* | Явная модель; если не задана — `client.models.list()` |

```powershell
# Локальный LM Studio (по умолчанию)
.\.venv\Scripts\python.exe agent.py --task "..."

# OpenRouter
$env:LLM_BASE_URL = "https://openrouter.ai/api/v1"
$env:LLM_API_KEY = "sk-or-..."
$env:LLM_MODEL = "openai/gpt-4o-mini"
.\.venv\Scripts\python.exe agent.py --task "..."

# Или через флаги
.\.venv\Scripts\python.exe agent.py --base-url "https://openrouter.ai/api/v1" --api-key "sk-or-..." --model "openai/gpt-4o-mini" --task "..."
```

Запросы к API обёрнуты в retry: до 3 повторов с задержками 1с → 2с → 4с при сетевых сбоях и HTTP 429/502/503.

## Docker

Нужен Docker Desktop. LM Studio (или другой шлюз) должен слушать на хосте (`1234`). Контейнер ходит к нему через `host.docker.internal`.

```powershell
docker compose up --build
```

UI: `http://localhost:8080`. Каталог `./sandbox` проброшен в контейнер.

```powershell
# только образ
docker build -t local-ai-agents .
docker run --rm -p 8080:8080 --add-host=host.docker.internal:host-gateway -e LLM_BASE_URL=http://host.docker.internal:1234/v1 -v ${PWD}/sandbox:/app/sandbox local-ai-agents
```

## Структура

| Файл | Назначение |
|------|------------|
| `llm.py` | Клиент шлюза, env-конфиг, retry, logging → `agent_runs.log`, сжатие контекста, loop-guard |
| `tools.py` | Workspace + бэкапы/rollback + `ast.parse` / `get_file_outline` + 10 инструментов |
| `events.py` | Шина событий для CLI, SSE и дедупа Thought |
| `workflow.py` | Детерминированный цикл генератор → запуск → критик (до 3 правок) |
| `agent.py` | Автономный ReAct-агент с tool calling + финальный отчёт при max_steps |
| `multi_agent.py` | Пайплайн из 3 ролей с ограниченными тулами |
| `ui.py` | `ThreadingHTTPServer` + SSE API на порту `8080` (сессия, бэкапы) |
| `static/index.html` | Офлайн UI (без CDN) |
| `Dockerfile` / `docker-compose.yml` | Упаковка Web-UI |
| `start_ui.bat` | Запуск UI в один клик |

## Workspace, `--clean` и авто-бэкапы

Рабочая папка задаётся через `set_workspace()` / флаг `--workspace` (по умолчанию `./sandbox`). Все пути инструментов проверяются строго внутри активной `WORKSPACE_DIR`.

```powershell
.\.venv\Scripts\python.exe agent.py --workspace ./sandbox --task "..."
.\.venv\Scripts\python.exe workflow.py --workspace D:\my-project --task "..."
.\.venv\Scripts\python.exe multi_agent.py --clean --task "..."   # только для ./sandbox
```

- `--clean` очищает **только** дефолтный `./sandbox` перед задачей (для другого `--workspace` будет ошибка).
- Перед изменением существующего файла (`write_file` / `edit_file` / `replace_lines`) копия сохраняется в `<workspace>/.agent_backup/<YYYYMMDD_HHMMSS>_<имя>`.
- Папки `__pycache__`, `.git`, `.venv`, `.agent_backup` и файлы `*.pyc` скрыты из `list_files` / `search_in_files` (и из дерева UI). `.agent_backup` также в `.gitignore`.
- Откат из Web-UI: `GET /api/backups` + `POST /api/rollback` — текущая версия перед восстановлением снова уходит в бэкап.
- Для `.py` сразу после записи выполняется `ast.parse`; при `SyntaxError` инструмент возвращает номер строки и текст ошибки, чтобы модель исправила код без лишнего `run_python`.
- Изменения файлов в UI показываются как **unified diff** (`difflib`), строки `+` зелёные, `-` красные.

## Автоопределение модели

Хардкода имени модели нет. Если `--model` / `LLM_MODEL` не заданы, берётся первая чат-модель через `client.models.list()` (`get_active_model` в `llm.py`).

```powershell
# автоопределение
.\.venv\Scripts\python.exe agent.py --task "..."

# явно
.\.venv\Scripts\python.exe agent.py --model "твоя-модель" --task "..."
```

## 1. Workflow (`workflow.py`)

Детерминированный Evaluator-Optimizer: генератор пишет `solution.py` → запуск → критик правит через `replace_lines` / `edit_file` (до 3 циклов).

```powershell
.\.venv\Scripts\python.exe workflow.py --task "Напиши скрипт, который считает сумму чисел от 1 до 100 и печатает результат"
```

Опции: `--model`, `--workspace`, `--clean`, `--base-url`, `--api-key`.

## 2. Agent (`agent.py`)

Автономный ReAct-агент со всеми 10 инструментами. Перед каждым запросом к LLM сжимает старые `role=tool` (кроме 3 свежих) до 250 символов; при повторном идентичном ошибочном вызове тула добавляет предупреждение; печатает `prompt_tokens` / `completion_tokens`, если шлюз их вернул.

Если цикл упёрся в `max_steps` и последний шаг был вызовом инструмента, агент делает **один** завершающий `chat_completion` **без** `tools`, чтобы сформулировать итоговый отчёт пользователю.

```powershell
.\.venv\Scripts\python.exe agent.py --task "Создай hello.py, который печатает Hello, agents! и запусти его" --max-steps 12
```

Опции: `--model`, `--max-steps`, `--workspace`, `--clean`, `--base-url`, `--api-key`.

В Web-UI история `messages` сеанса Agent сохраняется: следующая задача идёт как follow-up; кнопка **Новый сеанс** (`POST /api/session/reset`) сбрасывает контекст.

## 3. Multi-Agent (`multi_agent.py`)

Три специалиста с разными системными промптами и урезанными тулами (до 2 циклов Кодер↔Ревьюер):

1. **Архитектор** — план/ТЗ в `plan.md` (`list_files`, `read_file`, `get_file_outline`, `write_file`, `replace_lines`)
2. **Кодер** — код по плану (`write_file` / `replace_lines` / `edit_file` + `run_python`)
3. **Ревьюер** — читает код, пишет `test_solution.py`, гоняет тесты; при падении возвращает `BUG_REPORT:` Кодеру

```powershell
.\.venv\Scripts\python.exe multi_agent.py --task "Напиши функцию is_palindrome(s) и скрипт, который проверяет несколько строк" --max-steps 10
```

Опции: `--model`, `--max-steps`, `--workspace`, `--clean`, `--base-url`, `--api-key`.

## Инструменты (10 шт., workspace)

Все пути ограничены активной рабочей директорией (`Path.resolve()` + проверка границ):

| Инструмент | Описание |
|------------|----------|
| `list_files` | Список файлов workspace (без `__pycache__` / `.git` / `.venv` / `.agent_backup` / `*.pyc`) |
| `read_file` | Чтение с нумерацией строк и заголовком `=== file (строки a-b из N) ===` |
| `get_file_outline` | Outline `.py` через `ast`: классы, функции, номера строк |
| `write_file` | Создание / перезапись + бэкап + `ast.parse` для `.py` |
| `edit_file` | Точечная замена уникального `old_text` (+ бэкап / синтаксис) |
| `replace_lines` | Замена диапазона строк по номерам (+ бэкап / синтаксис) |
| `search_in_files` | Поиск подстроки по текстовым файлам (те же исключения, что у `list_files`) |
| `run_python` | Запуск `.py` (таймаут 15 сек) → JSON returncode/stdout/stderr |
| `pip_install` | Установка пакета через текущий pip (безопасные имена) |
| `sqlite_query` | Read-only: URI `?mode=ro`, только один `SELECT`/`PRAGMA` без `;` |

## API Web-UI (дополнительно)

| Метод | Путь | Назначение |
|-------|------|------------|
| `GET` | `/api/backups` | Список файлов в `<workspace>/.agent_backup/` |
| `POST` | `/api/rollback` | Восстановить выбранный бэкап (тело: `{"backup": "…"}`) |
| `GET` | `/api/session` | Статус follow-up сеанса Agent |
| `POST` | `/api/session/reset` | Очистить историю `messages` |

## Примеры запуска

```powershell
# Web-UI
.\start_ui.bat
.\.venv\Scripts\python.exe ui.py --no-browser

# Workflow
.\.venv\Scripts\python.exe workflow.py --clean --task "FizzBuzz от 1 до 20"

# ReAct-агент
.\.venv\Scripts\python.exe agent.py --workspace ./sandbox --task "Найди все .py и кратко опиши" --max-steps 8

# Мультиагент
.\.venv\Scripts\python.exe multi_agent.py --task "Реализуй калькулятор: + - * / с CLI-аргументами"

# Docker
docker compose up --build
```
