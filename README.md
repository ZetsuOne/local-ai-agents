English | [Русский](README_RU.md)

# local-ai-agents

![Web-UI Preview](docs/ui_preview.png)

A pure-Python + `openai` training/lab bench for local **12B–30B** models in LM Studio and any OpenAI-compatible gateway (`LLM_BASE_URL`).

Three modes without LangChain / LlamaIndex / CrewAI, plus a local Web UI:

| Mode | File | Pattern |
|------|------|---------|
| Workflow | `workflow.py` | Evaluator-Optimizer (generator → run → critic) |
| Agent | `agent.py` | Autonomous ReAct + Tool Calling |
| Multi-Agent | `multi_agent.py` | Architect → Coder ↔ Reviewer |
| Web-UI | `ui.py` + `static/index.html` | Dark offline UI (SSE) |

## Installation

**Windows (PowerShell):**

```powershell
cd C:\Projects\local-ai-agents
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS (bash):**

```bash
cd /path/to/local-ai-agents
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In LM Studio, start the local server (port `1234`) and load **one** chat model (embedding models are ignored).

## Quick start (Web-UI)

Double-click `start_ui.bat`, or:

```powershell
.\.venv\Scripts\python.exe ui.py
```

```bash
.venv/bin/python ui.py
```

Opens `http://localhost:8080` (`--no-browser` skips auto-opening the browser). Header controls: model indicator, mode (`Agent` / `Workflow` / `Multi-Agent`), `Workspace`, optional **API Base URL**, `Max steps`, sandbox clean checkbox, **Step X / max_steps** indicator, and **Run** / **Stop** / **New session** buttons.

The center feed streams steps over SSE (Thought, tool calls, unified diff `+`/`−`, `run_python` cards with stdout/stderr without raw JSON duplication, tokens). Above the task field are three template presets (“Script + tests”, “Refactor a file”, “SQLite analysis”). In **Agent** mode, the `messages` history is kept between runs so you can send a follow-up in the same context; **New session** clears history.

On the right: workspace file tree, line-numbered viewer, and **Backups** (versions from `.agent_backup/` + **Rollback**). Runs are also logged to `agent_runs.log`.

## AI gateways / OpenRouter

The client is OpenAI-compatible. Switch providers via environment variables (or the **API Base URL** field in the UI / `--base-url` / `--api-key` flags):

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_BASE_URL` | `http://localhost:1234/v1` | Gateway base URL |
| `LLM_API_KEY` | `lm-studio` | API key (a placeholder is enough for local LM Studio) |
| `LLM_MODEL` | *(empty)* | Explicit model; if unset — `client.models.list()` |

**Windows (PowerShell):**

```powershell
# Local LM Studio (default)
.\.venv\Scripts\python.exe agent.py --task "..."

# OpenRouter
$env:LLM_BASE_URL = "https://openrouter.ai/api/v1"
$env:LLM_API_KEY = "sk-or-..."
$env:LLM_MODEL = "openai/gpt-4o-mini"
.\.venv\Scripts\python.exe agent.py --task "..."

# Or via flags
.\.venv\Scripts\python.exe agent.py --base-url "https://openrouter.ai/api/v1" --api-key "sk-or-..." --model "openai/gpt-4o-mini" --task "..."
```

**Linux / macOS (bash):**

```bash
# Local LM Studio (default)
.venv/bin/python agent.py --task "..."

# OpenRouter
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export LLM_API_KEY="sk-or-..."
export LLM_MODEL="openai/gpt-4o-mini"
.venv/bin/python agent.py --task "..."

# Or via flags
.venv/bin/python agent.py --base-url "https://openrouter.ai/api/v1" --api-key "sk-or-..." --model "openai/gpt-4o-mini" --task "..."
```

API calls use retry: up to 3 attempts with delays 1s → 2s → 4s on network failures and HTTP 429/502/503.

## Docker

Requires Docker Desktop. LM Studio (or another gateway) must listen on the host (`1234`). The container reaches it via `host.docker.internal`.

```powershell
docker compose up --build
```

```bash
docker compose up --build
```

UI: `http://localhost:8080`. The `./sandbox` directory is mounted into the container.

```powershell
# image only
docker build -t local-ai-agents .
docker run --rm -p 8080:8080 --add-host=host.docker.internal:host-gateway -e LLM_BASE_URL=http://host.docker.internal:1234/v1 -v ${PWD}/sandbox:/app/sandbox local-ai-agents
```

```bash
# image only
docker build -t local-ai-agents .
docker run --rm -p 8080:8080 --add-host=host.docker.internal:host-gateway -e LLM_BASE_URL=http://host.docker.internal:1234/v1 -v "$PWD/sandbox:/app/sandbox" local-ai-agents
```

## Layout

| File | Purpose |
|------|---------|
| `llm.py` | Gateway client, env config, retry, logging → `agent_runs.log`, context compression, loop-guard |
| `tools.py` | Workspace + backups/rollback + `ast.parse` / `get_file_outline` + 10 tools |
| `events.py` | Event bus for CLI, SSE, and Thought dedup |
| `workflow.py` | Deterministic generator → run → critic loop (up to 3 edits) |
| `agent.py` | Autonomous ReAct agent with tool calling + final report at max_steps |
| `multi_agent.py` | 3-role pipeline with restricted tools |
| `ui.py` | `ThreadingHTTPServer` + SSE API on port `8080` (session, backups) |
| `static/index.html` | Offline UI (no CDN) |
| `Dockerfile` / `docker-compose.yml` | Web-UI packaging |
| `start_ui.bat` | One-click UI launch |

## Workspace, `--clean`, and auto-backups

The working directory is set via `set_workspace()` / `--workspace` (default `./sandbox`). All tool paths are enforced strictly inside the active `WORKSPACE_DIR`.

```powershell
.\.venv\Scripts\python.exe agent.py --workspace ./sandbox --task "..."
.\.venv\Scripts\python.exe workflow.py --workspace D:\my-project --task "..."
.\.venv\Scripts\python.exe multi_agent.py --clean --task "..."   # ./sandbox only
```

```bash
.venv/bin/python agent.py --workspace ./sandbox --task "..."
.venv/bin/python workflow.py --workspace /path/to/my-project --task "..."
.venv/bin/python multi_agent.py --clean --task "..."   # ./sandbox only
```

- `--clean` clears **only** the default `./sandbox` before a task (other `--workspace` values raise an error).
- Before modifying an existing file (`write_file` / `edit_file` / `replace_lines`), a copy is saved to `<workspace>/.agent_backup/<YYYYMMDD_HHMMSS>_<name>`.
- Directories `__pycache__`, `.git`, `.venv`, `.agent_backup` and `*.pyc` files are hidden from `list_files` / `search_in_files` (and from the UI tree). `.agent_backup` is also in `.gitignore`.
- Rollback from the Web-UI: `GET /api/backups` + `POST /api/rollback` — the current version is backed up again before restore.
- For `.py` files, `ast.parse` runs right after write; on `SyntaxError` the tool returns the line number and message so the model can fix the code without an extra `run_python`.
- File changes in the UI are shown as a **unified diff** (`difflib`), with `+` lines green and `-` lines red.

## Model auto-detection

There is no hardcoded model name. If `--model` / `LLM_MODEL` are unset, the first chat model from `client.models.list()` is used (`get_active_model` in `llm.py`).

```powershell
# auto-detect
.\.venv\Scripts\python.exe agent.py --task "..."

# explicit
.\.venv\Scripts\python.exe agent.py --model "your-model" --task "..."
```

```bash
# auto-detect
.venv/bin/python agent.py --task "..."

# explicit
.venv/bin/python agent.py --model "your-model" --task "..."
```

## 1. Workflow (`workflow.py`)

Deterministic Evaluator-Optimizer: the generator writes `solution.py` → run → the critic edits via `replace_lines` / `edit_file` (up to 3 cycles).

```powershell
.\.venv\Scripts\python.exe workflow.py --task "Write a script that sums numbers from 1 to 100 and prints the result"
```

```bash
.venv/bin/python workflow.py --task "Write a script that sums numbers from 1 to 100 and prints the result"
```

Options: `--model`, `--workspace`, `--clean`, `--base-url`, `--api-key`.

## 2. Agent (`agent.py`)

Autonomous ReAct agent with all 10 tools. Before each LLM request it compresses older `role=tool` messages (keeping the latest 3) to 250 characters; on a repeated identical failing tool call it injects a warning; prints `prompt_tokens` / `completion_tokens` when the gateway provides them.

If the loop hits `max_steps` and the last step was a tool call, the agent makes **one** final `chat_completion` **without** `tools` to produce a user-facing summary.

```powershell
.\.venv\Scripts\python.exe agent.py --task "Create hello.py that prints Hello, agents! and run it" --max-steps 12
```

```bash
.venv/bin/python agent.py --task "Create hello.py that prints Hello, agents! and run it" --max-steps 12
```

Options: `--model`, `--max-steps`, `--workspace`, `--clean`, `--base-url`, `--api-key`.

In the Web-UI, Agent session `messages` history is preserved: the next task is a follow-up; **New session** (`POST /api/session/reset`) clears context.

## 3. Multi-Agent (`multi_agent.py`)

Three specialists with distinct system prompts and restricted tools (up to 2 Coder↔Reviewer cycles):

1. **Architect** — plan/spec in `plan.md` (`list_files`, `read_file`, `get_file_outline`, `write_file`, `replace_lines`)
2. **Coder** — implement from the plan (`write_file` / `replace_lines` / `edit_file` + `run_python`)
3. **Reviewer** — reads code, writes `test_solution.py`, runs tests; on failure returns `BUG_REPORT:` to the Coder

```powershell
.\.venv\Scripts\python.exe multi_agent.py --task "Write is_palindrome(s) and a script that checks several strings" --max-steps 10
```

```bash
.venv/bin/python multi_agent.py --task "Write is_palindrome(s) and a script that checks several strings" --max-steps 10
```

Options: `--model`, `--max-steps`, `--workspace`, `--clean`, `--base-url`, `--api-key`.

## Tools (10, workspace-scoped)

All paths are confined to the active working directory (`Path.resolve()` + boundary checks):

| Tool | Description |
|------|-------------|
| `list_files` | List workspace files (excludes `__pycache__` / `.git` / `.venv` / `.agent_backup` / `*.pyc`) |
| `read_file` | Read with line numbers and header `=== file (lines a-b of N) ===` |
| `get_file_outline` | `.py` outline via `ast`: classes, functions, line numbers |
| `write_file` | Create / overwrite + backup + `ast.parse` for `.py` |
| `edit_file` | Unique `old_text` replacement (+ backup / syntax check) |
| `replace_lines` | Replace a line range by numbers (+ backup / syntax check) |
| `search_in_files` | Substring search across text files (same exclusions as `list_files`) |
| `run_python` | Run `.py` (15s timeout) → JSON returncode/stdout/stderr |
| `pip_install` | Install a package via the current pip (safe names only) |
| `sqlite_query` | Read-only: URI `?mode=ro`, single `SELECT`/`PRAGMA` without `;` |

## Web-UI API (extra)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/backups` | List files in `<workspace>/.agent_backup/` |
| `POST` | `/api/rollback` | Restore a selected backup (body: `{"backup": "…"}`) |
| `GET` | `/api/session` | Agent follow-up session status |
| `POST` | `/api/session/reset` | Clear `messages` history |

## Example commands

```powershell
# Web-UI
.\start_ui.bat
.\.venv\Scripts\python.exe ui.py --no-browser

# Workflow
.\.venv\Scripts\python.exe workflow.py --clean --task "FizzBuzz from 1 to 20"

# ReAct agent
.\.venv\Scripts\python.exe agent.py --workspace ./sandbox --task "Find all .py files and summarize briefly" --max-steps 8

# Multi-agent
.\.venv\Scripts\python.exe multi_agent.py --task "Implement a calculator: + - * / with CLI args"

# Docker
docker compose up --build
```

```bash
# Web-UI
.venv/bin/python ui.py --no-browser

# Workflow
.venv/bin/python workflow.py --clean --task "FizzBuzz from 1 to 20"

# ReAct agent
.venv/bin/python agent.py --workspace ./sandbox --task "Find all .py files and summarize briefly" --max-steps 8

# Multi-agent
.venv/bin/python multi_agent.py --task "Implement a calculator: + - * / with CLI args"

# Docker
docker compose up --build
```
