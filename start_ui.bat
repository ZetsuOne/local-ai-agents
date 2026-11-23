@echo off
cd /d "%~dp0"
".\.venv\Scripts\python.exe" ui.py
if errorlevel 1 (
  echo.
  echo Ошибка запуска UI. Проверьте .venv и LM Studio.
  pause
)
