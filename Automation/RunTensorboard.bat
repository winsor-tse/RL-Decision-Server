@echo off
cd /d "%~dp0.."
if exist "RL_venv\Scripts\python.exe" (
    "RL_venv\Scripts\python.exe" -m Automation.tensorboard_server --logdir runs
) else (
    python -m Automation.tensorboard_server --logdir runs
)
pause
