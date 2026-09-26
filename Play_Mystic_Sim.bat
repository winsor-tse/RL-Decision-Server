@echo off
cd /d "%~dp0"
if not exist "RL_venv\Scripts\python.exe" (
  echo Missing RL_venv. Set up the project Python environment first.
  pause
  exit /b 1
)
"RL_venv\Scripts\python.exe" -m Custom_enviornments.Mystic_Sim.viewer %*
if errorlevel 1 pause
