@echo off
setlocal
cd /d "%~dp0"
set "MYSTIC_PYTHON="
if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "MYSTIC_PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
if not defined MYSTIC_PYTHON if exist "RL_venv\Scripts\python.exe" set "MYSTIC_PYTHON=%CD%\RL_venv\Scripts\python.exe"
if not defined MYSTIC_PYTHON if exist ".venv\Scripts\python.exe" set "MYSTIC_PYTHON=%CD%\.venv\Scripts\python.exe"
if not defined MYSTIC_PYTHON (
  echo No Python virtual environment found. Activate your environment or create .venv or RL_venv.
  echo Install the project requirements using: python -m pip install -r requirements.txt
  pause
  exit /b 1
)
"%MYSTIC_PYTHON%" -m Custom_enviornments.Mystic_Sim.viewer %*
set "MYSTIC_EXIT=%ERRORLEVEL%"
if not "%MYSTIC_EXIT%"=="0" pause
exit /b %MYSTIC_EXIT%
