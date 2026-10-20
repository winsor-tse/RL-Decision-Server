@echo off
setlocal
cd /d "%~dp0"
set "MYSTIC_PYTHON="
if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "MYSTIC_PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
if not defined MYSTIC_PYTHON if exist ".venv\Scripts\python.exe" set "MYSTIC_PYTHON=%CD%\.venv\Scripts\python.exe"
if not defined MYSTIC_PYTHON if exist "RL_venv\Scripts\python.exe" set "MYSTIC_PYTHON=%CD%\RL_venv\Scripts\python.exe"
if not defined MYSTIC_PYTHON (
  echo Activate a Python environment or create .venv. See Training\Mystic_Sim\FULL_TRAINING.md.
  exit /b 1
)
"%MYSTIC_PYTHON%" -m Training.Mystic_Sim.train_full %*
exit /b %ERRORLEVEL%
