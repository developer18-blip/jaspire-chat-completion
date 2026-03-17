@echo off
setlocal

echo Setting up JASPIRE Chat API package...

python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH. Install Python 3.10+ and retry.
    pause
    exit /b 1
)

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -e .

echo.
echo Installation complete.
echo Starting one-command auto mode...
echo.

jaspire auto --with-search
