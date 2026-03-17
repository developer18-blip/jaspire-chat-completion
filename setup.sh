#!/bin/bash
set -e

echo "Setting up JASPIRE Chat API package..."

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is not installed. Please install Python 3.10+ and retry."
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

python -m pip install --upgrade pip
pip install -e .

echo ""
echo "Installation complete."
echo "Starting one-command auto mode..."
echo ""

jaspire auto --with-search
