#!/usr/bin/env bash
set -e

echo -e "\033[1;36m=================================================="
echo -e "   Ael-Agents // Horizon & Amnesia Breaker        "
echo -e "==================================================\033[0m"

if command -v uv &> /dev/null; then
    echo -e "\033[1;32m[✓] uv detected. Syncing virtual environment...\033[0m"
    uv sync
    uv run python main.py "$@"
else
    echo -e "\033[1;33m[!] uv not detected. Falling back to Python venv...\033[0m"
    if [ ! -d ".venv" ]; then
        echo -e "\033[1;36m[+] Creating virtual environment...\033[0m"
        python3 -m venv .venv
        .venv/bin/pip install -e .
    fi
    .venv/bin/python main.py "$@"
fi
