@echo off
title AI Monorepo Manager
cd /d "%~dp0Manager Projects"
pip install -r requirements.txt -q 2>nul
python main.py %*
pause
