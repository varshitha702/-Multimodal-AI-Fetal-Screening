@echo off
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
python check_hc18.py
pause
