@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3 -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r dashboardequirements.txt
streamlit run dashboardpp.py
pause
