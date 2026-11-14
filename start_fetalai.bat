@echo off
REM One-click start of the whole FetalAI demo (backend + frontend). Double-click this file.
REM First run installs Python/Node packages (needs internet); later runs start in under a minute.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_fetalai.ps1"
