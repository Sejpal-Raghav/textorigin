@echo off
cd /d "%~dp0"
py -3.11 run_gpu.py %*
exit /b %errorlevel%
