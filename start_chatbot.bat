@echo off
echo ================================================
echo  Insurance Advisory AI - Professional Version
echo ================================================
echo.

REM Start Ollama if not running
echo Checking Ollama...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo Starting Ollama...
    start /B "" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
    timeout /t 3 /nobreak >nul
)
echo Ollama: Ready

REM Start Chatbot with VENV
echo.
echo Starting Chatbot with Virtual Environment...
cd /d "%~dp0"
.\.venv\Scripts\python.exe -m streamlit run app_final.py

pause
