@echo off
chcp 65001 >nul
echo ============================================
echo   Text2STL - 啟動中
echo ============================================

set OLLAMA_URL=http://localhost:11434
set OLLAMA_MODEL=qwen3-nothink:latest

echo [*] 啟動 SSH 隧道到 DGX...
start "" ssh -o StrictHostKeyChecking=no -N -L 11434:localhost:11434 user@YOUR_DGX_HOST
timeout /t 3 /nobreak >nul

echo [*] 啟動 Text2STL 服務...
echo [*] 瀏覽器打開 http://localhost:8000
echo.

cd /d C:\Users\user\text2stl
python -m uvicorn app:app --host 0.0.0.0 --port 8000
pause
