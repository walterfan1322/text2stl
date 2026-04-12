@echo off
chcp 65001 >nul
echo ============================================
echo   Text2STL 啟動
echo ============================================

set OLLAMA_URL=http://localhost:11434
set OLLAMA_MODEL=qwen3-nothink:latest

echo.
echo [!] 確認 SSH 隧道已啟動：
echo     ssh -L 11434:localhost:11434 user@YOUR_DGX_HOST
echo.
echo [*] 服務位址: http://localhost:8000
echo.

cd /d C:\Users\user\text2stl
python -m uvicorn app:app --host 0.0.0.0 --port 8000
pause
