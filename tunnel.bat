@echo off
chcp 65001 >nul
echo ============================================
echo   SSH 隧道 - 連接到 DGX Ollama
echo ============================================
echo.
echo 連接到 YOUR_DGX_IP 轉發 Ollama API...
echo 密碼: 721nc334lab
echo.
ssh -N -L 11434:localhost:11434 user@YOUR_DGX_HOST
pause
