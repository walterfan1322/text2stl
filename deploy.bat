@echo off
chcp 65001 >nul
echo ============================================
echo   Text2STL 部署腳本
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python 未安裝，正在下載...
    curl -L -o python-installer.exe https://www.python.org/ftp/python/3.12.4/python-3.12.4-amd64.exe
    echo [!] 請手動執行 python-installer.exe 安裝 Python（記得勾選 Add to PATH）
    echo [!] 安裝完成後重新執行此腳本
    pause
    exit /b 1
)
echo [OK] Python 已安裝

:: Check OpenSCAD
where openscad >nul 2>&1
if %errorlevel% neq 0 (
    if exist "C:\Program Files\OpenSCAD\openscad.exe" (
        echo [OK] OpenSCAD 已安裝 (Program Files)
    ) else (
        echo [!] OpenSCAD 未安裝，正在下載...
        curl -L -o OpenSCAD-installer.exe https://files.openscad.org/OpenSCAD-2024.12.06-x86-64-Installer.exe
        echo [!] 請手動執行 OpenSCAD-installer.exe 安裝
        echo [!] 安裝完成後重新執行此腳本
        pause
        exit /b 1
    )
) else (
    echo [OK] OpenSCAD 已安裝
)

:: Install Python dependencies
echo.
echo [*] 安裝 Python 套件...
pip install -r requirements.txt -q

:: Set environment variables
set OLLAMA_URL=http://localhost:11434
set OLLAMA_MODEL=qwen3-nothink:latest

:: Find OpenSCAD path
if exist "C:\Program Files\OpenSCAD\openscad.exe" (
    set OPENSCAD_PATH=C:\Program Files\OpenSCAD\openscad.exe
) else (
    set OPENSCAD_PATH=openscad
)

echo.
echo ============================================
echo   啟動服務
echo ============================================
echo.
echo [1] 先在另一個終端視窗啟動 SSH 隧道：
echo     ssh -L 11434:localhost:11434 user@YOUR_DGX_HOST
echo.
echo [2] 服務啟動後，在瀏覽器打開：
echo     http://localhost:8000
echo.
echo ============================================
echo.

python -m uvicorn app:app --host 0.0.0.0 --port 8000
pause
