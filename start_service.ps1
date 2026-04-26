# Text2STL - Start as persistent background services
$ErrorActionPreference = "SilentlyContinue"
$workDir = "C:\Users\user\text2stl"

# Kill existing
taskkill /f /im python.exe 2>$null
Stop-Process -Name ssh -Force -ErrorAction SilentlyContinue
Start-Sleep 1

# Remove old scheduled tasks
schtasks /delete /tn "Text2STL_Tunnel" /f 2>$null
schtasks /delete /tn "Text2STL_Server" /f 2>$null

# Create tunnel task
schtasks /create /tn "Text2STL_Tunnel" /tr "ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -N -L 11434:localhost:11434 user@YOUR_DGX_HOST" /sc onstart /ru "user" /rp "1234" /rl highest /f
schtasks /run /tn "Text2STL_Tunnel"

Write-Host "[OK] SSH tunnel task created and started"
Start-Sleep 3

# Create server wrapper script
$wrapper = @"
import subprocess, sys, os
os.chdir(r'$workDir')
os.environ['OLLAMA_URL'] = 'http://localhost:11434'
os.environ['OLLAMA_MODEL'] = 'qwen3-nothink:latest'
os.environ['SSH_TUNNEL_ENABLED'] = 'false'
import uvicorn
uvicorn.run('app:app', host='0.0.0.0', port=8000, log_level='info')
"@
Set-Content -Path "$workDir\run_server.py" -Value $wrapper

# Create server task
schtasks /create /tn "Text2STL_Server" /tr "python $workDir\run_server.py" /sc onstart /ru "user" /rp "1234" /rl highest /f
schtasks /run /tn "Text2STL_Server"

Write-Host "[OK] Server task created and started"
Start-Sleep 5

# Verify
$ErrorActionPreference = "Stop"
try {
    $r = Invoke-WebRequest -Uri "http://localhost:8000/api/models" -UseBasicParsing -TimeoutSec 10
    Write-Host "[OK] Server running! Models: $($r.Content)"
} catch {
    Write-Host "[!!] Server not ready yet, may need a moment..."
}

Write-Host ""
Write-Host "Service URL: http://localhost:8000"
Write-Host "Tasks: Text2STL_Tunnel, Text2STL_Server"
