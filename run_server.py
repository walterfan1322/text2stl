import subprocess, sys, os
os.chdir(r'C:\Users\user\text2stl')
os.environ['OLLAMA_URL'] = 'http://localhost:11434'
os.environ['OLLAMA_MODEL'] = 'qwen3-nothink:latest'
os.environ['SSH_TUNNEL_ENABLED'] = 'false'
import uvicorn
uvicorn.run('app:app', host='0.0.0.0', port=8000, log_level='info')
