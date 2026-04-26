# -*- coding: utf-8 -*-
"""Watchdog for Text2STL: keep the FastAPI service alive on port 8000."""
import logging
import socket
import subprocess
import sys
import time
from pathlib import Path

# --- singleton guard (added by hardening) ---
import socket as _wd_socket
_WD_MUTEX_PORT = 47100
try:
    _wd_mutex = _wd_socket.socket(_wd_socket.AF_INET, _wd_socket.SOCK_STREAM)
    _wd_mutex.setsockopt(_wd_socket.SOL_SOCKET, _wd_socket.SO_REUSEADDR, 0)
    _wd_mutex.bind(("127.0.0.1", _WD_MUTEX_PORT))
    _wd_mutex.listen(1)
except OSError:
    print("[watchdog] another instance is already running, exiting.")
    import sys as _wd_sys
    _wd_sys.exit(0)
# --- end singleton guard ---

APP_DIR = Path(r"C:\Users\user\text2stl")
PORT = 8000
CHECK_INTERVAL = 15
RESTART_COOLDOWN = 8
MAX_RESTART_RETRIES = 8
PYTHON_EXE = sys.executable or r"C:\Program Files\Python312\python.exe"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(APP_DIR / "watchdog.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


def is_port_open(port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def kill_existing() -> None:
    try:
        result = subprocess.run(
            f'netstat -ano | findstr :{PORT} | findstr LISTENING',
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if not parts:
                continue
            pid = parts[-1]
            if pid.isdigit() and pid != '0':
                subprocess.run(
                    f'taskkill /F /PID {pid}',
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                logging.info(f'Killed PID {pid}')
    except Exception as exc:
        logging.warning(f'kill_existing error: {exc}')


def start_server() -> bool:
    try:
        stdout = open(APP_DIR / 'server_stdout.log', 'a', encoding='utf-8')
        stderr = open(APP_DIR / 'server_err.log', 'a', encoding='utf-8')
        subprocess.Popen(
            [PYTHON_EXE, '-m', 'uvicorn', 'app:app', '--host', '0.0.0.0', '--port', str(PORT)],
            cwd=str(APP_DIR),
            stdout=stdout,
            stderr=stderr,
            creationflags=0x00000008,
        )
        logging.info('Started Text2STL server')
        return True
    except Exception as exc:
        logging.error(f'Failed to start server: {exc}')
        return False


def main() -> None:
    logging.info(f'Text2STL watchdog started on port {PORT}')
    consecutive_failures = 0
    while True:
        time.sleep(CHECK_INTERVAL)
        if is_port_open(PORT):
            if consecutive_failures:
                logging.info('Port recovered and is healthy again')
            consecutive_failures = 0
            continue

        consecutive_failures += 1
        logging.warning(f'Port {PORT} is down (failure #{consecutive_failures})')
        if consecutive_failures > MAX_RESTART_RETRIES:
            logging.error('Too many restart failures, sleeping 5 minutes before retry')
            time.sleep(300)
            consecutive_failures = 0
            continue

        kill_existing()
        time.sleep(2)
        if start_server():
            time.sleep(RESTART_COOLDOWN)
            if is_port_open(PORT):
                logging.info('Restart successful')
                consecutive_failures = 0
            else:
                logging.error('Restart attempted but port is still closed')


if __name__ == '__main__':
    main()
