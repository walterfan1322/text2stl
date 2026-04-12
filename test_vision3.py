import requests, base64, struct, zlib

def make_png(w, h, r, g, b):
    def chunk(ct, d):
        c = ct + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    raw = b""
    for y in range(h):
        raw += b"\x00" + bytes([r, g, b]) * w
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

b64 = base64.b64encode(make_png(50, 50, 255, 0, 0)).decode()

# Unload other models first
print("Unloading other models...")
for m in ["qwen3.5:35b-a3b", "qwen3.5:9b"]:
    try:
        requests.post("http://localhost:11434/api/chat", json={"model": m, "keep_alive": 0, "messages": []}, timeout=10)
    except:
        pass

import time
time.sleep(2)

print("Testing qwen2.5vl:7b with image...")
try:
    r = requests.post("http://localhost:11434/api/chat", json={
        "model": "qwen2.5vl:7b",
        "messages": [{"role": "user", "content": "What color is this image? Answer in one word.", "images": [b64]}],
        "stream": False, "options": {"num_predict": 20},
    }, timeout=180)
    d = r.json()
    if "error" in d:
        print(f"ERROR: {d['error'][:200]}")
    else:
        print(f"SUCCESS: {d['message']['content']}")
except Exception as e:
    print(f"EXCEPTION: {e}")
