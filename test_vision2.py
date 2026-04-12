import requests, base64, struct, zlib

def make_png(w, h, r, g, b):
    def chunk(ct, d):
        c = ct + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    raw = b""
    for y in range(h):
        raw += b"\x00" + bytes([r, g, b]) * w
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

b64 = base64.b64encode(make_png(10, 10, 255, 0, 0)).decode()

for model in ["qwen3.5:9b", "qwen3.5:35b-a3b"]:
    print(f"\nTesting {model}...")
    try:
        # First unload any loaded model
        requests.post("http://localhost:11434/api/chat", json={"model": model, "keep_alive": 0, "messages": []}, timeout=30)
    except:
        pass
    
    # Try text-only first to see if model loads
    try:
        r = requests.post("http://localhost:11434/api/chat", json={
            "model": model,
            "messages": [{"role": "user", "content": "Say hello in 3 words"}],
            "stream": False, "options": {"num_predict": 10}, "think": False
        }, timeout=120)
        d = r.json()
        if "error" in d:
            print(f"  Text-only: ERROR - {d['error'][:100]}")
            continue
        print(f"  Text-only: OK - {d['message']['content'][:50]}")
    except Exception as e:
        print(f"  Text-only: EXCEPTION - {e}")
        continue

    # Now try with image
    try:
        r = requests.post("http://localhost:11434/api/chat", json={
            "model": model,
            "messages": [{"role": "user", "content": "What color is this image?", "images": [b64]}],
            "stream": False, "options": {"num_predict": 20}, "think": False
        }, timeout=120)
        d = r.json()
        if "error" in d:
            print(f"  With image: ERROR - {d['error'][:100]}")
        else:
            print(f"  With image: OK - {d['message']['content'][:100]}")
    except Exception as e:
        print(f"  With image: EXCEPTION - {e}")
