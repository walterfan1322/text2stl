import requests, base64, json

# Create a simple test image (10x10 red square)
# Minimal PNG bytes
import struct, zlib
def make_png(w, h, r, g, b):
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF)
    ihdr = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    raw = b''
    for y in range(h):
        raw += b'\x00' + bytes([r, g, b]) * w
    idat = zlib.compress(raw)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', idat) + chunk(b'IEND', b'')

png_data = make_png(10, 10, 255, 0, 0)
b64_img = base64.b64encode(png_data).decode()

payload = {
    "model": "qwen3.5:35b-a3b",
    "messages": [{"role": "user", "content": "What is in this image? One sentence.", "images": [b64_img]}],
    "stream": False,
    "options": {"num_predict": 50},
    "think": False,
}

try:
    r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    data = r.json()
    if "error" in data:
        print(f"ERROR: {data['error']}")
    else:
        print(f"SUCCESS: {data['message']['content']}")
except Exception as e:
    print(f"EXCEPTION: {e}")
