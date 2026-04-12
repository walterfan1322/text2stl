"""Quick test script for the Text2STL API."""
import requests, json, sys

url = "http://localhost:8000/api/generate"
prompt = sys.argv[1] if len(sys.argv) > 1 else "a simple vase, bottom radius 20mm, top radius 30mm, height 100mm"

print(f"Prompt: {prompt}")
print("Generating...")
try:
    r = requests.post(url, json={"prompt": prompt}, timeout=900)
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"Job ID: {data['id']}")
        print(f"STL URL: {data['stl_url']}")
        print(f"Code:\n{data['code']}")
    else:
        print(f"Error: {r.text}")
except Exception as e:
    print(f"Exception: {e}")
