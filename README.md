# Text2STL

Natural language to 3D printable STL model generator.

Input a description like "a chair" or "a gear with 20 teeth", and get a downloadable STL file ready for 3D printing.

## Architecture

```
Browser  -->  FastAPI (port 8000)  -->  LLM (Ollama or Cloud API)
                  |                           |
                  v                           v
             trimesh (Python)          Generate Python code
                  |                    that builds 3D mesh
                  v
              STL file
```

**Pipeline:** User prompt --> Web search for references --> LLM enriches design spec --> LLM generates trimesh Python code --> Execute code --> Export STL

## Features

- Natural language input (supports Chinese and English)
- Web search for design references (dimensions, structure)
- LLM-powered design enrichment and code generation
- 3D preview in browser (Three.js)
- Auto-review and iterative refinement
- Manual feedback loop for adjustments
- Multiple LLM backends:
  - **Local**: Ollama models via SSH tunnel to GPU server
  - **Cloud**: MiniMax, OpenAI-compatible APIs

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy `config.example.json` to `config.json` and fill in your settings:

```bash
cp config.example.json config.json
```

**For cloud API (recommended):**
```json
{
    "cloud_api_key": "your-api-key",
    "cloud_api_base": "https://api.minimax.io/v1",
    "cloud_models": ["MiniMax-M2.7"]
}
```

**For local Ollama:**
```json
{
    "ollama_url": "http://localhost:11434",
    "ollama_model": "qwen3.5:35b-a3b",
    "ssh_tunnel_host": "user@your-gpu-server"
}
```

### 3. Run

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

## Supported 3D Modeling Strategies

| Strategy | Use Case | Example |
|----------|----------|---------|
| Extrude polygon | Flat objects with uniform cross-section | Bookend, bracket, wrench |
| Solid of revolution | Round/symmetric objects | Cup, vase, bowl |
| Composite primitives | Multi-part assembled objects | Chair, table, phone stand |

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **3D Engine**: trimesh (Python)
- **LLM**: Ollama (local) / MiniMax (cloud) / OpenAI-compatible APIs
- **Frontend**: Vanilla JS + Three.js (STL viewer)
- **Tunnel**: Paramiko SSH (for remote GPU server)

## License

MIT
