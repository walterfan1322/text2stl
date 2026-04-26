# Text2STL

Natural language to 3D printable STL model generator.

Input a description like "a chair" or "a mug with handle", and get a downloadable STL ready for 3D printing.

## Pipeline

```
Prompt -> Web search -> LLM enrich -> LLM code gen -> Sandbox exec
                                                        |
                                                        v
                                                   trimesh / cadquery
                                                        |
                                                        v
                          Geom check + Watertight repair + Judge (VLM)
                                                        |
                                                        v
                                                       STL
```

## Features (Sprint 5-7)

**Generation**
- Multi-backend: trimesh + cadquery (cadquery default for solids)
- Multiple cloud LLMs with shape-based routing (MiniMax-M2.7, deepseek-chat, gemini-2.5-flash)
- Pattern cache for instant cache hits (zero token, zero latency on repeats)
- Best-of-N candidate generation per shape category

**Quality**
- AST validator + RestrictedPython sandbox (`sandbox_strict`)
- Programmatic geometry checks (`judge_geometric` — zero-token validation)
- Mesh repair pipeline (pymeshfix + watertight check)
- VLM judge with auto-retry on score < 6
- Print readiness analyzer (overhang/support hints)

**Robustness**
- Output cache (avoid re-running same prompt+model+seed)
- Structured JSONL logging for token tracking
- Auto-failover between cloud providers
- Multi-format export (STL/OBJ/3MF/STEP)

**Frontend**
- Three.js STL viewer with auto-rotate
- Cloud-only model dropdown
- Iterative refine via natural-language feedback

## Quick Start

```bash
git clone https://github.com/walterfan1322/text2stl
cd text2stl
pip install -r requirements.txt
pip install pymeshfix scipy networkx pyvista RestrictedPython pymeshlab

# Configure
cp config.example.json config.json
# Edit config.json: fill cloud_providers.*.key with your API keys

# Run
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

## Configuration

`config.json` controls everything. Key sections:

```jsonc
{
  "cloud_providers": {
    "minimax":  {"base": "https://api.minimax.io/v1",                       "key": "..."},
    "deepseek": {"base": "https://api.deepseek.com/v1",                     "key": "..."},
    "gemini":   {"base": "https://generativelanguage.googleapis.com/...",   "key": "..."}
  },
  "shape_routing": {"bottle": "deepseek-chat"},   // route specific shapes to specific models
  "feature_flags": {
    "output_cache":    true,
    "geom_check":      true,
    "sandbox_strict":  true,
    "best_of_n":       true,
    "slicer_check":    false   // requires desktop session
  }
}
```

API keys can also be supplied via environment variables: `MINIMAX_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`.

## Modeling Strategies

| Strategy | Use Case | Backend |
|----------|----------|---------|
| Extrude polygon | Flat objects with uniform cross-section | trimesh / cadquery |
| Solid of revolution | Round/symmetric objects (cup, vase, bottle) | trimesh / cadquery |
| Composite primitives | Multi-part assembled objects (chair, table) | trimesh / cadquery |

## Tests

```bash
python tests/test_pattern_cache.py
python tests/test_mesh_repair.py
python tests/test_judge_geometric.py
python tests/test_sandbox_strict.py
python tests/test_best_of_n.py
# ... see tests/ for full list
```

CI workflow at `.github/workflows/smoke.yml` runs unit tests on every PR plus a smoke benchmark when generation logic changes.

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **3D Engine**: trimesh + cadquery + pymeshfix
- **LLM**: MiniMax / DeepSeek / Gemini (OpenAI-compatible)
- **Frontend**: Vanilla JS + Three.js (STL viewer)

See [PLAN_v2.md](PLAN_v2.md) and [SPRINT_5_7_RESULTS.md](SPRINT_5_7_RESULTS.md) for design rationale and benchmarks.

## License

MIT
