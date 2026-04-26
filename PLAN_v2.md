# Text2STL 改版計劃書 v2.0 — Sprint 5 ~ 8

承接 v1.0 (Phase 0~3) 與 Sprint 3+4 之後的下一輪。**不含 LLM 替換**（單獨 A/B 處理，不入此計劃）。

範圍：12 項升級分 4 個 Sprint，從「機械工」一路做到「新能力」。每個 Sprint 都可獨立 ship，feature flag 翻一行就能回滾。

---

## 整體里程碑

| Sprint | 主題 | 預估天數 | 主要產出 |
|---|---|---:|---|
| 5 | 機械工：低成本高頻價值 | 3~5 | output cache / 多格式 export / CI / render 升級 |
| 6 | 品質：把對的東西生對 | 5~7 | 程式化幾何驗證 / mesh 後處理鏈 / slicer |
| 7 | 健壯性：產量、可觀測、隔離 | 5~7 | Best-of-N / Langfuse / RestrictedPython |
| 8 | 新能力：hybrid backend + 新輸入 | 7~10 | Hunyuan3D 後端 / sketch-to-3D / image-to-3D / voice |

**總計**：~20~30 工作日，可全部串成一條 timeline，也可挑食。

---

# Sprint 5 — 機械工週

目標：把所有「純機械工」搞定，立刻能感受到的 latency / DX 提升。  
特性：無架構改動、無新依賴衝突、可一天 ship 一個。

## 5.1 Output cache（exact-match LLM 回應快取）

**目標**：同 prompt + 同 model + 同 system_prompt → 直接命中歷史 STL，0 token 0 延遲。

- [ ] 在 `pattern_cache.py` 旁加 `output_cache.py`：sqlite 表 `(prompt_hash, model, sys_hash) -> stl_path, meta_json`
- [ ] `app.generate()` 開頭做 lookup；命中 → `_failover_stats` 加 `cache_hit` 欄位、直接回傳檔
- [ ] 加 `?no_cache=1` query param 讓使用者強制重生
- [ ] 加 `tests/test_output_cache.py`：3 個情境（hit / miss / 強制 no_cache）

驗收：
- 重複 prompt latency < 200ms（從 30~70s）
- `/api/stats.cache` 出現命中率
- 同 prompt 不同 model 不互相污染

成本：S（半天）

## 5.2 多輸出格式（STEP / 3MF / GLB）

**目標**：使用者可下載真正的 CAD 檔（不只 STL）。

- [ ] `backends/cadquery_backend.py` 加 `export_step()` / `export_3mf()`
- [ ] `app.execute_and_export()` 同時產出 4 種格式到 `outputs/<job_id>/model.{stl,step,3mf,glb}`
- [ ] GLB：用 `trimesh.exchange.gltf` 從 STL 轉
- [ ] 前端下載按鈕改 dropdown：「STL / STEP / 3MF / GLB」
- [ ] `cleanup_outputs.py` 一併清理新格式

驗收：
- 4 種檔案都能在對應軟體開啟（FreeCAD 開 STEP、PrusaSlicer 開 3MF、瀏覽器 model-viewer 開 GLB）
- 失敗一種 format 不影響其他三種

成本：S（半天 ~ 一天）

## 5.3 CI 自動化（GitHub Action smoke gate）

**目標**：每個 PR 跑 3 分鐘 smoke benchmark，retains regression-free trunk。

- [ ] `.github/workflows/smoke.yml`：build env、跑 `tests/benchmark_smoke.py`、跑 4 個 unit test suite
- [ ] secrets：將 cloud API keys 放到 GitHub repo secrets，CI 透過 `.env.local` 注入
- [ ] `regression_diff` 自動跑，比對 PR vs main baseline；diff > 閾值阻擋 merge
- [ ] README 加 status badge

驗收：
- main 上有 ✅ smoke badge
- 故意 push 一個會壞的 commit，CI 應紅燈
- secrets 不外洩到 log

成本：S（一天）

## 5.4 Render 品質升級（PyVista）

**目標**：把 VLM judge 看到的圖從「看得到形狀」變成「看得到細節」，預期 score 平均 +0.3~0.5。

- [ ] `pip install pyvista[all]` + `pyvistaqt`
- [ ] `rendering.py` 加 `render_stl_views_pyvista()`，pyrender 路徑保留為 fallback
- [ ] 從 4 視角擴到 8 視角 turntable（30°一張），給 VLM 更多訊號
- [ ] 加環境光 + 邊緣描邊（PyVista `mesh.silhouette`），形狀辨識更明顯
- [ ] judge prompt 對應改成 8 視角描述

驗收：
- A/B：同 STL，PyVista 渲圖 vs pyrender 渲圖各送 VLM 5 次，PyVista avg 分數 ≥ 持平
- 渲染時間 < 3s（pyrender 大概 1.5s — 接受 2x 換品質）

成本：M（一天 ~ 一天半）  
風險：PyVista 需要 VTK，macOS arm64 有 wheel；但若 CI 環境裝不起來要保留 pyrender 路徑

---

# Sprint 6 — 品質週

目標：正確性 / 印表機就緒度的硬實力升級。  
特性：每一項都是「從 N% 拉到 N+10%」的具體質量提升。

## 6.1 程式化幾何驗證器（zero-token judge layer）

**目標**：在 VLM judge 之前先做程式化檢查，抓 VLM 看不出來的拓樸錯誤。

- [ ] 新增 `judge_geometric.py`：`check_topology(stl_path, expected_category) -> CheckResult`
- [ ] 規則表（per-category）：
  - chair / table → `body_count >= 4`（4 條腿） + `bbox_height/bbox_width` 比例
  - bottle / vase → `is_volume == True` + 內部射線測試（中空檢查）
  - mug → 把手檢查：`(凸包 - 實體).body_count >= 1`
  - teapot → 同 mug + 壺嘴檢查
  - keychain → 厚度 < 5mm + 有孔（topology genus >= 1）
- [ ] `app.generate()` 流程：exec → mesh_repair → **geom_check** → vlm_judge
  - geom_check fail 直接 retry with hint，跳過 VLM（省 token）
- [ ] `tests/test_judge_geometric.py`：每條規則一個正例 + 一個負例

驗收：
- chair 應該抓到 < 4 條腿的錯誤（baseline benchmark 中至少 1 case）
- mug 應該抓到沒有把手的錯誤
- VLM call 數降 30~40%
- 既有 4 個 unit test 套件全綠

成本：M（兩天）— 規則寫 6 條，每條 0.5 天  
風險：規則太嚴 → false positive 卡住正確的 case；先 conservative + 加 `--strict` flag

## 6.2 Mesh 後處理鏈擴充（PyMeshLab + Open3D + manifold3d）

**目標**：從「watertight」拉到「印表機友善」。

- [ ] `mesh_repair.py` 加 Pass 3（在 trimesh + pymeshfix 之後）：
  - PyMeshLab `simplification_quadric_edge_collapse` — 高 poly count 的 mesh 降到 50K verts 以下，加快 slice
  - Open3D `remove_duplicated_vertices` + `compute_vertex_normals`
  - manifold3d boolean — 處理 CSG 造成的 self-intersection
- [ ] 加 `print_readiness_check.py`：
  - 牆厚 < 1.2mm 警告（用 ray casting 估）
  - 懸空 > 45° 警告（需 support）
  - body_count > 1 警告（多體要分開印或加 bridge）
- [ ] `/api/generate` 回傳加 `print_warnings: [...]`
- [ ] 前端在 STL 預覽下方顯示警告 chips

驗收：
- PyMeshLab pass 對 vertex count > 50K 的 STL，輸出 < 50K 且 watertight 保持
- 印表機警告 list 對 5 個已知 case（ex: 薄壁的 keychain）正確命中

成本：M（兩天）  
風險：PyMeshLab 安裝可能有 GLIBC issue（純 Linux 比 macOS 多坑）→ 包成 optional 像 pymeshfix 一樣

## 6.3 Slicer 整合（PrusaSlicer / OrcaSlicer headless）

**目標**：印表機就緒檢查：能切片成功 = 真的能印。

- [ ] `slicer_check.py`：subprocess call PrusaSlicer CLI（`--export-gcode --info`）
- [ ] 解析 stderr 抓 error/warning（"manifold needed"、"too thin" 等）
- [ ] `config.json` 加 `slicer_path: "/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer"`
- [ ] 流程：mesh_repair → slicer_check（optional）→ judge
- [ ] 如果 slicer 沒裝、跑不起來 → 警告但繼續，不擋 generate

驗收：
- 已知 watertight + 結構 OK 的 STL 能切片成功
- 已知有問題的 STL（自己手打洞測試）切片失敗，warning 正確進到回應

成本：M（一天）  
風險：使用者沒裝 slicer → optional gracefully degrade

---

# Sprint 7 — 健壯性週

目標：產量、可追蹤、可信任。  
特性：直接連到 SLO 與成本控管。

## 7.1 Best-of-N 取樣

**目標**：對 unstable shape (figurine, bottle, teapot) 做 N=3 並發，取最高分。

- [ ] `app.generate()` 加 `best_of: int = 1` 參數
- [ ] `asyncio.gather` 並發跑 N 個 LLM call（不同 temperature 0.3 / 0.7 / 1.0）
- [ ] 全部過 mesh_repair + geom_check，再 VLM judge，挑分最高
- [ ] `config.json` 加 `best_of_per_category: {"figurine": 3, "bottle": 3}` — 只對歷史不穩定 shape 開 N>1
- [ ] `/api/stats` 加 `best_of_invocations` 欄位
- [ ] 加 `tests/test_best_of_n.py` 至少 1 個 mock 測試

驗收：
- figurine + bottle 平均分數 +1.0 以上（trials=3 對比）
- token 成本對應 +N 倍，記錄在 stats
- best_of=1 行為與舊版完全一致

成本：M（兩天）  
風險：並發失敗一個會拖累整體 latency；要 timeout each candidate

## 7.2 Observability 升級（Langfuse / Arize Phoenix）

**目標**：每個 generation 都能 trace 回放，分析「為什麼這個 prompt 卡了 3 次」。

選一個（推薦 Langfuse — open source、self-host 友善）：
- [ ] `pip install langfuse`，docker-compose 起 self-host instance
- [ ] `app.py` 加 `@langfuse_trace` 包 generate / judge / mesh_repair
- [ ] 結構化記錄：`(prompt, model, system_prompt, generated_code, exec_ok, judge_score, latency_breakdown)`
- [ ] 加 dashboard view：per-shape pass rate、token spend over time、failover events

驗收：
- 任何一筆 generation 可在 Langfuse UI 上完整回放（prompt、retry 的 hint、最終 code）
- 跑 1 週後可從 Langfuse 看到 trend

成本：M（兩天）  
風險：self-host 增加維運負擔；可選擇 Langfuse cloud 免費額度作為起點

## 7.3 Sandbox 執行升級（RestrictedPython）

**目標**：把現有 AST allowlist 升級為更嚴格的 sandbox，公網部署時不被 LLM 噴的 code 玩壞。

- [ ] `pip install RestrictedPython`
- [ ] `validators.py` 加 `compile_restricted_safe(code)` — 比現有 allowlist 多一層 builtin 隔離
- [ ] 限制：no `__import__`、no `getattr` 動態取、no file write 除了 outputs/
- [ ] feature flag `config.sandbox_strict = true|false`
- [ ] 跑 baseline benchmark 確認 strict 模式下 pass rate 不降

驗收：
- 故意餵 `__import__("os").system("rm -rf /")` 必須被擋
- 故意餵 `open("/etc/passwd").read()` 必須被擋
- 12 個 benchmark shape 在 strict 模式下 pass rate 與 lenient 持平 (±2pp)

成本：M（一天半）  
風險：RestrictedPython 太嚴可能擋掉 CadQuery 合理用法 → 先 shadow mode 跑一週，比對結果再切換

---

# Sprint 8 — 新能力週

目標：把產品從「文字 → STL」擴成「多模態 → 多後端 → 多輸出」。  
特性：架構變動最大，建議 Sprint 5~7 都穩了再啟動。

## 8.1 Hybrid backend：3D-native 模型走有機形狀

**目標**：figurine / shoe / 玩偶 / 動物這類 CadQuery 吃力的有機形狀，改走 3D-native 模型。

- [ ] 新增 `backends/hunyuan3d_backend.py`（或 trellis_backend.py）
  - 走 HuggingFace Inference API 或 self-host
  - 介面：`generate(prompt: str) -> Path[stl]`，回傳 mesh path
- [ ] `pattern_cache.infer_category` 已有；新增 `_route_backend_for_prompt(prompt)`：
  - figurine / shoe / animal / toy / character → hunyuan3d
  - 其他 → cadquery
- [ ] `config.json` 加 `backend_routing: {"figurine": "hunyuan3d", ...}`，feature flag `backend_routing_enabled`
- [ ] hybrid 後不適用 AST validate（mesh 不是 code）→ 流程分支，重用 mesh_repair + judge
- [ ] 對 hybrid path 的 STL 做特別 print readiness 檢查（3D-native 出來通常 poly count 高、可能有 floating geometry）
- [ ] benchmark 12 shapes：對 figurine / shoe 應該大幅改善，其他持平

驗收：
- figurine 在 hunyuan3d 路徑下 watertight rate ≥ 80%（CadQuery baseline 0%）
- chair / table 仍走 cadquery，行為不變
- 兩條路徑 latency 各自 < 90s

成本：L（三天）  
風險：
- HuggingFace API 配額 / 費用結構與目前 OpenAI-compat 不同 — 要新加 provider 抽象
- 3D-native 模型 output mesh 沒有 parametric — user 不能再改尺寸（接受 trade-off，UI 警告）
- 自部署 Hunyuan3D 需 GPU；沒 GPU 場景只能 API 路徑

## 8.2 Sketch-to-3D 輸入

**目標**：使用者畫線稿（或上傳線稿）→ 3D。

- [ ] 前端 `static/index.html` 加 canvas 繪圖區（fabric.js 或 純 canvas）
- [ ] `/api/generate-from-sketch` endpoint：image (PNG) → image-to-3D
- [ ] 後端走 Stable Zero123 / Magic123 / SyncDreamer（HF API）
- [ ] 回傳沿用 generate flow 的 mesh_repair + judge

驗收：
- 畫一個簡單的杯子線稿能生出可辨識的 mug
- 失敗有清楚錯誤訊息（「請畫得清楚一點」之類）

成本：M（兩天）

## 8.3 Image-to-3D 輸入

**目標**：拍/上傳一張椅子照片 → 模型。

- [ ] 前端加上傳 + 拍照（getUserMedia）按鈕
- [ ] 後端 `/api/generate-from-image` endpoint
- [ ] 同 8.2 的 image-to-3D 模型，照片 input 通常 native 支援
- [ ] 加 referer 防護，避免被當白嫖 image-to-3D 服務

驗收：
- 上傳 1 張清楚的物件照片能生 3D
- 模糊照片 / 多物件照片有清楚錯誤

成本：M（一天，與 8.2 共用大半 backend code）

## 8.4 Voice → text 輸入

**目標**：手機端可講話直接 generate。

- [ ] 前端用 Web Speech API（Chrome 原生支援）
- [ ] 加錄音按鈕，Whisper.cpp / OpenAI Whisper API fallback
- [ ] 講「給我一個花瓶」→ 自動填入 prompt 框 + 觸發 generate

驗收：
- Chrome 桌面 + iOS Safari 都能用
- 中英文識別都 OK

成本：S（半天）— 主要是前端工

---

# 跨 Sprint 共用設施

## 統一 feature flag 表（更新到 `/api/stats`）

```json
"feature_flags": {
  "mesh_repair": true,         // S4
  "watertight_gate": true,     // S3
  "judge": true,
  "ast_validate": true,
  "pattern_cache": true,
  "shape_routing": true,       // S4 P3

  "output_cache": false,       // S5.1
  "multi_format_export": false,// S5.2
  "render_pyvista": false,     // S5.4
  "geom_check": false,         // S6.1
  "slicer_check": false,       // S6.3
  "best_of_n": false,          // S7.1
  "langfuse": false,           // S7.2
  "sandbox_strict": false,     // S7.3
  "backend_routing": false,    // S8.1
  "sketch_input": false,       // S8.2
  "image_input": false,        // S8.3
  "voice_input": false         // S8.4
}
```

每個都是「翻一行 config 就能停」的單獨開關，回滾零風險。

## 共用 unit test 規範

- 每個新模組必須有對應 `tests/test_<module>.py`
- 每加一個 feature flag，至少 1 個「flag=False 行為與舊版一致」的測試
- Sprint 5~8 結束時 unit test 總數應 ≥ **75 tests**（目前 37）

## 文檔更新節奏

- 每個 Sprint 結束更新 `SPRINT_<n>_RESULTS.md`，沿用 Sprint 3+4 的格式（headline numbers / 對照表 / 測試清單 / 配置 / 下一步）

---

# 取捨建議

如果預算只夠跑 1 個 Sprint，建議：**Sprint 5（機械工週）**。  
理由：4 項都是純機械工，立刻能感受到，沒有架構風險，後續 Sprint 都受惠（CI 會抓 regression、cache 降本、render 升級提升判斷準確率）。

如果預算 2 個：5 + 6（機械工 + 品質）。Sprint 6 的 1.3 程式化驗證是純白拿的零 token 質量提升。

如果預算 3 個：5 + 6 + 7。Sprint 7 的 best-of-N 對 figurine/bottle 這種尾部 case 有 step-function 提升。

Sprint 8 因為架構變動大、引入新依賴（GPU / 3rd-party API），值得獨立評估時機。
