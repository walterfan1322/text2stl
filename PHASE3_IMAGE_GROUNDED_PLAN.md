# Phase 3 計畫書：Image-Grounded Self-Correction Pipeline

**起草日**：2026-04-29
**目標**：把目前 9.33 分（working-STL 平均）的 text-only 流程升級為 image-grounded loop，主攻「LLM 對抽象物件想像不夠具體」的問題。
**符合通用原則**：所有機制都是 prompt-agnostic，不分類別。

---

## 1. 目前流程的瓶頸（為什麼要做這個）

8-cat 調查顯示，當 LLM 寫得出有效程式時平均 9.33；但仍有兩種失敗：
- **LLM coding stochastic error**（dog/hammer 偶發 `revolve()`/`loft()` 退化）— 通用機制無法救，只能 best-of-N。
- **抽象想像差**（chair 早期把腿插穿座面、car 輪子位置全憑猜）— **這就是 image grounding 該救的**。LLM 如果有一張參考圖在前面，腿幾條、頭朝哪、輪距多寬都會具體化。

---

## 2. 架構（資料流）

```
prompt "一張椅子"
  │
  ├─[1] 譯 ZH→EN（DeepSeek-V4-Flash, 已有 enricher 改寫）
  │
  ├─[2] 生 4 張 canonical reference views（Flux-schnell @ fal.ai）
  │       front / side / iso / top，white bg, low-poly clay render style
  │
  ├─[3] VLM 看圖→寫 parts list（PLAN JSON 裡的 parts/sizes/attach）
  │       ← critic VLM（不是 DeepSeek，下面 §4 D2）
  │
  ├─[4] DeepSeek-V4-Pro（thinking）寫 CadQuery code
  │       輸入：原 prompt + parts list（純文字，DeepSeek 看不到圖）
  │
  ├─[5] exec → STL → render 同樣 4 個 canonical views
  │       ★ 跟 step 2 同 camera matrix 同 shading（這是 IoU 公平性的關鍵）
  │
  ├─[6] 比對：
  │       (a) silhouette IoU per view（硬性 threshold ≥ 0.65）
  │       (b) VLM Q&A on per-view diff（軟性、結構化抱怨）
  │
  └─[7] if fail and iter < 3:
        把 diff 描述塞回 step 4 retry（保留 best-IoU candidate）
        else: 輸出 best-so-far
```

---

## 3. 我原始想法的漏洞（survey 補齊的部分）

| 我原本的想法 | survey 戳破 / 補齊 |
|---|---|
| 「上網找參考圖」 | ❌ 版權風險 + 真實照片風格跟 CAD render 對不上（CLIP 會用風格落差打分而非幾何落差）。改用 **text-to-image 生圖**，prompt 強制 low-poly clay render 風格，跟 CadQuery output 視覺對齊。 |
| 「DeepSeek 看圖描述結構」 | ❌ DeepSeek API 不支援 image input。Janus-Pro-7B 雖然 DeepSeek 出的，但只在 fal.ai/Replicate 第三方上，且 MMBench/RealWorldQA 都輸 GPT-4o/Gemini/Claude 一截。**critic VLM 必須換家**。 |
| 「Render vs Image 比對」 | ⚠ 沒講清楚要怎麼比。SOTA 共識：**silhouette IoU 當硬指標**（CADrille），**VLM Q&A 當軟指標**（CADCodeVerify）。**單看 CLIP/DreamSim 在 CAD 場景會 collapse 到風格差不是幾何差**。 |
| 「自我修正、重複」 | ⚠ 沒講停止條件。CADCodeVerify 顯示 **fixed N（3-5）** 收益就 plateau。**必須有 monotone metric（IoU）做 best-so-far gate**，否則 loop 會越跑越爛。 |
| 「同一個 LLM 跑全程」 | ❌ Anti-pattern: same-family VLM 提案+評審會 self-agree 進 local optimum。CADSmith 明確分開兩家。**coder 與 critic 必須不同家**。 |
| Camera pose 不一致 | ❌ Flux 的 iso 跟 CadQuery render 的 iso 不會剛好對齊。直接做 IoU 會偏差很大。**先把 reference 圖也輸出成 4 個明確角度（用 prompt: "front view"/"side view"...），candidate 也 render 對應的固定 camera matrix**。 |
| 風格不匹配 | ❌ 真實照片有材質、陰影、地板，CAD render 沒有。Reference 圖必須生成「**flat shading, no textures, white background, blender clay**」風格才能對齊。 |

---

## 4. 關鍵設計決策

### D1：參考圖來源 → **fal.ai Flux-schnell**
- 不用 web search（版權 + 風格不對）
- 不用 DALL-E 3（會加場景背景，難控制；貴）
- 不用 SDXL（背景髒）
- 不用 Janus-Pro 自己生（品質落後 Flux）
- **Flux-schnell @ fal.ai**：$0.003/image × 4 views = **$0.012/prompt**，~3s latency
- Prompt 模板：
  ```
  isometric view of a single {object_en}, low-poly 3D model,
  flat shading, matte gray, pure white background,
  centered, full object visible, blender clay render style
  ```

### D2：VLM 角色分離 → 必須非 DeepSeek
DeepSeek API 不支援 image，所以 critic 一定要換家。三個選項，**等使用者決定**：

| 選項 | 成本 | 品質 | 風險 |
|---|---|---|---|
| **(A) Gemini 2.5 Flash API** | ~$0.005/check | 好（多模態強） | 需新 API key |
| **(B) 195 自架 Janus-Pro-7B** | 免費 | 中等（VL 比 GPT-4o 弱） | 195 GPU 占用、setup 工 |
| **(C) 只用 silhouette IoU 不用 VLM critic** | $0 | 弱（看不到結構抱怨） | 訊號退化 |

**我的建議**：先做 (A) Gemini 2.5 Flash — 便宜可靠、跟 DeepSeek 不同家、API key 一個就夠。如果 cost 是問題，後續改 (B)。

### D3：比對指標
- **硬指標 silhouette IoU**：reference 跟 candidate 都先二值化（>0.5 alpha → 1）再算 IoU
  - per-view threshold = 0.65（容忍輕微 pose 差）
  - 4 views 平均 ≥ 0.7 算過
- **軟指標 VLM Q&A**：把 diff 圖（reference XOR candidate）傳給 critic VLM
  - critic 產生結構化抱怨：「front view 顯示 candidate 缺一個 backrest」
  - 抱怨文字塞回 retry hint，不直接用來打分

### D4：Loop control
- `N_max = 3`（CADCodeVerify plateau 點）
- **best-of-N gate**：每 iter 都記 IoU，最後輸出 IoU 最高的版本
- early exit if mean IoU > 0.85
- Retry hint 形式：
  ```
  Per-view IoU: front=0.82 side=0.71 iso=0.45 top=0.68
  Critic notes: "iso view shows candidate missing the backrest entirely;
  side view shows legs too short."
  Adjust the PLAN and code accordingly.
  ```

---

## 5. DeepSeek API 在這計畫裡的角色

| 步驟 | 用什麼 | 為什麼 |
|---|---|---|
| ZH→EN translate | DeepSeek V4-Flash | 便宜、文字夠了 |
| 圖→parts list | **不用 DeepSeek**（沒 vision） | Gemini/GPT-4o/Janus 任一 |
| CadQuery code 生成 | **DeepSeek V4-Pro thinking** | 1M context、cache hit $0.0036/1M、reasoning 強 |
| Render 比對 | **不用 DeepSeek**（沒 vision） | 同 critic VLM |
| Retry hint LLM | DeepSeek V4-Pro thinking | 同 coder（可以 cache 命中） |

→ DeepSeek 負責純文字推理（最強項），vision 工作外包給其他家。

---

## 6. 我需要使用者拍板的決策

1. **要不要付 fal.ai 費用**做圖片生成？預估每次 prompt $0.012，跑 100 次 prompt = $1.2。（替代方案：在 195 上自架 Flux-schnell，需要 12GB VRAM；195 是 RTX 4090 嗎？）
2. **critic VLM 用哪個**？(A) Gemini API key、(B) 自架 Janus-Pro-7B、(C) 只用 IoU 不用 VLM critic
3. 要先做 **小規模驗證**（單一類別如 chair 從頭跑 image-grounded loop）還是 **直接做完整 8-cat 比較**？

---

## 7. 分階段實作

### Phase 3.1（1-2 天）— Minimal viable image grounding
- 加 `image_gen.py`：fal.ai Flux-schnell 包裝，產 4 視角 PNG
- 加 `silhouette_iou.py`：兩張 PNG → IoU 浮點數
- 加 `_render_canonical_views(stl_path)`：CadQuery → 4 PNG（重用現有 view code）
- 改 `app.py`：在 retry loop 加 IoU gate（先不接 VLM critic）
- 跑 chair 單測：觀察 IoU 訊號是否能驅動 retry 改善
- 跑 8-cat survey：對照 P2 baseline

### Phase 3.2（1 天）— VLM critic
- 加 `vlm_critic.py`：呼叫選定的 VLM（A/B/C）
- 加 diff 視覺化（PIL XOR + colorise）
- 把 critic 結構化抱怨塞 retry hint
- 重跑 8-cat

### Phase 3.3（後續）— Multi-view consistency
- 把 4 張獨立 Flux gen 換成 MV-Adapter 一次出 4 張
- 預期能再降 IoU 雜訊

---

## 8. 風險登記

| 風險 | 緩解 |
|---|---|
| Flux 生出來的椅子是寫實風 | Prompt 強制 "low-poly, flat shading, blender clay" |
| Pose mismatch（Flux iso ≠ CadQuery iso 角度） | Reference 4 個視角用明確 prompt（"front view"/"side view"），candidate render 用對應的固定 camera matrix；接受 0.6-0.7 IoU 為門檻而非 0.9 |
| VLM critic 跟 coder self-agree | 強制不同家（DeepSeek coder + Gemini/Janus critic） |
| 加了 image gen 反而拖慢 | image gen 並行做（同時 enricher 跑），預估 +3-5s 不超過 10% latency |
| Cost 失控 | 每 prompt cap：1 次 Flux 4 views + 最多 3 次 retry × VLM critic =  ~$0.05/prompt 上限 |
| 流程變複雜後新失敗類型 | feature flag `image_grounded_mode` 預設 off，A/B 對照才開 |

---

## 9. 成功指標

| 指標 | 現在 (P2) | 目標 (P3) |
|---|---|---|
| Working-STL mean | 9.33 | ≥9.5（可能已 saturate）|
| 「Looks like」率（identifiable）| 6/8 | 8/8 |
| 「Wrong proportions / wrong layout」失敗 | 仍有（chair 早期、car wheels 早期）| 接近 0 |
| LLM stochastic exec_failed | 2/8 | 不變（這次正交問題，留給 best-of-N）|
| 平均 latency | 2-5 min | <4 min |
| 平均 cost / prompt | ~$0.001 | <$0.05 |

---

## 10. Survey 來源（給後人查）

- DeepSeek API 實況（2026-04 V4 release，無 vision/no image gen）：https://api-docs.deepseek.com/quick_start/pricing
- CADCodeVerify（VLM Q&A + Chamfer loop）：https://arxiv.org/abs/2410.05340
- CADSmith（multi-agent + VLM Judge against text spec）：https://arxiv.org/abs/2603.26512
- CADrille（IoU as RL reward）：https://arxiv.org/abs/2502.09819 (AIDL related)
- Img2CAD（image→CAD with VLM factorization）：https://arxiv.org/abs/2408.01437
- GenCAD（image-conditioned CAD diffusion）：https://arxiv.org/abs/2409.16294
- Flux.1-schnell：https://huggingface.co/black-forest-labs/FLUX.1-schnell
- MV-Adapter（multi-view consistency）：https://huggingface.co/huanngzh/MV-Adapter
- Janus-Pro-7B：https://huggingface.co/deepseek-ai/Janus-Pro-7B
