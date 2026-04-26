"""VLM-based visual judge for generated 3D models.

Given a user's description + rendered views of the produced model,
ask a vision LLM whether the model is recognisable as the described object,
scores 1-10, and suggests a concrete fix if poor.

Uses an OpenAI-compatible Vision API (MiniMax-VL, Gemini, GPT-4o etc).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import httpx

from rendering import encode_pngs_as_data_urls

log = logging.getLogger("text2stl.judge")


JUDGE_SYSTEM_PROMPT = """You are a strict but fair 3D model reviewer.
The user wants a specific real-world object to be 3D-printed.
You will be shown 4 rendered views of a generated 3D model (isometric, front, side, top).

Your job: judge whether the shape is RECOGNISABLE as the described object.
Ignore surface texture, color, small blemishes — focus on overall silhouette and key structural features.

Respond with a STRICT JSON object. Do not include any text before or after the JSON.
The JSON schema:
{
  "identifiable": true | false,
  "category": "what you actually see (e.g. 'vase', 'blob', 'flat plate')",
  "match_score": integer 1-10 (10 = perfect match, 1 = totally wrong),
  "geometry_issues": ["short bullet points about what's wrong, if anything"],
  "fix_suggestion": "one short sentence that NAMES the CadQuery operation to use"
}

When writing `fix_suggestion`, be CONCRETE and NAME the operation. The
downstream code-gen LLM needs a clear directive, not generic advice.

Good fix_suggestion examples:
- "Use LOFT between two DIFFERENT foot-shaped outlines, not REVOLVE of a circle."
- "Add a SWEEP handle attached to the outer wall at Z=25 and Z=75."
- "Hollow the interior with .cut() of a smaller cylinder; it's currently solid."
- "Build as COMPOSITE of 4 leg boxes + seat slab, unioned."
- "Re-extrude the side profile with an angled back (~75°) instead of a vertical wall."

Bad fix_suggestion examples (too vague — DO NOT write these):
- "Improve the shape."
- "Make it look more like a shoe."
- "Adjust dimensions."
"""


@dataclass
class JudgeResult:
    identifiable: bool
    category: str
    match_score: int | None  # None signals "judge unavailable" (API error)
    geometry_issues: list[str]
    fix_suggestion: str
    raw_response: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_response", None)
        return d

    @classmethod
    def from_response(cls, text: str) -> "JudgeResult":
        """Parse the VLM's JSON reply. Tolerates fence markers and extra text."""
        raw = text.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*\n", "", raw)
            raw = re.sub(r"\n```\s*$", "", raw)
        # Extract the first {...} block in case there's preamble
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse judge JSON ({e}). Falling back to lenient parse.")
            return cls(
                identifiable=False,
                category="parse_error",
                match_score=3,
                geometry_issues=[f"Judge returned non-JSON: {text[:100]}"],
                fix_suggestion="Regenerate — previous output was not parseable.",
                raw_response=text,
            )
        # match_score may come back as int, float, or stringified int
        raw_score = data.get("match_score", 5)
        try:
            score: int | None = int(raw_score)
        except (TypeError, ValueError):
            score = None
        return cls(
            identifiable=bool(data.get("identifiable", False)),
            category=str(data.get("category", "unknown")),
            match_score=score,
            geometry_issues=list(data.get("geometry_issues", [])),
            fix_suggestion=str(data.get("fix_suggestion", "")),
            raw_response=text,
        )


# HTTP statuses that deserve a retry (overload / rate limit / transient)
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


async def _post_judge_once(
    client: httpx.AsyncClient,
    api_base: str,
    headers: dict,
    payload: dict,
) -> httpx.Response:
    return await client.post(f"{api_base}/chat/completions",
                             headers=headers, json=payload)


async def judge_model(
    user_description: str,
    view_paths: list[Path],
    vision_specs: list[dict] | None = None,
    # Legacy single-provider args (still supported for backwards compat)
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    fallback_models: list[str] | None = None,
    timeout: float = 120.0,
) -> JudgeResult:
    """Call the cloud vision API and get a JudgeResult.

    `vision_specs` is the preferred form: a list of
        {"model": "<name>", "api_base": "...", "api_key": "..."}
    dicts. They are tried in order; each entry retries transient errors
    up to 3 times with exponential backoff. This lets the judge fall
    back ACROSS providers (e.g. Gemini-lite → MiniMax-VL → Gemini-flash)
    so one provider's quota exhaustion doesn't break the judge.

    Legacy shape (`model`/`api_base`/`api_key` + `fallback_models`) is
    auto-expanded into a single-provider spec list.
    """
    if not view_paths:
        raise ValueError("No rendered views to judge")

    # Normalise inputs into a single spec list
    if not vision_specs:
        if not model or not api_key:
            log.warning("No vision API key/model configured; skipping judge")
            return JudgeResult(
                identifiable=True,
                category="judge_disabled",
                match_score=None,
                geometry_issues=[],
                fix_suggestion="",
            )
        vision_specs = [
            {"model": m, "api_base": api_base, "api_key": api_key}
            for m in [model] + list(fallback_models or [])
        ]

    # Filter out entries that are missing essentials
    vision_specs = [
        s for s in vision_specs
        if s.get("model") and s.get("api_base") and s.get("api_key")
    ]
    if not vision_specs:
        log.warning("Vision judge: no usable providers; skipping")
        return JudgeResult(
            identifiable=True,
            category="judge_disabled",
            match_score=None,
            geometry_issues=[],
            fix_suggestion="",
        )

    data_urls = encode_pngs_as_data_urls(view_paths)
    content = [
        {"type": "text", "text": f"The user wants this object: {user_description!r}. "
                                   "Judge the rendered views below."}
    ]
    for url in data_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    last_status = 0
    last_body = ""

    async with httpx.AsyncClient(timeout=timeout) as client:
        for spec in vision_specs:
            mdl = spec["model"]
            spec_base = spec["api_base"]
            spec_key = spec["api_key"]
            headers = {
                "Authorization": f"Bearer {spec_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": mdl,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                # 2048 leaves room for Gemini-2.5's hidden "thinking" tokens.
                "max_tokens": 2048,
                "temperature": 0.2,
            }
            # up to 3 attempts per model with exponential backoff on transient errors
            for attempt in range(3):
                log.info(f"Calling vision judge: {mdl} "
                         f"(try {attempt+1}/3, desc={user_description[:40]!r})")
                try:
                    resp = await client.post(
                        f"{spec_base}/chat/completions",
                        headers=headers, json=payload,
                    )
                except httpx.HTTPError as e:
                    last_status = -1
                    last_body = str(e)
                    log.warning(f"Judge HTTP error ({mdl}, try {attempt+1}): {e}")
                else:
                    if resp.status_code == 200:
                        data = resp.json()
                        text = (data.get("choices", [{}])[0]
                                    .get("message", {})
                                    .get("content") or "").strip()
                        log.info(f"Judge response ({mdl}): {text[:150]!r}")
                        return JudgeResult.from_response(text)
                    last_status = resp.status_code
                    last_body = resp.text[:300]
                    log.warning(f"Judge API {resp.status_code} ({mdl}, try {attempt+1}): "
                                f"{last_body[:150]}")
                    if resp.status_code not in _RETRYABLE_STATUSES:
                        break  # auth / bad request — don't waste retries on this spec
                # backoff: 2s, 5s
                if attempt < 2:
                    await asyncio.sleep(2.0 * (2 ** attempt))
            log.warning(f"Judge: giving up on {mdl}, trying next spec if any")

    log.error(f"Judge: all models exhausted, last status={last_status}")
    # S1.1: Do NOT silently return match_score=6 (which accidentally passed
    # the JUDGE_MIN_SCORE gate). Use None so the caller can distinguish
    # "judge says pass" from "judge never ran".
    return JudgeResult(
        identifiable=False,
        category="judge_api_error",
        match_score=None,
        geometry_issues=[f"Judge API returned {last_status} after retries"],
        fix_suggestion="",
    )


def build_retry_instruction(description: str, previous_code: str, judge: JudgeResult) -> str:
    """Build a user message that feeds the judge's feedback back to the LLM."""
    issues = "\n".join(f"- {i}" for i in judge.geometry_issues) or "- (none listed)"
    score_s = f"{judge.match_score}/10" if judge.match_score is not None else "unscored"
    return (
        f"The previous code produced a 3D model that was judged as: "
        f"{judge.category!r} (score {score_s}).\n"
        f"Issues the visual reviewer identified:\n{issues}\n"
        f"Suggested fix: {judge.fix_suggestion}\n\n"
        f"Please regenerate code that produces a model clearly recognisable as "
        f"\"{description}\". Output ONLY the complete corrected Python code."
    )
