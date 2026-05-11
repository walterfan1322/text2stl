"""Vision critic: compare a Flux reference image to a candidate STL render
and produce structured `{feature, answer, evidence}` complaints that the
next code-generation attempt can act on.

This is the P3.2 fine-grained signal layer. The split:

    silhouette IoU  → "is it shaped right at all?"      (cheap, deterministic)
    judge (4-view)  → "is this thing recognisable?"     (medium cost, 1 image set)
    image_critic    → "what specifically is missing?"   (1 vision call, 2 images)

The critic is only worth its latency (~2-5s) when both:
    a) the silhouette is in the right ballpark (IoU ≥ 0.10), AND
    b) the judge said the model isn't recognisable yet.

Mirrors `judge.py`'s tier-fallback chain so one provider's quota
exhaustion doesn't kill the critic.

The prompt is intentionally GENERAL — the critic decides which features
to check based on what it sees in the reference. No per-category rules
live in this file (consistent with the project directive).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

import httpx

log = logging.getLogger("text2stl.image_critic")


CRITIC_SYSTEM_PROMPT = """You are a structural critic for 3D models. You will see TWO images:
- REFERENCE: an idealised render showing the object the user asked for.
- CANDIDATE: a render of an LLM-generated 3D model attempting to match it.

Your job: identify 3–5 distinctive STRUCTURAL features visible in the
REFERENCE, then check whether the CANDIDATE has each one. Then write 0–3
short imperative complaints telling the next code generator what to add
or fix.

Focus on geometry only:
- COUNT (how many legs / wheels / teeth / fingers / parts)
- TOPOLOGY (parts connected vs. floating apart)
- PROPORTION (tall vs. squat, thick vs. thin, large vs. small relative parts)
- PRESENCE (handle / lid / spout / hole / cavity present or missing)
- AXIS (correct upright orientation)

Ignore style, colour, lighting, surface finish, materials, and rendering
quality. Both images are clay-style renders for comparison only.

Output STRICT JSON only — no prose, no markdown fences:
{
  "match_quality": "good" | "partial" | "poor",
  "checks": [
    {
      "feature": "<short noun phrase, e.g. 'four cylindrical legs'>",
      "in_reference": true,
      "in_candidate": true,
      "evidence": "<one short sentence of visual evidence>"
    }
  ],
  "complaints": [
    "<imperative sentence describing what to add or fix in the next code attempt>"
  ]
}

Rules:
- `checks` must have 3–5 entries.
- `complaints` has 0–3 entries. Empty array if match_quality is "good".
- Complaints must be SPECIFIC and ACTIONABLE so a code generator can
  apply them. Bad: "make it look more like a chair." Good: "Replace
  the single block with four separate leg cylinders (each ~30mm tall)
  unioned to a flat seat slab."
- Complaints describe what TO DO to the candidate, not what the
  reference has. Write as if the reader will only see your text — they
  cannot see the images.
- match_quality:
    "good"    — silhouette and structure roughly match
    "partial" — silhouette ok but structure off (wrong part count,
                missing detail, wrong proportion)
    "poor"    — looks like a different object entirely
"""


@dataclass
class CritiqueResult:
    """Result of one critic call.

    `match_quality` carries one of four values. The first three come
    from the model itself; "critic_unavailable" is the soft-fail path
    used when every provider in the chain failed (auth/quota/network).
    Callers should treat "critic_unavailable" the same as "no critic
    feedback this round" — fall through to the judge's fix_suggestion
    or just plain "regenerate".
    """
    match_quality: str  # good | partial | poor | critic_unavailable
    checks: list[dict] = field(default_factory=list)
    complaints: list[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_response", None)
        return d

    @classmethod
    def unavailable(cls, reason: str) -> "CritiqueResult":
        return cls(match_quality="critic_unavailable", raw_response=reason)

    @classmethod
    def from_response(cls, text: str) -> "CritiqueResult":
        """Parse the VLM JSON reply. Tolerant to ```json fences and
        leading/trailing prose (some Gemini variants wrap JSON anyway).
        """
        raw = text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*\n", "", raw)
            raw = re.sub(r"\n```\s*$", "", raw)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning(f"critic JSON parse failed ({e}); text[:120]={text[:120]!r}")
            return cls(
                match_quality="critic_unavailable",
                raw_response=f"parse_error: {e}",
            )

        # Normalise match_quality
        mq = str(data.get("match_quality", "")).strip().lower()
        if mq not in ("good", "partial", "poor"):
            mq = "partial"  # safe default — neither passes nor catastrophe-fails

        # Normalise checks
        checks_raw = data.get("checks") or []
        checks: list[dict] = []
        if isinstance(checks_raw, list):
            for c in checks_raw:
                if not isinstance(c, dict):
                    continue
                checks.append({
                    "feature": str(c.get("feature", "")).strip()[:200],
                    "in_reference": bool(c.get("in_reference", False)),
                    "in_candidate": bool(c.get("in_candidate", False)),
                    "evidence": str(c.get("evidence", "")).strip()[:200],
                })

        # Normalise complaints
        comp_raw = data.get("complaints") or []
        complaints: list[str] = []
        if isinstance(comp_raw, list):
            for s in comp_raw:
                s = str(s).strip()
                if s:
                    complaints.append(s[:400])
        # Cap at 3 to keep retry message bounded
        complaints = complaints[:3]

        return cls(
            match_quality=mq,
            checks=checks,
            complaints=complaints,
            raw_response=text,
        )


# Same retry policy as judge.py — keep consistent across vision callers.
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def _png_to_data_url(path: Path) -> str:
    b = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")


async def critique(
    reference_path: Path,
    candidate_path: Path,
    user_description: str,
    vision_specs: list[dict],
    timeout: float = 90.0,
) -> CritiqueResult:
    """Run the vision critic, comparing reference vs candidate.

    `vision_specs` follows the same shape as judge.py:
        [{"model": "<name>", "api_base": "...", "api_key": "..."}]
    Tries each in order; up to 3 retries per spec on transient errors.
    """
    if not Path(reference_path).exists():
        return CritiqueResult.unavailable(f"missing reference: {reference_path}")
    if not Path(candidate_path).exists():
        return CritiqueResult.unavailable(f"missing candidate: {candidate_path}")

    vision_specs = [
        s for s in (vision_specs or [])
        if s.get("model") and s.get("api_base") and s.get("api_key")
    ]
    if not vision_specs:
        return CritiqueResult.unavailable("no vision providers configured")

    ref_url = _png_to_data_url(Path(reference_path))
    cand_url = _png_to_data_url(Path(candidate_path))

    # The user message orders REFERENCE first, then CANDIDATE — order is
    # called out in text so the model can't mix them up.
    user_content = [
        {"type": "text", "text": (
            f"The user requested: {user_description!r}.\n"
            f"Image 1 below is the REFERENCE (what was wanted).\n"
            f"Image 2 below is the CANDIDATE (what the 3D code produced).\n"
            f"Compare them and output the JSON described in the system message."
        )},
        {"type": "image_url", "image_url": {"url": ref_url}},
        {"type": "image_url", "image_url": {"url": cand_url}},
    ]

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
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                # Same as judge: leave room for Gemini-2.5 hidden thinking.
                "max_tokens": 1536,
                "temperature": 0.2,
            }
            for attempt in range(3):
                log.info(f"Calling vision critic: {mdl} "
                         f"(try {attempt+1}/3, desc={user_description[:40]!r})")
                try:
                    resp = await client.post(
                        f"{spec_base}/chat/completions",
                        headers=headers, json=payload,
                    )
                except httpx.HTTPError as e:
                    last_status = -1
                    last_body = str(e)
                    log.warning(f"Critic HTTP error ({mdl}, try {attempt+1}): {e}")
                else:
                    if resp.status_code == 200:
                        data = resp.json()
                        text = (data.get("choices", [{}])[0]
                                    .get("message", {})
                                    .get("content") or "").strip()
                        log.info(f"Critic response ({mdl}): {text[:150]!r}")
                        result = CritiqueResult.from_response(text)
                        log.info(
                            f"Critic verdict: {result.match_quality} "
                            f"({len(result.checks)} checks, "
                            f"{len(result.complaints)} complaints)"
                        )
                        return result
                    last_status = resp.status_code
                    last_body = resp.text[:300]
                    log.warning(f"Critic API {resp.status_code} "
                                f"({mdl}, try {attempt+1}): {last_body[:150]}")
                    if resp.status_code not in _RETRYABLE_STATUSES:
                        break
                if attempt < 2:
                    await asyncio.sleep(2.0 * (2 ** attempt))
            log.warning(f"Critic: giving up on {mdl}, trying next spec if any")

    log.error(f"Critic: all models exhausted, last status={last_status}")
    return CritiqueResult.unavailable(
        f"all_providers_failed last_status={last_status}"
    )


def build_critic_retry_block(
    description: str,
    iou: float | None,
    critique: CritiqueResult,
) -> str:
    """Build the text block that gets prepended/appended to the judge's
    retry message.

    Layout (only included when the critic actually returned complaints):

        VISUAL CRITIC FEEDBACK (silhouette IoU = 0.42):
          ✗ four cylindrical legs — reference has them, candidate is missing them
            (evidence: candidate has a single thick base, not four legs)
        Complaints to address in the next attempt:
          - Replace the single base block with four separate leg cylinders.
          - Make sure each leg attaches to the underside of the seat.

    Returns "" when the critic produced no useful feedback (so callers
    can append it unconditionally).
    """
    if critique.match_quality == "critic_unavailable":
        return ""
    if not critique.complaints and not critique.checks:
        return ""

    iou_str = f"{iou:.2f}" if isinstance(iou, (int, float)) else "n/a"
    lines: list[str] = [
        f"VISUAL CRITIC FEEDBACK (silhouette IoU = {iou_str}, "
        f"verdict = {critique.match_quality}):"
    ]

    # Show only the failing checks — passing ones are not informative
    # at retry time and would just dilute the LLM's attention.
    failing = [
        c for c in critique.checks
        if c.get("in_reference") and not c.get("in_candidate")
    ]
    for c in failing[:5]:
        ev = c.get("evidence") or ""
        lines.append(
            f"  ✗ {c['feature']} — present in reference, missing in candidate"
            + (f" (evidence: {ev})" if ev else "")
        )

    if critique.complaints:
        lines.append("Complaints to address in the next attempt:")
        for s in critique.complaints:
            lines.append(f"  - {s}")
    return "\n".join(lines)
