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


JUDGE_SYSTEM_PROMPT = """You are a strict but fair 3D model reviewer using a CHECKLIST protocol.
The user wants a SPECIFIC real-world object to be 3D-printed and your job is to
verify that the rendered model actually matches THE OBJECT THE USER REQUESTED —
NOT to describe what the silhouette happens to look like.

You will be shown 4 rendered views of a generated 3D model (isometric, front,
side, top) plus the user's request text.

VIEW USAGE GUIDE:
- ISOMETRIC view shows the overall 3-D form. A 3-D box from iso angle
  has a diamond/pyramid-shaped outline; do NOT mistake that 2-D outline
  for the actual shape. Use the iso view together with the orthographic
  views to assess form.
- FRONT and SIDE views show the silhouette without 3-D foreshortening.
  These are reliable for height-to-width proportions and overall shape.
- TOP view is the PRIMARY view for assessing flat surface patterns,
  grids, arrays of features (chessboard squares, keyboard keys, button
  layouts, tile patterns). Surface bumps that are too short to see in
  iso are usually clearly visible from the top via shading edges.

When checking for a "checker pattern", "grid", or "array", look at the
TOP view for visible cell-edge shading. A truly featureless slab will
have a uniform top face; a chessboard will show 64 cell boundaries.

PROTOCOL — you will do THREE things, in order, in a SINGLE JSON reply:

(0) IDENTIFY THE REQUESTED OBJECT TYPE from the user's request (e.g.
    "西洋棋盤" → "chessboard"; "a water bottle" → "bottle"). All checks
    you propose in step 1 must be checks for THIS REQUESTED TYPE,
    regardless of what the rendered silhouette appears to show.

(1) PROPOSE 4-6 binary structural checks the object MUST satisfy to be
    recognisable as THE REQUESTED OBJECT TYPE. Each check is a yes/no
    question about geometry — not surface finish, not color, not mesh
    quality. Phrase each check so that "yes" = the model matches the
    REQUESTED object's expectations.

    The FIRST check is MANDATORY and FIXED:
      "Does the overall silhouette resemble a <REQUESTED OBJECT TYPE>?"
    Answer this honestly based on the rendered views, **using the FRONT,
    SIDE, and TOP views as primary evidence — NOT the isometric view's
    diamond-shaped outline**. (Iso outline of a flat slab is a diamond,
    which can be mistaken for a pyramid; check the side / front view to
    confirm.) If the model is obviously a different shape (e.g. a true
    pyramid when the user asked for a chessboard, where the side view
    shows tapered triangular profile), the answer MUST be "no". For
    flat / planar objects (chessboard, keyboard, plate, tile), a flat
    slab profile in side/front view is CORRECT — answer "yes" — but
    THEN check the top view for the expected pattern in subsequent
    checks; do NOT pass an unpatterned slab.

    Checks 2-N are object-specific structural checks for the requested
    type (presence of distinctive parts, part count, attachment topology,
    proportions, axis orientation, cavities for containers, etc.).

(2) ANSWER each check with "yes", "no", or "unclear", citing one short
    evidence string from the rendered views. "unclear" is for cases where
    none of the four views can resolve the question (occluded, tiny, etc).

Then respond with a STRICT JSON object. No prose before or after the JSON.

Schema:
{
  "requested_object": "<short noun phrase, normalized from the user request>",
  "rendered_silhouette": "<what the silhouette actually depicts, in your honest opinion>",
  "checks": [
    {"q": "Does the overall silhouette resemble a <requested_object>?",
     "answer": "yes" | "no" | "unclear",
     "evidence": "<short, view-grounded reason>"},
    {"q": "<object-specific structural check>", "answer": "...",
     "evidence": "..."},
    ...
  ],
  "fix_suggestion": "<one CONCRETE CadQuery directive for the failed checks; NAME the op>"
}

Rules for `checks`:
- Provide 4-6 checks (no fewer, no more), with the silhouette-resemblance
  check as the FIRST one.
- Each `q` must be answerable from the four views alone.
- For container objects (planter, vase, mug, bowl, cup, box): one of the
  remaining checks MUST ask about a hollow interior / open top.
- For multi-part objects (chair, car, hammer): one of the remaining checks
  MUST ask about attachment / part-count.
- Do NOT pad with cosmetic checks ("is it smooth?"). Only structural.

Rules for `answer`:
- Be honest. If the silhouette obviously does NOT match the requested
  object, answer "no" on the first check and let the score reflect it.
- "unclear" counts as a half-pass — use it only when truly ambiguous.

Rules for `fix_suggestion`:
- Concrete CadQuery operation. Bad: "make it more like a chair." Good:
  "Add 4 separate leg boxes unioned to a seat slab; current legs are
  embedded in the seat."
- If all checks pass, set fix_suggestion to "" (empty string).

Examples of good check sets:
- Request "chessboard" (FIRST check fixed, then 3-5 type checks):
  1. "Does the overall silhouette resemble a chessboard?"
  2. "Is there an 8x8 grid of square cells visible on the top face?"
  3. "Do alternating cells differ from their neighbours in height or color?"
  4. "Is the overall shape a flat square slab (not a pyramid or block)?"
- Request "chair" (FIRST check fixed, then 3-5 type checks):
  1. "Does the overall silhouette resemble a chair?"
  2. "Are there exactly 4 distinct legs visible from below?"
  3. "Is the seat attached on top of the legs (not floating, embedded)?"
  4. "Is there a backrest extending upward from one edge of the seat?"
- Request "water bottle" (FIRST check fixed, then 3-5 type checks):
  1. "Does the overall silhouette resemble a water bottle?"
  2. "Is the body roughly cylindrical with a narrower neck on top?"
  3. "Is there a visible hollow interior / opening at the top?"
  4. "Is the height-to-width ratio approximately 2:1 to 4:1?"
"""


# P1 (2026-04-28): keywords that bump the clamp ceiling DOWN to 4.
# When a judge issue contains any of these, the model has fundamental
# structural problems and we don't want a score of 5 to slip through.
_SEVERE_ISSUE_KEYWORDS = (
    "not attached", "not connected", "disconnect", "floating", "detached",
    "missing", "wrong orientation", "wrong axis", "upside down",
    "intersect", "passes through", "embedded in",
    "no leg", "no head", "no body", "no handle", "no wheel",
    "lacks", "doesn't have", "does not have",
)


@dataclass
class JudgeResult:
    identifiable: bool
    category: str
    match_score: int | None  # None signals "judge unavailable" (API error)
    geometry_issues: list[str]
    fix_suggestion: str
    raw_response: str = ""
    # P1 (2026-04-28): server-side clamp telemetry. Kept for backwards
    # compat with dashboards that read this field. With the Q&A protocol
    # the score is derived from passed/total directly so clamp is rarely
    # invoked, but legacy `match_score`-only responses still go through it.
    clamped_from: int | None = None
    # P5 (2026-04-28): structured Q&A protocol fields. `checks` is the
    # list of {q, answer, evidence} dicts the VLM produced. `passed` and
    # `total` are derived counts. Empty list means the VLM returned a
    # legacy-format response (handled by fallback parser below).
    checks: list[dict] | None = None
    passed: int | None = None
    total: int | None = None
    # P3.5 (2026-04-29): category-match flag derived from the MANDATORY
    # first check ("Does the overall silhouette resemble a <requested>?").
    # `False` means the VLM judged the silhouette as NOT matching the
    # requested object — this MUST gate the retry loop regardless of the
    # other checks' scores. `None` means we couldn't infer a verdict
    # (legacy responses, parse error, or judge disabled).
    category_match: bool | None = None
    requested_object: str | None = None
    rendered_silhouette: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_response", None)
        return d

    @classmethod
    def from_response(cls, text: str) -> "JudgeResult":
        """Parse the VLM's JSON reply. Tolerates fence markers and extra text."""
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
            log.warning(f"Failed to parse judge JSON ({e}). Falling back to lenient parse.")
            return cls(
                identifiable=False,
                category="parse_error",
                match_score=3,
                geometry_issues=[f"Judge returned non-JSON: {text[:100]}"],
                fix_suggestion="Regenerate — previous output was not parseable.",
                raw_response=text,
            )

        # P3.5 (2026-04-29): preferred new fields, with backward-compat
        # fallback to the legacy `category` key.
        requested_object = data.get("requested_object")
        rendered_silhouette = data.get("rendered_silhouette")
        category = str(data.get("category",
                                rendered_silhouette or
                                requested_object or
                                "unknown"))
        fix_suggestion = str(data.get("fix_suggestion", ""))
        checks_raw = data.get("checks")

        # New Q&A protocol path
        if isinstance(checks_raw, list) and checks_raw:
            normalised: list[dict] = []
            for c in checks_raw:
                if not isinstance(c, dict):
                    continue
                ans = str(c.get("answer", "")).strip().lower()
                if ans not in ("yes", "no", "unclear"):
                    ans = "unclear"
                normalised.append({
                    "q": str(c.get("q", "")).strip()[:200],
                    "answer": ans,
                    "evidence": str(c.get("evidence", "")).strip()[:200],
                })
            total = len(normalised)
            # P3.5: derive category_match from the MANDATORY first check
            # ("Does the overall silhouette resemble a <requested>?").
            # We detect it heuristically by looking for "resemble" or
            # "look like" in the first check's question; if not found,
            # we leave category_match = None to avoid false positives.
            cat_match: bool | None = None
            if normalised:
                first_q = normalised[0]["q"].lower()
                if any(kw in first_q for kw in (
                        "resemble", "look like", "looks like",
                        "resembles", "appear as", "appear to be")):
                    first_ans = normalised[0]["answer"]
                    if first_ans == "yes":
                        cat_match = True
                    elif first_ans == "no":
                        cat_match = False
                    # "unclear" -> leave None (don't gate, but don't endorse)
            # yes = 1, unclear = 0.5, no = 0
            score_units = sum(
                1.0 if c["answer"] == "yes"
                else 0.5 if c["answer"] == "unclear"
                else 0.0
                for c in normalised
            )
            passed = sum(1 for c in normalised if c["answer"] == "yes")
            if total > 0:
                # Floor at 1, cap at 10. round() to integer.
                score = max(1, min(10, round((score_units / total) * 10)))
            else:
                score = None
            # P3.5: if category_match is False, hard-cap the score at 3.
            # The structural checks below the silhouette one cannot redeem
            # a fundamentally-wrong-shape model.
            if cat_match is False and score is not None:
                if score > 3:
                    log.warning(
                        f"Judge clamped {score} -> 3 because category_match=False "
                        f"(requested={requested_object!r}, "
                        f"rendered={rendered_silhouette!r})"
                    )
                    score = 3
            issues = [
                f"{c['q']} → NO ({c['evidence']})"
                for c in normalised if c["answer"] == "no"
            ]
            unclear = [
                f"{c['q']} → UNCLEAR ({c['evidence']})"
                for c in normalised if c["answer"] == "unclear"
            ]
            issues.extend(unclear)
            # P3.5: identifiable now requires both passed >= half AND not
            # category-mismatched. A pyramid-when-asked-for-chessboard
            # cannot be "identifiable" no matter how many secondary checks
            # the VLM was forced to pass.
            identifiable_q = (passed >= max(1, total // 2)) if total > 0 else False
            identifiable = identifiable_q and (cat_match is not False)
            return cls(
                identifiable=identifiable,
                category=category,
                match_score=score,
                geometry_issues=issues,
                fix_suggestion=fix_suggestion,
                raw_response=text,
                checks=normalised,
                passed=passed,
                total=total,
                category_match=cat_match,
                requested_object=str(requested_object) if requested_object else None,
                rendered_silhouette=str(rendered_silhouette) if rendered_silhouette else None,
            )

        # Legacy path — VLM didn't follow Q&A protocol. Treat as before.
        raw_score = data.get("match_score", 5)
        try:
            score = int(raw_score)
        except (TypeError, ValueError):
            score = None
        issues = list(data.get("geometry_issues", []))

        clamped_from: int | None = None
        if score is not None and issues:
            ceiling = 5
            joined = " ".join(issues).lower()
            if any(kw in joined for kw in _SEVERE_ISSUE_KEYWORDS):
                ceiling = 4
            if score > ceiling:
                clamped_from = score
                log.warning(
                    f"Judge clamped {score} -> {ceiling} "
                    f"(issues={len(issues)}, ceiling={ceiling})"
                )
                score = ceiling

        return cls(
            identifiable=bool(data.get("identifiable", False)),
            category=category,
            match_score=score,
            geometry_issues=issues,
            fix_suggestion=fix_suggestion,
            raw_response=text,
            clamped_from=clamped_from,
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
    score_s = f"{judge.match_score}/10" if judge.match_score is not None else "unscored"

    # P5 (2026-04-28): structured Q&A retry. List passed checks (so the LLM
    # knows what to PRESERVE) and failed checks (so it knows what to FIX).
    if judge.checks:
        passed_lines = [
            f"  ✓ {c['q']}" for c in judge.checks if c["answer"] == "yes"
        ]
        failed_lines = [
            f"  ✗ {c['q']}\n      → reviewer says: {c['evidence']}"
            for c in judge.checks if c["answer"] == "no"
        ]
        unclear_lines = [
            f"  ? {c['q']}\n      → reviewer says: {c['evidence']}"
            for c in judge.checks if c["answer"] == "unclear"
        ]
        passed_block = "\n".join(passed_lines) or "  (none)"
        failed_block = "\n".join(failed_lines) or "  (none)"
        unclear_block = "\n".join(unclear_lines) or "  (none)"

        # P3.7 (2026-04-29): split retry framing by whether the silhouette
        # ITSELF is wrong vs only secondary structural checks failed.
        # When category_match is False, telling the LLM "the model looks
        # like X, you wanted Y" causes it to scrap the design wholesale —
        # which is correct. When category_match is True (silhouette OK,
        # only details wrong), the previous framing was misleading: it
        # reported `category` (which falls back to `rendered_silhouette`)
        # as "judged as 'a pyramid'", and the LLM would strip out the
        # very features that made it match — regressing the design. Now
        # we tell it explicitly: silhouette is OK, just refine the
        # failed checks.
        if judge.category_match is False:
            req = judge.requested_object or description
            rendered = judge.rendered_silhouette or judge.category
            header = (
                f"The previous code produced a 3D model whose overall shape "
                f"was interpreted as {rendered!r}, NOT a {req!r} as requested. "
                f"This is a fundamental form mismatch — re-design the model "
                f"from scratch with the CORRECT overall {req} silhouette. "
                f"(score {score_s})"
            )
        else:
            header = (
                f"The previous code's overall silhouette MATCHES the requested "
                f"{description!r} — DO NOT redesign it from scratch. The score "
                f"({score_s}) is held back by specific failed checks below; "
                f"REFINE the code to fix only those, while keeping every "
                f"passed feature intact."
            )

        return (
            f"{header}\n\n"
            f"PASSED checks (preserve these — your code already gets them right):\n"
            f"{passed_block}\n\n"
            f"FAILED checks (fix these — these are the reasons the score is low):\n"
            f"{failed_block}\n\n"
            f"UNCLEAR checks (make sure these read clearly from all 4 views):\n"
            f"{unclear_block}\n\n"
            f"Suggested operation: {judge.fix_suggestion}\n\n"
            f"Regenerate code for \"{description}\" that satisfies the FAILED "
            f"checks while preserving the PASSED ones. Follow the PLAN-THEN-CODE "
            f"protocol from the system prompt. Output ONLY the complete code."
        )

    # Legacy path
    issues = "\n".join(f"- {i}" for i in judge.geometry_issues) or "- (none listed)"
    return (
        f"The previous code produced a 3D model that was judged as: "
        f"{judge.category!r} (score {score_s}).\n"
        f"Issues the visual reviewer identified:\n{issues}\n"
        f"Suggested fix: {judge.fix_suggestion}\n\n"
        f"Please regenerate code that produces a model clearly recognisable as "
        f"\"{description}\". Output ONLY the complete corrected Python code."
    )
