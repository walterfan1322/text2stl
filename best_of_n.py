"""Best-of-N sampling helper (S7.1).

For shapes that historically score erratically (figurine, bottle,
teapot), it's cheaper than retrying serially to ask the LLM N times in
parallel at varying temperatures, run all candidates through the
geometric / VLM judges, and pick the highest-scoring one.

Pure helper — no app.py state. The caller wires it into the generation
loop.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

log = logging.getLogger("text2stl.best_of_n")


@dataclass
class Candidate:
    idx: int
    temperature: float
    code: Optional[str] = None
    stl_path: Optional[str] = None
    exec_ok: bool = False
    judge_score: Optional[int] = None
    geom_passed: bool = True
    watertight: bool = False
    error: str = ""
    elapsed_s: float = 0.0


# Default temperature spread when N candidates requested.
DEFAULT_TEMPS = {
    1: [0.5],
    2: [0.3, 0.8],
    3: [0.3, 0.6, 0.9],
    4: [0.2, 0.5, 0.8, 1.0],
}


def temps_for(n: int) -> list[float]:
    if n in DEFAULT_TEMPS:
        return DEFAULT_TEMPS[n]
    # Spread evenly between 0.3 and 1.0
    if n <= 1:
        return [0.5]
    step = (1.0 - 0.3) / (n - 1)
    return [round(0.3 + step * i, 2) for i in range(n)]


def score_candidate(c: Candidate) -> tuple[int, int, int, float]:
    """Sort key — higher is better.

    Tiebreak order:
      1. exec_ok (yes > no)
      2. geom_passed (yes > no)
      3. watertight (yes > no)
      4. judge_score (defaulting to 0)
      5. faster generation (negate elapsed_s as last tiebreak)
    """
    return (
        1 if c.exec_ok else 0,
        1 if c.geom_passed else 0,
        1 if c.watertight else 0,
        int(c.judge_score or 0),
    )


def pick_best(candidates: list[Candidate]) -> Candidate:
    if not candidates:
        raise ValueError("no candidates")
    return max(candidates,
               key=lambda c: score_candidate(c) + (-c.elapsed_s,))


async def run_best_of_n(
    n: int,
    runner: Callable[[float, int], Awaitable[Candidate]],
) -> tuple[Candidate, list[Candidate]]:
    """Run N candidate runners concurrently, return (best, all).

    `runner(temperature, idx) -> Candidate` is supplied by the caller —
    it's the per-candidate generation flow (LLM call → exec → repair →
    judge).
    """
    n = max(1, n)
    temps = temps_for(n)
    coros = [runner(t, i) for i, t in enumerate(temps)]
    candidates = await asyncio.gather(*coros, return_exceptions=False)
    best = pick_best(list(candidates))
    log.info(f"best_of_{n}: picked idx={best.idx} t={best.temperature} "
             f"score={best.judge_score} exec={best.exec_ok}")
    return best, list(candidates)


# Default per-category N — start conservative, only multiply tokens
# for shapes we know are unstable.
DEFAULT_BEST_OF_PER_CATEGORY = {
    "figurine": 3,
    "bottle":   2,
    "teapot":   2,
    "shoe":     2,
    # all others default to 1 (no fan-out)
}


def n_for_category(category: str, override: dict | None = None) -> int:
    table = override or DEFAULT_BEST_OF_PER_CATEGORY
    return int(table.get(category, 1))
