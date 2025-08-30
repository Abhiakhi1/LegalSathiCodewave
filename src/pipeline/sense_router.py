# src/pipeline/sense_router.py
from __future__ import annotations
import json, re
from typing import Dict, List, Tuple
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUES_PATH = ROOT / "data" / "cues.json"
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")

def _tok(text: str) -> List[str]:
    return [t.lower() for t in TOKEN.findall(text)]

def _load_cues() -> Dict[str, Dict[str, List[str]]]:
    if not CUES_PATH.exists():
        return {}
    return json.loads(CUES_PATH.read_text("utf-8"))

CUES = _load_cues()

def _score_vertical(qt: List[str], vertical: str, cues: Dict[str, Dict[str, List[str]]]) -> float:
    d = cues.get(vertical, {})
    pos = set(d.get("pos", []))
    pos_ctx = set(d.get("pos_context", []))
    neg = set(d.get("neg", []))

    qs = set(qt)
    score = 0.0
    score += 1.6 * len(qs & pos)
    # bonus for bigram presence inside query text (join tokens to string for contains)
    qstr = " ".join(qt)
    score += 1.1 * sum(1 for b in pos_ctx if b in qstr)
    score -= 1.2 * len(qs & neg)
    return score

def sense_enrich(prompt: str) -> Tuple[List[str], Dict[str, float]]:
    """
    Generic, dataset-driven sense disambiguation.

    Returns:
      expansions: extra terms to add to extractor/BM25
      category_boosts: {"CATEGORY": 1.1, ...}
    """
    if not CUES:
        return [], {}

    qt = _tok(prompt)
    vertical_scores = {v: _score_vertical(qt, v, CUES) for v in CUES.keys()}
    # take top-3 positives
    top = sorted(vertical_scores.items(), key=lambda x: x[1], reverse=True)[:3]

    expansions: List[str] = []
    boosts: Dict[str, float] = {}

    for v, s in top:
        if s <= 0:
            continue
        d = CUES.get(v, {})
        expansions += d.get("expansion", []) + d.get("pos", [])[:6]  # add a few pos terms too
        # small capped boost
        boosts[v] = 1.0 + min(0.25, s * 0.05)

    # deduplicate expansions
    seen = set(); uniq = []
    for e in expansions:
        el = e.strip().lower()
        if el and el not in seen:
            uniq.append(el); seen.add(el)

    return uniq[:30], boosts
