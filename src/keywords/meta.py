# src/keywords/meta.py
from __future__ import annotations
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Set

ROOT = Path(__file__).resolve().parents[2]
SECTIONS_INDEX = ROOT / "data" / "sections_index.json"

# Strict patterns (noise-free)
ACT_PAT  = re.compile(r"^[A-Z][A-Za-z&()' .\-]+ Act, \d{4}$")
SECT_PAT = re.compile(r"^(Section|Article|Rule)\s+[0-9A-Z]+(?:\([0-9A-Za-z]+\))*$")

def _load_known_acts() -> Set[str]:
    acts: Set[str] = set()
    try:
        rows = json.loads(SECTIONS_INDEX.read_text("utf-8"))
        for r in rows:
            a = (r.get("act") or "").strip()
            if a and ACT_PAT.match(a):
                acts.add(a)
    except Exception:
        pass
    return acts

KNOWN_ACTS = _load_known_acts()

def build_meta_filters(extracted_terms: List[Dict[str, Any]], **_: Any) -> Dict[str, List[str]]:
    """
    Build high-precision meta filters from extracted keyword candidates.
    Only allow:
      - Acts that look like 'Xxx Xxx Act, 1999' AND (if available) appear in our dataset.
      - Sections/Articles/Rules that match strict patterns 
      like 'Section 52', 'Article 21', 'Rule 11C', 'Section 2(1)(j)'.

    Any extra kwargs (like prompt=...) are ignored safely.
    """
    acts: List[str] = []
    sections: List[str] = []

    for t in extracted_terms or []:
        term = (t.get("term") or "").strip()
        if not term:
            continue
        if SECT_PAT.match(term):
            sections.append(term)
            continue
        if ACT_PAT.match(term) and ((not KNOWN_ACTS) or (term in KNOWN_ACTS)):
            acts.append(term)

    # De‑dup and trim
    acts = sorted(set(acts))[:5]
    sections = sorted(set(sections))[:8]

    return {"acts": acts, "sections": sections, "jurisdictions": []}
