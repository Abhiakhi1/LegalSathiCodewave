# src/pipeline/normalizer.py
from __future__ import annotations
import re
from typing import List, Tuple

# Minimal Hinglish → English helpers (extend as you see patterns)
HINGLISH_MAP = {
    "kya": "what", "kaise": "how", "kab": "when", "kahan": "where",
    "kis section": "which section", "konsa section": "which section",
    "complaint kaise": "how to file complaint",
    "padosi": "neighbor", "zamin": "land", "khet": "field",
    "encrochment": "encroachment", "encroched": "encroached",
    "tehasildar": "tehsildar", "tahsildar": "tehsildar",
    "fir": "first information report",
}

# Light legal synonyms / aliases (feed the retriever richer terms)
SYNONYMS = {
    "encroachment": ["property boundary", "illegal occupation", "land dispute"],
    "murder": ["homicide", "section 302", "punishment for murder"],
    "rape": ["sexual offence", "section 376"],
    "dowry": ["dowry harassment", "dowry prohibition act 1961"],
    "tehsildar": ["revenue officer"],
}

# Optional stop-ish fillers to trim from expansions (we do NOT touch original prompt)
FILLERS = {"please", "plz", "kindly", "sir", "maam", "ji", "bhai", "bro"}

_WORD = re.compile(r"[a-zA-Z]+(?:'[a-z]+)?")

def normalize_prompt_for_semantics(text: str) -> Tuple[str, List[str]]:
    """
    Returns: (clean_text_for_semantic_models, expansions[])
    - We keep the original prompt for keyword regexes (Acts/Sections need case),
      but semantic models can benefit from normalized text + extra terms.
    """
    # lowercase copy for normalization (keep original elsewhere)
    low = text.lower()

    # inject hinglish replacements
    for k, v in HINGLISH_MAP.items():
        low = low.replace(k, v)

    # simple de-noising
    low = re.sub(r"\s+", " ", low).strip()

    # tokens for expansions (drop fillers)
    toks = [t for t in _WORD.findall(low) if t not in FILLERS]
    expansions: List[str] = []

    # heuristic: if key terms present, add synonyms
    low_space = " " + low + " "
    for key, syns in SYNONYMS.items():
        if f" {key} " in low_space:
            expansions.extend(syns)

    # dedup expansions while preserving order
    seen, uniq = set(), []
    for e in expansions:
        if e not in seen:
            uniq.append(e); seen.add(e)

    return low, uniq
