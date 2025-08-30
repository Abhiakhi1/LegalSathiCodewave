# src/ingestion/build_cues.py
from __future__ import annotations
import json, re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "Dataset"
SECTIONS_INDEX = ROOT / "data" / "sections_index.json"
OUT = ROOT / "data" / "cues.json"

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")
SPLIT_SENT = re.compile(r"[.!?]\s+")

# Light stoplist
STOP = {
    "the","a","an","of","and","or","in","on","at","to","for","by","with","as","is","are","be","was","were",
    "from","this","that","these","those","it","its","into","within","their","there","such","under","over",
    "not","no","any","all","may","can","could","shall","should","will","would","etc","eg","ie",
    "act","section","article","rule","sub","clause","also"
}

def _read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return p.read_text(errors="ignore")

def _tok(text: str) -> List[str]:
    return [t.lower() for t in TOKEN.findall(text)]

def _ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _top_tfidf(per_cat_counts: Dict[str, Counter], df: Counter, total_docs: int, k: int) -> Dict[str, List[Tuple[str,float]]]:
    out = {}
    for cat, cnt in per_cat_counts.items():
        scores = {}
        for term, f in cnt.items():
            # docfreq ~ in how many categories term appears
            d = df.get(term, 1)
            idf = max(0.0, __import__("math").log((1+total_docs)/d))
            scores[term] = (f ** 0.5) * idf
        # pick top-k
        out[cat] = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return out

def _load_acts_sections() -> Dict[str, List[str]]:
    """Return per-category expansions mined from sections_index.json (Acts names / frequent sections)."""
    m: Dict[str, List[str]] = defaultdict(list)
    if not SECTIONS_INDEX.exists(): 
        return m
    rows = json.loads(SECTIONS_INDEX.read_text("utf-8"))
    for r in rows:
        cat = r.get("category") or r.get("file") or ""
        cat = cat.replace(".txt","").upper()
        act = (r.get("act") or "").strip()
        sec = (r.get("section_display") or "").strip()
        if act and act not in m[cat]:
            m[cat].append(act)
        if sec and sec not in m[cat]:
            m[cat].append(sec)
    # trim
    for k in list(m.keys()):
        m[k] = m[k][:20]
    return m

def main():
    assert DATASET_DIR.exists(), f"Dataset folder not found at {DATASET_DIR}"
    files = sorted([p for p in DATASET_DIR.glob("*.txt") if p.is_file()])
    if not files:
        print("[CUES] No .txt files in Dataset/")
        return

    print(f"[CUES] Scanning {len(files)} files to learn cues…")

    # Per-category unigrams/bigrams
    per_uni: Dict[str, Counter] = {}
    per_bi: Dict[str, Counter] = {}
    df_uni, df_bi = Counter(), Counter()

    for p in files:
        cat = p.stem.upper()
        txt = _read_text(p).lower()
        toks = [t for t in _tok(txt) if t not in STOP and not t.isdigit()]
        unis = Counter(toks)
        bis = Counter(_ngrams(toks, 2))

        per_uni[cat] = unis
        per_bi[cat] = bis

        for term in set(unis):
            df_uni[term] += 1
        for term in set(bis):
            df_bi[term] += 1

    total = len(per_uni)

    # Distinctive unigrams/bigrams by TF-IDF
    top_uni = _top_tfidf(per_uni, df_uni, total, k=120)  # generous; we'll filter later
    top_bi  = _top_tfidf(per_bi,  df_bi,  total, k=60)

    # Build positives (pos), pos_context (bigrams/top), negatives (strong terms from other cats), expansions (acts/sections)
    acts_map = _load_acts_sections()
    cues = {}

    # Global negatives that often cause wrong routing
    GLOBAL_NEG = {
        "online","internet","email","server","hacking","phishing","malware","account","otp","password","data","breach",
        "neighbour","neighbor","house","land","boundary","property","village","market","street","company","corporate","insolvency"
    }

    for cat in per_uni.keys():
        # base pools
        uni_terms = [t for t, s in top_uni.get(cat, []) if len(t) > 2 and t not in STOP][:80]
        bi_terms  = [t for t, s in top_bi.get(cat, [])  if all(w not in STOP for w in t.split())][:40]

        # choose positives: keep legal-ish words + common category words
        pos = []
        for term in uni_terms:
            # prefer words that aren't pure boilerplate
            if term.isalpha() and len(term) >= 4:
                pos.append(term)
        # Add a few high-signal bigrams as context/positives
        pos_context = []
        for term in bi_terms:
            if len(term) >= 7 and not any(w.isdigit() for w in term.split()):
                pos_context.append(term)

        # Build negatives: take top strong terms from *other* categories
        negatives = set()
        for other, items in top_uni.items():
            if other == cat:
                continue
            for t, _ in items[:30]:
                if t not in pos and len(t) >= 5:
                    negatives.add(t)
        negatives |= GLOBAL_NEG
        negatives = list(sorted(negatives))[:60]

        # Expansions: Acts/Sections mined from index
        expansions = acts_map.get(cat, [])[:20]

        # Trim lists
        pos = pos[:60]
        pos_context = pos_context[:30]

        cues[cat] = {
            "pos": pos,
            "pos_context": pos_context,
            "neg": negatives,
            "expansion": expansions
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cues, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[CUES] Wrote {OUT} (learned cues for {len(cues)} categories).")

if __name__ == "__main__":
    main()
