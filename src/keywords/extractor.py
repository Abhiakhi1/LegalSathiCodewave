# src/keywords/extractor.py
from __future__ import annotations
import re, json, math
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
from src.pipeline.section_ranker import SectionRanker

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

def _load_corpus_stats():
    with open(DATA_DIR / "corpus_stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    N = int(stats.get("N_docs", 1))
    df = Counter({str(k).lower(): int(v) for k, v in stats.get("df", {}).items()})
    return N, df

N_DOCS, DF = _load_corpus_stats()

ACT_RE  = re.compile(r'\b([A-Z][A-Za-z]*(?: [A-Za-z&\-]+)* Act,? \d{4})\b')
SECT_RE = re.compile(r'(?i)\b(?:Section|Sec\.)\s+(\d+[A-Z\-]*)\b')
JUR_RE  = re.compile(r'(?i)\b(India|Delhi|Maharashtra|Uttar Pradesh|Karnataka|Tamil Nadu|Kerala|Gujarat|Pune|Mumbai|Bengaluru|Chennai|Hyderabad)\b')

STOP_PHRASES = [
    "what to do","how to do","please help","help me","what should i do",
    "kya karu","kya karoon","kya karna chahiye","suggest me","give advice"
]
STOP_WORDS = set("""
a an the and or of for to in on by with is are be been being this that those these
very really please kindly help advice about tell explain give suggest regarding under
over against if then it i me my mine we our you your their someone anyone anything
what which who whose when where how do does did doing done
""".split())

WORD = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
PHRASE = re.compile(r"(?i)\b([a-z][a-z\- ]{3,})\b")

def _idf(t: str) -> float:
    return math.log((N_DOCS + 1) / (DF.get(t.lower(), 1) + 0.5))

def _clean_text(t: str) -> str:
    low = t.lower()
    for p in STOP_PHRASES:
        low = low.replace(p, " ")
    return re.sub(r"\s+", " ", low).strip()

def _candidate_terms(text: str) -> List[str]:
    text = _clean_text(text)
    c: List[str] = []
    c += ACT_RE.findall(text)
    c += [f"Section {s}" for s in SECT_RE.findall(text)]
    c += JUR_RE.findall(text)
    # short phrases (≤4 tokens), drop stop-words
    for m in PHRASE.finditer(text):
        ph = re.sub(r"\s+", " ", m.group(1).strip())
        if not ph or len(ph.split()) > 4:
            continue
        if ph.lower() in STOP_WORDS:
            continue
        c.append(ph)
    # dedup
    seen, out = set(), []
    for t in c:
        k = t.lower()
        if k not in seen:
            seen.add(k); out.append(t)
    return out

def _specificity(term: str) -> float:
    s = 0.0
    if ACT_RE.search(term) or SECT_RE.search(term): s += 0.4
    s += 0.6 * _idf(term)
    return s

def _top_idf_terms(text: str, limit: int = 3) -> List[str]:
    text = _clean_text(text)
    terms = [m.group(0) for m in WORD.finditer(text) if m.group(0).lower() not in STOP_WORDS]
    uniq = list({t.lower(): t for t in terms}.values())
    scored = sorted(((t, _idf(t)) for t in uniq), key=lambda x: x[1], reverse=True)
    return [t for t,_ in scored[:limit]]

def extract_keywords(prompt: str, k: int = 12, tau: float = 0.65, expansions: List[str] | None = None) -> List[Dict[str, Any]]:
    cands = _candidate_terms(prompt)
    scored = []
    for t in cands:
        spec = _specificity(t)
        if spec >= tau:
            scored.append({"term": t, "specificity": round(spec, 3), "reason": f"idf={round(_idf(t),3)}"})
    if len(scored) < 3:
        for t in _top_idf_terms(prompt, limit=max(3, min(5, k//2 or 3))):
            if all(t.lower() != s["term"].lower() for s in scored):
                scored.append({"term": t, "specificity": round(_idf(t), 3), "reason": "fallback_idf"})
    scored.sort(key=lambda x: x["specificity"], reverse=True)
    return scored[:k]

def build_meta_filters(keywords: List[Dict[str, Any]], prompt: str | None = None) -> Dict[str, List[str]]:
    acts, secs, jurs = [], [], []
    for kw in keywords:
        t = kw["term"]
        if ACT_RE.search(t): acts.append(t)
        elif SECT_RE.search(t): secs.append(t)
        elif JUR_RE.search(t): jurs.append(t)

    # Data‑driven suggestions from SectionRanker if user didn’t name a section
    if not secs and prompt:
        suggestions = SectionRanker.suggest(prompt, top_k=3)
        for s in suggestions:
            if s["section"] and s["section"] not in secs:
                secs.append(s["section"])
            if s["act"] and s["act"] not in acts:
                acts.append(s["act"])

    # dedup preserve order
    def dedup(xs):
        seen=set(); out=[]
        for v in xs:
            if v and v not in seen: seen.add(v); out.append(v)
        return out

    return {"acts": dedup(acts), "sections": dedup(secs), "jurisdictions": dedup(jurs)}
