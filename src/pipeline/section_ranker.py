# src/pipeline/section_ranker.py
from __future__ import annotations
import json, math, re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Allow digits & parentheses so tokens like 498A, 2(7), 11C are indexed
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\(\)\/\.]*")

STOP = set("""
a an the and or of for to in on by with is are be been being this that those these
very really please kindly help advice about tell explain give suggest regarding under
over against if then it i me my mine we our you your their someone anyone anything
what which who whose when where how do does did doing done act acts chapter chapters
section sections sec s article articles art rule rules r
""".split())

def _tok(s: str) -> List[str]:
    toks = [w.lower() for w in WORD.findall(s.lower())]
    return [t for t in toks if t not in STOP]


class SectionRanker:
    _rows: List[Dict[str, Any]] = []
    _Vt: np.ndarray | None = None  # title embeddings
    _Vl: np.ndarray | None = None  # lead embeddings
    _model: SentenceTransformer | None = None

    # BM25 state
    _docs_tokens: List[List[str]] = []
    _df: Dict[str, int] = {}
    _avgdl: float = 1.0
    _N: int = 1

    # Act embeddings for hierarchical boost
    _acts: List[str] = []
    _act_emb: np.ndarray | None = None
    _row_act_idx: List[int] = []

    @classmethod
    def _ensure_loaded(cls):
        if not cls._rows:
            with open(DATA_DIR / "sections_index.json", "r", encoding="utf-8") as f:
                cls._rows = json.load(f)
        if cls._Vt is None:
            cls._Vt = np.load(DATA_DIR / "sections_emb_titles.npy")
        if cls._Vl is None:
            cls._Vl = np.load(DATA_DIR / "sections_emb_leads.npy")
        if cls._model is None:
            cls._model = SentenceTransformer(MODEL_NAME, device="cpu")

        # Build BM25 tokens (titles only, small + fast)
        if not cls._docs_tokens:
            titles = [r.get("display","") for r in cls._rows]
            cls._docs_tokens = [_tok(t) for t in titles]
            cls._N = len(cls._docs_tokens)
            lengths = [len(toks) or 1 for toks in cls._docs_tokens]
            cls._avgdl = float(sum(lengths)) / max(1, len(lengths))
            df: Dict[str, int] = {}
            for toks in cls._docs_tokens:
                for term in set(toks):
                    df[term] = df.get(term, 0) + 1
            cls._df = df

        # Unique acts + embeddings + row->act index
        if cls._act_emb is None:
            acts = []
            row_act_idx = []
            idx_map: Dict[str, int] = {}
            for r in cls._rows:
                a = (r.get("act") or "").strip()
                if a not in idx_map:
                    idx_map[a] = len(acts)
                    acts.append(a)
                row_act_idx.append(idx_map[a])
            cls._acts = acts
            cls._row_act_idx = row_act_idx
            cls._act_emb = cls._model.encode(acts, normalize_embeddings=True)

    # BM25 for a single query over titles
    @classmethod
    def _bm25_scores(cls, q_tokens: List[str]) -> np.ndarray:
        k1, b = 1.5, 0.75
        idf = {}
        for t in set(q_tokens):
            df = cls._df.get(t, 0)
            idf[t] = math.log((cls._N - df + 0.5) / (df + 0.5) + 1.0)
        scores = np.zeros((cls._N,), dtype=np.float32)
        for i, doc in enumerate(cls._docs_tokens):
            dl = len(doc) or 1
            tf = {}
            for t in doc:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in q_tokens:
                if t not in tf: 
                    continue
                denom = tf[t] + k1 * (1 - b + b * dl / cls._avgdl)
                s += idf.get(t, 0.0) * (tf[t] * (k1 + 1) / denom)
            scores[i] = s
        m = float(scores.max()) if cls._N else 1.0
        if m > 0:
            scores /= m  # normalize 0-1
        return scores

    @classmethod
    def suggest(cls, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        cls._ensure_loaded()
        qvec = cls._model.encode([query], normalize_embeddings=True)[0]

        # Dense
        cos_title = cls._Vt @ qvec              # title semantic match
        cos_lead  = cls._Vl @ qvec              # snippet/lead semantic match

        # Sparse
        bm25 = cls._bm25_scores(_tok(query))    # term match on titles

        # Hierarchical: Act boost (small)
        act_sim = (cls._act_emb @ qvec) if cls._act_emb is not None else None
        act_boost = np.zeros_like(cos_title)
        if act_sim is not None:
            # softplus around 0.6 threshold
            for i, aidx in enumerate(cls._row_act_idx):
                s = float(act_sim[aidx])
                act_boost[i] = max(0.0, s - 0.6)  # only reward strong act match

        # Fusion weights (tuneable)
        # embeddings already in [0,1]; bm25 normalized to [0,1]
        score = (
            0.55 * cos_title +
            0.25 * bm25 +
            0.15 * cos_lead +
            0.05 * act_boost
        )

        order = np.argsort(-score)[:top_k]
        out: List[Dict[str, Any]] = []
        for i in order:
            r = cls._rows[int(i)]
            out.append({
                "act": r.get("act",""),
                "section": r.get("section",""),
                "title": r.get("title",""),
                "display": r.get("display",""),
                "doc_id": r.get("doc_id",""),
                "score": float(score[int(i)]),
                "scores": {
                    "cos_title": float(cos_title[int(i)]),
                    "bm25": float(bm25[int(i)]),
                    "cos_lead": float(cos_lead[int(i)]),
                    "act_boost": float(act_boost[int(i)]),
                }
            })
        return out
