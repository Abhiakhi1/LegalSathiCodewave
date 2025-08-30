# src/search/retriever.py
from __future__ import annotations

import json
import os
import sys
import math
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple


# Ensure "src" is importable when uvicorn starts from project root
_THIS_DIR = os.path.dirname(__file__)
_SYS_SRC = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _SYS_SRC not in sys.path:
    sys.path.append(_SYS_SRC)

# ----------------------- Paths & constants -----------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_DATADIR = os.path.join(_PROJECT_ROOT, "data")
_VECTORDIR = os.path.join(_PROJECT_ROOT, "vectordb")

_COLLECTION_MAP = os.path.join(_DATADIR, "collection_map.json")

# Use the same model you used at ingestion time
_EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Controls for automatic selection / pooling
_MAX_TOP_CATS_CAP = 6            # hard upper guardrail
_MIN_PER_CAT_K = 3               # don’t query less than this per category
_MAX_PER_CAT_K = 10              # avoid huge fan-out per category
_DEFAULT_GLOBAL_K_CAP = 8        # UI sends a max; we may stop earlier on novelty
_BASE_NOVELTY_THRESH = 0.035     # stop adding hits when marginal coverage gain < this
_MIN_RESULTS_BEFORE_NOVELTY = 3  # always allow at least this many before checking novelty


# ---------------------------- Utilities --------------------------------------
def _safe_read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[a-zA-Z0-9_]+", str(text).lower())


# -------- Embeddings ---------------------------------------------------------
@lru_cache(maxsize=1)
def _get_embedder():
    # Lazy import so the app starts even if torch has not warmed yet
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_EMBED_MODEL)


def _embed(texts: List[str]) -> List[List[float]]:
    model = _get_embedder()
    return model.encode(texts, normalize_embeddings=True).tolist()


# -------- Chroma client / collections ----------------------------------------
@lru_cache(maxsize=1)
def _load_chroma_client():
    import chromadb
    # IMPORTANT: single client per-process to avoid "different settings" error
    return chromadb.PersistentClient(path=_VECTORDIR)


@lru_cache(maxsize=1)
def _load_collection_map() -> Dict[str, str]:
    """
    label -> collection_name (with hash suffix).
    """
    return _safe_read_json(_COLLECTION_MAP, {})


def _get_collection(label: str):
    """
    Resolve label to collection and return a live Collection handle.
    """
    client = _load_chroma_client()
    cmap = _load_collection_map()
    name = cmap.get(label, label)  # try exact label first, otherwise raw
    try:
        return client.get_collection(name=name)
    except Exception:
        # As a last resort, scan all collections for a suffix match
        for c in client.list_collections():
            if c.name == name or c.name.endswith(name):
                return client.get_collection(name=c.name)
        raise


# -------------------------- Keyword helpers ----------------------------------
def _extract_keywords(prompt: str) -> List[Dict[str, Any]]:
    """
    Use your project’s extractor if available, otherwise a safe fallback.
    """
    try:
        from src.keywords import extract_keywords  # type: ignore
        kws = extract_keywords(prompt)
        out = []
        for k in kws:
            if isinstance(k, dict):
                term = k.get("term") or k.get("token") or k.get("text")
                spec = float(k.get("specificity", 1.0))
                reason = k.get("reason", "kw")
            else:
                term = str(k)
                spec = 1.0
                reason = "kw"
            if term:
                out.append({"term": term, "specificity": spec, "reason": reason})
        if out:
            return out
    except Exception:
        pass
    # fallback
    terms = [t.strip() for t in prompt.split() if t.strip()]
    return [{"term": t, "specificity": 1.0, "reason": "fallback_idf"} for t in terms]


def _build_meta_filters(keywords: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    try:
        from src.keywords import build_meta_filters  # type: ignore
        m = build_meta_filters(keywords)
        if isinstance(m, dict):
            return {
                "acts": list(m.get("acts", []) or []),
                "sections": list(m.get("sections", []) or []),
                "jurisdictions": list(m.get("jurisdictions", []) or []),
            }
    except Exception:
        pass
    return {"acts": [], "sections": [], "jurisdictions": []}


# ------------------------------ Core search ----------------------------------
def _query_collection(label: str, q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    """
    Query a single category (collection) and return normalized hits.
    """
    coll = _get_collection(label)
    result = coll.query(
        query_embeddings=[q_emb],
        n_results=max(1, k),
        include=["metadatas", "documents", "distances"],
    )
    hits: List[Dict[str, Any]] = []
    ids = result.get("ids", [[]])[0] if isinstance(result.get("ids"), list) else []
    docs = result.get("documents", [[]])[0] if isinstance(result.get("documents"), list) else []
    dists = result.get("distances", [[]])[0] if isinstance(result.get("distances"), list) else []
    metas = result.get("metadatas", [[]])[0] if isinstance(result.get("metadatas"), list) else []
    for i in range(len(ids)):
        md = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        sections = md.get("sections") or md.get("section") or md.get("Sections") or ""
        acts = md.get("acts") or md.get("Acts") or ""
        juris = md.get("jurisdiction") or md.get("jurisdictions") or ""
        # chroma returns smaller distance for closer match; turn into similarity
        sim = 1.0 - float(dists[i]) if i < len(dists) else None
        # build a small feature set for novelty/coverage
        feat = set(_tokenize(sections)) | set(_tokenize(acts)) | set(_tokenize(juris))
        feat |= set(_tokenize(ids[i])) | set(_tokenize(label))
        # optionally light-touch from document head
        if i < len(docs) and docs[i]:
            head = str(docs[i])[:400]  # small slice to avoid heavy memory
            feat |= set(_tokenize(head))
        hits.append(
            {
                "category": label,
                "id": ids[i],
                "sim": sim,
                "sections": sections,
                "acts": acts,
                "jurisdiction": juris,
                "doc": docs[i] if i < len(docs) else None,
                "metadata": md,
                "_features": list(feat),
            }
        )
    return hits


def _choose_top_n_categories(cat_scores: List[Tuple[str, float]], cap: int) -> int:
    """
    Decide how many top categories to keep from (label, score) sorted desc.
    Heuristic: count early sharp drops; ensure at least 1; broaden when the
    head is flat or low-confidence.
    """
    if not cat_scores:
        return 1
    cap = max(1, min(int(cap), _MAX_TOP_CATS_CAP))
    sims = [s for _, s in cat_scores]
    top = sims[0]
    if top <= 0:
        return 1
    # normalize by top score
    nrm = [s / top for s in sims]
    deltas = [nrm[i] - nrm[i + 1] for i in range(min(len(nrm) - 1, 6))]
    # count meaningful gaps
    gap_thresh = 0.12  # sharpness needed to say "next topic drops"
    count = sum(1 for g in deltas if g > gap_thresh)
    n = max(1, min(1 + count, cap))
    # if head is very flat or confidence low, widen to at least 2 (when possible)
    if n < 2 and (nrm[0] < 0.55 or (deltas and max(deltas) < 0.08)) and cap >= 2:
        n = 2
    return n


def _pool_with_novelty(
    hits_by_cat: Dict[str, List[Dict[str, Any]]],
    global_cap: int,
    novelty_threshold: float,
    min_before_check: int = _MIN_RESULTS_BEFORE_NOVELTY,
) -> Tuple[List[Dict[str, Any]], float, str]:
    """
    Global pool across categories (by sim), but stop early when new hits stop
    adding coverage (novelty).
    Returns (pooled_hits, last_gain, stop_reason)
    """
    pool: List[Dict[str, Any]] = []
    for arr in hits_by_cat.values():
        pool.extend(arr)
    # sort by sim descending; None sims go last
    pool.sort(key=lambda h: (h.get("sim") is not None, h.get("sim", -1.0)), reverse=True)

    used: List[Dict[str, Any]] = []
    seen_features: set = set()
    last_gain: float = 0.0
    stop_reason: str = "cap_reached"

    def novelty_gain(h) -> float:
        feats = set(h.get("_features") or [])
        if not feats:
            return 0.0
        new = feats - seen_features
        union = feats | seen_features
        return len(new) / max(1.0, float(len(union)))

    for h in pool:
        if len(used) >= global_cap:
            stop_reason = "cap_reached"
            break
        gain = novelty_gain(h)
        h["novelty_gain"] = round(gain, 4)
        # accept freely for the first few; then require novelty
        if len(used) < min_before_check or gain >= novelty_threshold:
            used.append(h)
            seen_features |= set(h.get("_features") or [])
            last_gain = gain
        else:
            # low novelty; skip
            continue

    # if we stopped because everything left was low-novelty
    if len(used) < global_cap:
        stop_reason = "novelty_plateau"

    # strip internal fields
    for h in used:
        h.pop("_features", None)

    return used, last_gain, stop_reason


def search_top_categories(
    query: str,
    top_n_cats: int = 3,
    per_cat_k: int = 6,
    global_k: int = _DEFAULT_GLOBAL_K_CAP,
    meta_filters: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    End-to-end retrieval:
      1) keywords (+ optional meta filters)
      2) score all categories quickly (k=1)
      3) AUTO: choose how many top categories to keep
      4) AUTO: choose per-cat k based on target global cap
      5) within each category fetch per_cat_k hits from Chroma
      6) globally pool with novelty/coverage stop (AUTO global_k_used)
    """
    # 1. keywords + meta
    keywords = _extract_keywords(query)
    mfilters = _build_meta_filters(keywords)
    if meta_filters:
        for k, v in (meta_filters or {}).items():
            if v:
                mfilters.setdefault(k, [])
                for item in v:
                    if item not in mfilters[k]:
                        mfilters[k].append(item)

    # 2. one embed
    q_emb = _embed([query])[0]

    # 2b. score categories (k=1)
    client = _load_chroma_client()
    colls = client.list_collections()
    cat_scores: List[Tuple[str, float]] = []
    for c in colls:
        try:
            r = c.query(query_embeddings=[q_emb], n_results=1, include=["distances"])
            dist = r.get("distances", [[1.0]])[0][0]
            sim = 1.0 - float(dist)
            cat_scores.append((c.name, sim))
        except Exception:
            continue

    # map collection name back to label if we have a mapping
    cmap = _load_collection_map()
    inv = {v: k for k, v in cmap.items()}  # collection_name -> label
    cat_scores = [(inv.get(name, name), sim) for (name, sim) in cat_scores]
    cat_scores.sort(key=lambda x: x[1], reverse=True)

    # 3. AUTO: choose top-N categories to keep (respect the UI cap if provided)
    cap_top = max(1, int(top_n_cats))
    chosen_n = _choose_top_n_categories(cat_scores, cap=cap_top)

    top_labels = [name for (name, _s) in cat_scores[: max(1, chosen_n)]]

    db_ranking = [
        {"category": lab, "score": float(score), "collection": cmap.get(lab, lab)}
        for (lab, score) in cat_scores[: max(1, chosen_n)]
    ]

    # 4. AUTO: choose per-cat k based on global cap & chosen_n
    # ensure enough depth to be able to fill the pool with a little buffer
    global_cap = max(1, int(global_k or _DEFAULT_GLOBAL_K_CAP))
    auto_per_cat = math.ceil(1.5 * global_cap / max(1, chosen_n)) + 1
    per_cat_k_used = int(max(_MIN_PER_CAT_K, min(_MAX_PER_CAT_K, auto_per_cat)))

    # 5. fetch hits
    hits_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for lab in top_labels:
        try:
            hits_by_cat[lab] = _query_collection(lab, q_emb, per_cat_k_used)
        except Exception:
            hits_by_cat[lab] = []

    # 6. global pool with novelty/coverage stopping
    # make novelty threshold slightly adaptive to query size
    adaptive_thresh = _BASE_NOVELTY_THRESH
    q_len = len(_tokenize(query))
    if q_len >= 14:
        adaptive_thresh = _BASE_NOVELTY_THRESH * 0.85
    pooled, last_gain, stop_reason = _pool_with_novelty(
        hits_by_cat, global_cap=global_cap, novelty_threshold=adaptive_thresh
    )

    trace = {
        "prompt": query,
        "keywords": keywords,
        "meta_filters": mfilters,
        "collections_scanned": len(cat_scores),
        "db_ranking_sample": db_ranking[:3],
        # decisions
        "auto_top_cats_used": chosen_n,
        "per_cat_k_used": per_cat_k_used,
        "global_cap": global_cap,
        "global_k_used": len(pooled),
        "novelty_threshold": adaptive_thresh,
        "coverage_gain_last": round(last_gain, 4),
        "stop_reason": stop_reason,
    }

    # clean result hits (keep backward compat fields)
    for h in pooled:
        # surface common fields under metadata too (for your UI)
        md = h.get("metadata") or {}
        if "sections" not in md and h.get("sections"):
            md["sections"] = h["sections"]
        if "acts" not in md and h.get("acts"):
            md["acts"] = h["acts"]
        if "jurisdictions" not in md and h.get("jurisdiction"):
            md["jurisdictions"] = h["jurisdiction"]
        h["metadata"] = md

    return {
        "prompt": query,
        "keywords": keywords,
        "meta_filters": mfilters,
        "db_ranking": db_ranking,
        "hits": pooled,
        "trace": trace,
    }
