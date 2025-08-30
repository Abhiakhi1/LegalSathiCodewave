# src/router/ranker.py
import json, re, functools
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL = SentenceTransformer(MODEL_NAME, device="cpu")

def _load_labels() -> List[str]:
    with open(DATA_DIR / "labels.json", "r", encoding="utf-8") as f:
        return json.load(f)

def _load_collection_map() -> Dict[str, str]:
    p = DATA_DIR / "collection_map.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _clean_label(lbl: str) -> str:
    s = re.sub(r'[_\-]+', ' ', lbl)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _describe_label(lbl: str) -> str:
    base = _clean_label(lbl)
    return f"{base}. This category covers acts, sections, procedures, and remedies in Indian law."

def _embed(texts: List[str]) -> np.ndarray:
    vecs = _MODEL.encode(texts, normalize_embeddings=True)
    return np.array(vecs)

@functools.lru_cache(maxsize=1)
def _label_matrix() -> Dict[str, Any]:
    labels = _load_labels()
    descs  = [_describe_label(l) for l in labels]
    mat    = _embed(descs)               # (L, D) normalized
    return {"labels": labels, "V": mat}

def rank_categories(prompt: str, top_k: int = 5) -> List[Dict[str, Any]]:
    colmap = _load_collection_map()
    M = _label_matrix()
    labels, V = M["labels"], M["V"]      # (L,), (L,D)
    q = _embed([prompt])[0]               # (D,)
    sims = V @ q                          # cosine (normalized)
    order = np.argsort(-sims)[:top_k]
    return [
        {"category": labels[i], "sanitized_collection": colmap.get(labels[i], labels[i]), "score": float(sims[i])}
        for i in order
    ]
