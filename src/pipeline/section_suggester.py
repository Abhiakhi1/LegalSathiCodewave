# src/pipeline/section_suggester.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class SectionSuggester:
    _rows: List[Dict[str, Any]] = []
    _V: np.ndarray | None = None
    _model: SentenceTransformer | None = None

    @classmethod
    def _ensure_loaded(cls):
        if not cls._rows:
            with open(DATA_DIR / "sections_index.json", "r", encoding="utf-8") as f:
                cls._rows = json.load(f)
        if cls._V is None:
            cls._V = np.load(DATA_DIR / "sections_emb.npy")
        if cls._model is None:
            cls._model = SentenceTransformer(MODEL_NAME, device="cpu")

    @classmethod
    def suggest(cls, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        cls._ensure_loaded()
        q = cls._model.encode([query], normalize_embeddings=True)[0]
        sims = cls._V @ q  # cosine (normalized)
        idx = np.argsort(-sims)[:top_k]
        out = []
        for i in idx:
            r = cls._rows[int(i)]
            out.append({
                "act": r.get("act",""),
                "section": r.get("section",""),
                "title": r.get("title",""),
                "display": r.get("display",""),
                "score": float(sims[int(i)]),
                "doc_id": r.get("doc_id",""),
            })
        return out
