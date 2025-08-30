# src/ingestion/build_sections_index.py
from __future__ import annotations
import re, json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Fallback to Chroma chunks for full coverage
import chromadb
from chromadb.config import Settings

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "Dataset"
DATA_DIR    = ROOT / "data"
VDB_DIR     = ROOT / "vectordb"
DATA_DIR.mkdir(exist_ok=True, parents=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --------- Patterns ----------
# Act name (best-effort)
ACT_NAME = re.compile(r"(?i)\b([A-Z][A-Za-z&\-\s]+ Act,\s*\d{4})\b")

# Header-style line: "Section 2(1)(j): Title", "Art. 14 - Equality"
HEADER = re.compile(
    r"(?i)^\s*(section|sections|sec\.?|s\.|§|article|articles|art\.|rule|rules|r\.)\s*"
    r"([0-9A-Z]+(?:\([0-9A-Za-z]+\))*)"
    r"(?:\s*(?:[-–:]\s*|\.\s*))?"
    r"(.*)$"
)

# Inline mentions within a line/paragraph (captures short title/snippet)
INLINE = re.compile(
    r"(?i)(section|sections|sec\.?|s\.|§|article|articles|art\.|rule|rules|r\.)\s*"
    r"([0-9A-Z]+(?:\([0-9A-Za-z]+\))*)"
    r"(?:\s*(?:[-–:]\s*|\.\s*))?"
    r"(.{0,120})"  # short tail as a pseudo-title
)

# Range like "Sections 84–92" or "Sections 3-5"
RANGE = re.compile(
    r"(?i)\bsections?\s+(\d+)\s*(?:[-–]|to)\s*(\d+)\b"
)

def _read(p: Path) -> str:
    try:
        return p.read_text("utf-8", errors="ignore")
    except Exception:
        return ""

def _expand_range(s: str) -> List[str]:
    """Expand 'Sections 84–92' into ['Section 84', ..., 'Section 92'] when numeric."""
    m = RANGE.search(s)
    if not m:
        return []
    a, b = m.group(1), m.group(2)
    try:
        lo, hi = int(a), int(b)
        if lo <= hi and (hi - lo) <= 50:  # safety cap
            return [f"Section {i}" for i in range(lo, hi + 1)]
    except Exception:
        pass
    return []

def _clean_title(t: str) -> str:
    t = (t or "").strip()
    # stop at sentence/bullet boundary to avoid overly long titles
    t = re.split(r"[.;•|]\s", t, maxsplit=1)[0].strip()
    return t[:160]

def _scan_files() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(DATASET_DIR.glob("*.txt")):
        text = _read(p)
        if not text:
            continue

        # file-level act hint (best effort)
        m_act = ACT_NAME.search(text)
        act_hint = m_act.group(1).strip() if m_act else ""

        lines = text.splitlines()
        n = len(lines)

        # Pass 1: header-style at line start
        for i, raw in enumerate(lines):
            line = raw.strip()
            if not line:
                continue
            # expand numeric ranges on this line, if any
            expanded = _expand_range(line)
            if expanded:
                for sec in expanded:
                    rows.append({
                        "doc_id": p.name,
                        "act": act_hint,
                        "section": sec,
                        "title": "",
                        "display": f"{(act_hint + ' — ') if act_hint else ''}{sec}",
                        "lead": "",  # filled later by body scan
                    })
                continue
            m = HEADER.match(line)
            if m:
                _kind, secno, title = m.group(1), m.group(2), _clean_title(m.group(3))
                sec = f"Section {secno}" if not _kind.lower().startswith("art") and _kind != "§" else f"Article {secno}"
                if _kind.lower().startswith("r"):
                    sec = f"Rule {secno}"
                disp = f"{(act_hint + ' — ') if act_hint else ''}{sec}{(': ' + title) if title else ''}"
                rows.append({
                    "doc_id": p.name,
                    "act": act_hint,
                    "section": sec,
                    "title": title or sec,
                    "display": disp,
                    "lead": "",  # filled later
                })

        # Pass 2: inline mentions within any line (fallback for narrative lists)
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            for m in INLINE.finditer(line):
                _kind, secno, tail = m.groups()
                sec = f"Section {secno}" if not _kind.lower().startswith("art") and _kind != "§" else f"Article {secno}"
                if _kind.lower().startswith("r"):
                    sec = f"Rule {secno}"
                title = _clean_title(tail)
                disp = f"{(act_hint + ' — ') if act_hint else ''}{sec}{(': ' + title) if title else ''}"
                rows.append({
                    "doc_id": p.name,
                    "act": act_hint,
                    "section": sec,
                    "title": title or sec,
                    "display": disp,
                    "lead": "",  # filled later
                })

        # Pass 3: attach leads (body after a header line) for header rows
        # build index of header positions
        header_pos: List[Tuple[int, Dict[str, Any]]] = []
        for i, raw in enumerate(lines):
            if HEADER.match(raw.strip()):
                header_pos.append((i, {}))
        header_pos.append((n, {}))  # sentinel

        # collect leads between headers
        h_idx = 0
        for idx in range(len(rows)):
            r = rows[idx]
            if r["doc_id"] != p.name:
                continue
            # find the nearest header position before this entry
            while h_idx + 1 < len(header_pos) and header_pos[h_idx + 1][0] <= n:
                # try to align roughly by title presence; break when beyond
                break
            # We’ll just use the entire file’s first 800 chars as lead if we can’t precisely map
            # (Header exact lead mapping is complex across mixed sources; acceptable for snippet scoring)
            if not r["lead"]:
                r["lead"] = text[:800]

    return rows

def _fallback_from_chroma() -> List[Dict[str, Any]]:
    """Ensure at least one entry per chunk from existing Chroma DB."""
    client = chromadb.PersistentClient(path=str(VDB_DIR), settings=Settings(anonymized_telemetry=False))
    rows: List[Dict[str, Any]] = []
    for coll_info in client.list_collections():
        coll = client.get_collection(coll_info.name)
        # Do NOT include "ids" here; Chroma returns them by default.
        recs = coll.get(include=["metadatas", "documents"])
        metadatas = recs.get("metadatas") or []
        documents = recs.get("documents") or []
        ids       = recs.get("ids") or []

        # Safety: if some client/build doesn't return ids, synthesize stable ones
        if not ids:
            ids = [f"{coll_info.name}#{i}" for i in range(len(documents))]

        for meta, doc, cid in zip(metadatas, documents, ids):
            acts = ""
            secs = ""
            if isinstance(meta, dict):
                acts = meta.get("acts") or ""
                secs = meta.get("sections") or ""
            # choose a section (if multiple, pick the first token)
            section = ""
            if isinstance(secs, str) and secs.strip():
                section = secs.split(";")[0].split(",")[0].strip()

            # a short, displayable title
            first_line = ""
            if doc:
                for line in doc.splitlines():
                    if line.strip():
                        first_line = line.strip()
                        break

            title = section if section else (first_line[:120] if first_line else cid)
            act_hint = acts.split(";")[0].split(",")[0].strip() if isinstance(acts, str) and acts.strip() else ""
            display = f"{(act_hint + ' — ') if act_hint else ''}{title}"
            lead = (doc or "")[:800]

            rows.append({
                "doc_id": cid,        # chunk-level id
                "act": act_hint,
                "section": section,   # may be empty
                "title": title,
                "display": display,
                "lead": lead,
            })
    return rows


def _dedup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = (r.get("doc_id",""), r.get("display",""))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out

def main():
    print("[DSI] Scanning dataset (headers + inline + ranges)…")
    rows = _scan_files()
    print(f"[DSI] Text-derived entries: {len(rows)}")

    print("[DSI] Adding entries from Chroma chunks for full coverage…")
    rows += _fallback_from_chroma()
    rows = _dedup(rows)
    print(f"[DSI] Total entries after union & dedup: {len(rows)}")

    if not rows:
        print("[DSI] No entries to index.")
        return

    print("[DSI] Embedding titles and leads…")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    titles = [r["display"] for r in rows]
    leads  = [r.get("lead","") or "" for r in rows]

    V_titles = model.encode(titles, normalize_embeddings=True)
    V_leads  = model.encode(leads,  normalize_embeddings=True)

    (DATA_DIR / "sections_index.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
    np.save(DATA_DIR / "sections_emb_titles.npy", V_titles)
    np.save(DATA_DIR / "sections_emb_leads.npy",  V_leads)
    print(f"[DSI] Wrote {DATA_DIR/'sections_index.json'} and embeddings (titles/leads).")

if __name__ == "__main__":
    main()
