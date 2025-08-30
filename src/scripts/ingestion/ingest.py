# src/ingestion/ingest.py
"""
FREE ingestion pipeline (no OpenAI, no cloud)
- Reads all *.txt files from ./Dataset
- Chunk-splits content (legal-aware)
- Extracts metadata (Acts / Sections / Years / Jurisdictions)
- Embeds with SentenceTransformers (all-MiniLM-L6-v2) on CPU
- Stores vectors locally in ChromaDB (persistent folder: ./vectordb)
- Writes:
    data/corpus_stats.json   (chunk-level DF for IDF)
    data/labels.json         (category labels from filenames)
    data/collection_map.json (orig category -> sanitized collection name)

Run:
  python -m src.ingestion.ingest
"""

import os, re, json, hashlib
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import Counter

from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np

import chromadb
from sentence_transformers import SentenceTransformer

# -------------------- Setup & Config --------------------
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # repo root
DATASET_DIR  = PROJECT_ROOT / "Dataset"              # unzipped here
OUT_DIR      = PROJECT_ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_MODEL = SentenceTransformer(LOCAL_EMBED_MODEL, device="cpu")
EMB_DIM = _MODEL.get_sentence_embedding_dimension()

VECTOR_DIR = PROJECT_ROOT / "vectordb"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(path=str(VECTOR_DIR))

# -------------------- Chunking (legal-aware) --------------------
SECTION_SPLIT = re.compile(
    r'(?im)^\s*(section\s+\d+[a-z\-]*|sec\.\s*\d+[a-z\-]*|chapter\s+\w+|act\s+\d{4})\s*$'
)
PARA_SPLIT = re.compile(r'\n{2,}')

def chunk_text(text: str, target_tokens: int = 800, overlap_tokens: int = 120) -> List[str]:
    parts = SECTION_SPLIT.split(text)
    raw = "\n".join([p for p in parts if p and p.strip()])
    paras = [p.strip() for p in PARA_SPLIT.split(raw) if p.strip()]

    chunks: List[str] = []
    buff: List[str] = []
    size = 0
    for p in paras:
        tok = len(p.split())
        if size + tok > target_tokens and buff:
            chunks.append("\n\n".join(buff))
            if overlap_tokens > 0 and buff:
                tail_words = buff[-1].split()
                tail = " ".join(tail_words[-overlap_tokens:]) if tail_words else ""
                buff = [tail] if tail else []
                size = len(tail.split()) if tail else 0
            else:
                buff, size = [], 0
        buff.append(p)
        size += tok
    if buff:
        chunks.append("\n\n".join(buff))
    return chunks

# -------------------- Metadata extraction --------------------
ACT_RE   = re.compile(r'\b([A-Z][A-Za-z]*(?: [A-Za-z&\-]+)* Act,? \d{4})\b')
SECT_RE  = re.compile(r'(?i)\b(?:Section|Sec\.)\s+(\d+[A-Z\-]*)\b')
YEAR_RE  = re.compile(r'\b(18\d{2}|19\d{2}|20\d{2})\b')
JUR_RE   = re.compile(
    r'(?i)\b(India|Andhra Pradesh|Assam|Bihar|Chhattisgarh|Delhi|Goa|Gujarat|Haryana|Himachal Pradesh|'
    r'Jammu(?: &| and)? Kashmir|Jharkhand|Karnataka|Kerala|Madhya Pradesh|Maharashtra|Manipur|Meghalaya|'
    r'Mizoram|Nagaland|Odisha|Punjab|Rajasthan|Sikkim|Tamil Nadu|Telangana|Tripura|Uttar Pradesh|Uttarakhand|'
    r'West Bengal|Pune|Mumbai|Bengaluru|Chennai|Hyderabad)\b'
)

def extract_metadata(category: str, doc_name: str, idx: int, chunk: str) -> Dict[str, Any]:
    acts = sorted(set(ACT_RE.findall(chunk)))
    secs = sorted(set(SECT_RE.findall(chunk)))
    yrs  = [int(y) for y in sorted(set(YEAR_RE.findall(chunk)))]
    jurs = sorted(set(JUR_RE.findall(chunk)))
    payload = {
        "doc_id": doc_name,
        "chunk_id": f"{doc_name}#{idx:03d}",
        "category": category,                      # original label (unsanitized)
        # 'collection' will be added later at add_points() time
        "acts": acts,
        "sections": [f"Section {s}" for s in secs],
        "jurisdictions": jurs,
        "year_mentions": yrs,
        "keywords": sorted(set(acts + [f"Section {s}" for s in secs])),
        "source": f"Dataset/{doc_name}",
        "chunk_text_head": chunk[:220],
        "hash": hashlib.sha256(chunk.encode("utf-8", errors="ignore")).hexdigest(),
    }
    return payload

# -------------------- Embeddings (local) --------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    emb = _MODEL.encode(texts, normalize_embeddings=True)
    if isinstance(emb, np.ndarray):
        emb = emb.tolist()
    return emb

# -------------------- Corpus stats for IDF --------------------
TERM_RE = re.compile(r'(?i)\b([a-z][a-z\- ]{3,})\b')

def terms_for_idf(text: str) -> Set[str]:
    cands = [m.group(1).strip().lower() for m in TERM_RE.finditer(text)]
    cands = [re.sub(r'\s+', ' ', t) for t in cands if len(t.split()) <= 4]
    return set(cands)

# -------------------- Chroma helpers --------------------
def normalize_collection_name(name: str) -> str:
    """
    Make a Chroma-safe collection name:
    - Allowed: [a-zA-Z0-9._-], start/end alphanumeric
    - Also add a short hash to guarantee uniqueness
    """
    base = re.sub(r'[^a-zA-Z0-9._-]+', '-', name)       # replace invalid with '-'
    base = base.strip('-._')                             # must start/end alnum
    if len(base) < 3:
        base = (base + "cat")[:3]                        # min length 3
    if len(base) > 80:
        base = base[:80]                                 # keep it tidy
    suffix = hashlib.sha1(name.encode('utf-8')).hexdigest()[:6]
    return f"{base}-{suffix}"

COLLECTION_MAP: Dict[str, str] = {}  # original -> sanitized

def get_collection(orig_name: str):
    safe = COLLECTION_MAP.get(orig_name)
    if not safe:
        safe = normalize_collection_name(orig_name)
        COLLECTION_MAP[orig_name] = safe
    return client.get_or_create_collection(
        name=safe,
        metadata={"hnsw:space": "cosine"}
    )

# HARD FLATTEN → Chroma only accepts primitives in metadata
PRIMS = (str, int, float, bool, type(None))

def _to_primitive(v):
    if isinstance(v, PRIMS):
        return v
    if isinstance(v, list):
        try:
            return "; ".join(str(x) for x in v)
        except Exception:
            return json.dumps(v, ensure_ascii=False)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def _flatten_meta(meta: dict, collection_name: str) -> dict:
    meta = dict(meta)
    meta["collection"] = collection_name  # store the sanitized name too
    return {k: _to_primitive(v) for k, v in meta.items()}

def add_points(category_label: str, ids: List[str], docs: List[str], metas: List[Dict[str, Any]], vecs: List[List[float]]):
    coll = get_collection(category_label)
    safe_name = COLLECTION_MAP[category_label]
    metas_flat = [_flatten_meta(m, safe_name) for m in metas]
    coll.add(ids=ids, documents=docs, metadatas=metas_flat, embeddings=vecs)

# -------------------- Ingestion per file --------------------
def ingest_file(path: Path):
    category_label = path.stem                  # original filename stem
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text)
    if not chunks:
        print(f"[WARN] No chunks for {path.name}")
        return 0, set()

    batch = 32
    total_chunks = 0
    df_terms: Set[str] = set()

    for i in range(0, len(chunks), batch):
        sub = chunks[i:i+batch]
        vecs = embed_texts(sub)

        ids   = [f"{path.name}#{j}" for j in range(i, i+len(sub))]
        metas = [extract_metadata(category_label, path.name, j, ch) for j, ch in enumerate(sub, start=i)]

        add_points(category_label, ids=ids, docs=sub, metas=metas, vecs=vecs)

        for ch in sub:
            df_terms |= terms_for_idf(ch)

        total_chunks += len(sub)

    return total_chunks, df_terms

# -------------------- Main --------------------
def main():
    txt_files = sorted(DATASET_DIR.glob("*.txt"))
    if not txt_files:
        raise SystemExit("[ERROR] No .txt files found in ./Dataset. Did you unzip Dataset.zip at the project root?")

    print(f"[INFO] Free stack: SentenceTransformers={LOCAL_EMBED_MODEL} (dim={EMB_DIM}), ChromaDB at {VECTOR_DIR}")
    print(f"[INFO] Found {len(txt_files)} files. Starting ingestion...")

    corpus_df_counter = Counter()
    total_chunks = 0

    for f in tqdm(txt_files, desc="Ingesting"):
        n_chunks, df_terms = ingest_file(f)
        total_chunks += n_chunks
        for t in df_terms:
            corpus_df_counter[t] += 1

    stats = {
        "N_docs": int(total_chunks),
        "df": {k: int(v) for k, v in corpus_df_counter.items()}
    }
    with open(OUT_DIR / "corpus_stats.json", "w", encoding="utf-8") as fp:
        json.dump(stats, fp)

    labels = [p.stem for p in txt_files]
    with open(OUT_DIR / "labels.json", "w", encoding="utf-8") as fp:
        json.dump(labels, fp, indent=2)

    # write the original->sanitized map
    with open(OUT_DIR / "collection_map.json", "w", encoding="utf-8") as fp:
        json.dump(COLLECTION_MAP, fp, indent=2)

    print(f"[OK] Ingested {len(txt_files)} files, {total_chunks} chunks total.")
    print(f"[OK] Wrote {OUT_DIR/'corpus_stats.json'}, {OUT_DIR/'labels.json'}, and {OUT_DIR/'collection_map.json'}")
    print(f"[OK] Vector DB persisted at: {VECTOR_DIR}")

if __name__ == "__main__":
    main()
