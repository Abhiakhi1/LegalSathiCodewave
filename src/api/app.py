# src/api/app.py
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import json

# Optional but useful for /debug/collections
import chromadb
from chromadb.config import Settings

# ---------- paths ----------
ROOT = Path(__file__).resolve().parents[2]          # project root
DATADIR = ROOT / "data"
VECTORDIR = ROOT / "vectordb"
WEBDIR = ROOT / "web"

# ---------- app ----------
app = FastAPI(title="LegalSathi – API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # relax if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the UI
if WEBDIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEBDIR), html=True), name="web")

# Simple redirect to the UI
@app.get("/", include_in_schema=False)
def _root():
    return RedirectResponse(url="/web/")

# Nice JSON errors for the /debug routes (you'll still see the full trace in the console)
@app.exception_handler(Exception)
async def json_errors(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": str(exc)})

# ---------- debug ----------
@app.get("/debug/status")
def debug_status():
    return {
        "corpus_stats_exists": (DATADIR / "corpus_stats.json").exists(),
        "collection_map_exists": (DATADIR / "collection_map.json").exists(),
        "labels_exists": (DATADIR / "labels.json").exists(),
        "vectordb_exists": VECTORDIR.exists(),
    }

def _chroma_client():
    VECTORDIR.mkdir(parents=True, exist_ok=True)
    # one consistent Setting prevents the “different settings” crash
    return chromadb.PersistentClient(path=str(VECTORDIR), settings=Settings(anonymized_telemetry=False))

@app.get("/debug/collections")
def debug_collections():
    client = _chroma_client()
    cols = [{"name": c.name, "count": c.count()} for c in client.list_collections()]
    mapped = {}
    map_path = DATADIR / "collection_map.json"
    if map_path.exists():
        mapped = json.loads(map_path.read_text(encoding="utf-8"))
    return {"chroma": cols, "mapped": mapped}

# ---------- first step agent ----------
from src.search.retriever import search_top_categories

class FirstStepRequest(BaseModel):
    prompt: str
    top_categories: int = 3
    per_cat_k: int = 6
    global_k: int = 8
    # optional meta filters coming from UI (keep names flat for simplicity)
    acts: list[str] | None = None
    sections: list[str] | None = None
    jurisdictions: list[str] | None = None

@app.post("/first-step-agent")
def first_step_agent(body: FirstStepRequest):
    meta_filters = {
        "acts": body.acts or [],
        "sections": body.sections or [],
        "jurisdictions": body.jurisdictions or [],
    }

    # retriever builds keywords, ranking, and hits; we just forward parameters
    result = search_top_categories(
        body.prompt,
        top_n_cats=body.top_categories,
        per_cat_k=body.per_cat_k,
        global_k=body.global_k,
        meta_filters=meta_filters,
    )

    if "hits" in result and "pointers" not in result:
        result["pointers"] = result["hits"]

    return result
