<<<<<<< HEAD
LegalSathi – First-Step Agent

A fast, offline-friendly legal “first step” agent that:
- understands a user prompt,
- auto-selects the right legal fields (categories),
- retrieves only those document chunks that cover most of the query,
- exposes a simple web UI at http://127.0.0.1:8000/web/.

The system uses hybrid scoring (dense embeddings + curated cues + IDF keywords) and adapts top_categories, per_cat_k, and global_k automatically for each query.
Repo Structure

legal-sathi-agent/
├─ data/                 # runtime artifacts (see below)
│  ├─ collection_map.json      # collection → display/category mapping
│  ├─ corpus_stats.json        # IDF stats for keyword fallback
│  ├─ cues.json                # (optional) curated cue phrases
│  ├─ labels.json              # label aliases used by ranking layer
│  ├─ sections_index.json      # registry of all sections/chunks + metadata
│  ├─ sections_emb.npy         # LARGE: embeddings (main bodies)   [not in Git]
│  ├─ sections_emb_leads.npy   # LARGE: embeddings (lead paras)    [not in Git]
│  └─ sections_emb_titles.npy  # LARGE: embeddings (titles)        [not in Git]
├─ src/
│  ├─ api/            # FastAPI app (entrypoint)
│  ├─ search/         # retriever & ranking
│  ├─ pipeline/       # normalizer, section_ranker, sense_router, ...
│  ├─ ingestion/      # scripts to build data/ artifacts
│  ├─ keywords/       # keyword extraction / meta filters
│  ├─ router/         # category router (lightweight)
│  └─ tools/          # debug helpers
├─ web/               # simple UI
├─ .env               # local config (not committed)
├─ .gitignore
└─ README.md
Data Artifacts
Keep in Git (small, portable)
- data/collection_map.json
- data/corpus_stats.json
- data/cues.json *(optional but recommended)*
- data/labels.json
- data/sections_index.json
Do NOT commit (large binaries; excluded via .gitignore)
- data/sections_emb.npy
- data/sections_emb_leads.npy
- data/sections_emb_titles.npy

👉 These .npy files are large vector embeddings.
They are either:
1. Published once via GitHub Releases/S3/etc. (preferred), or
2. Regenerated locally with the ingestion scripts in src/ingestion/.
Setup
Requirements
- Python 3.10+
- Windows/Mac/Linux
Create a virtual env
# Windows (PowerShell)
py -3 -m venv .venv
. .venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
Install dependencies
pip install -r requirements.txt
Running
From repo root (after activating venv):

uvicorn src.api.app:app --reload --port 8000

Now open the UI:
http://127.0.0.1:8000/web/
Embeddings: How to Get Them
Because .npy embeddings are too large for Git, you need them locally before running.

Options:

1. Download Prebuilt Embeddings (recommended)
   - Go to this repo’s Releases page (or S3/Drive link shared by team).
   - Download:
     - sections_emb.npy
     - sections_emb_leads.npy
     - sections_emb_titles.npy
   - Place them into data/ folder.

2. Regenerate Locally
   - If you have the dataset, run:
     python -m src.ingestion.build_sections_index
   - This will rebuild the .npy files from raw dataset.
Debug / Health Check
Two helper endpoints:

- Check status
  curl http://127.0.0.1:8000/debug/status

- List collections
  curl http://127.0.0.1:8000/debug/collections
Features Recap
- Hybrid keyword + embedding + cue ranking
- Automatic tuning of retrieval parameters (top_categories, per_cat_k, global_k)
- FastAPI backend + lightweight HTML/JS frontend
- Designed to run fully local (no API keys needed)
License
MIT License – free to use and modify.
=======
# LegalSathi
>>>>>>> 0890fc2557cab1a26472b796d3e35e96f1dad974
