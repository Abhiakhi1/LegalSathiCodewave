from pathlib import Path
import json, sys

import chromadb
from chromadb.config import Settings

ROOT = Path(__file__).resolve().parents[2]
VDB  = ROOT / "vectordb"
DATA = ROOT / "data"

print(f"[DBG] VDB path : {VDB}")
print(f"[DBG] DATA path: {DATA}")

client = chromadb.PersistentClient(path=str(VDB), settings=Settings(anonymized_telemetry=False))
names = [c.name for c in client.list_collections()]
print(f"[DBG] Collections in vectordb ({len(names)}):")
for n in sorted(names):
    try:
        cnt = client.get_collection(n).count()
    except Exception:
        cnt = "?"
    print(f"  - {n} (count={cnt})")

try:
    cmap = json.load(open(DATA / "collection_map.json", "r", encoding="utf-8"))
    print(f"\n[DBG] Mapped labels: {len(cmap)}")
    missing = [(lbl, coll) for lbl, coll in cmap.items() if coll not in names]
    if missing:
        print("[WARN] Missing collections referenced by collection_map.json:")
        for lbl, coll in missing[:50]:
            print(f"  - {lbl} -> {coll}")
    else:
        print("[OK] All mapped collections exist.")
except FileNotFoundError:
    print("[WARN] data/collection_map.json not found")
