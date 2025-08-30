# src/ingestion/diagnose_sections.py
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "sections_index.json"

def main():
    rows = json.loads(DATA.read_text("utf-8"))
    by_file = defaultdict(int)
    for r in rows:
        by_file[r.get("doc_id","(unknown)")] += 1
    total = 0
    print("Entries per file:")
    for k in sorted(by_file, key=lambda x: by_file[x], reverse=True):
        print(f"  {k:50s}  {by_file[k]}")
        total += by_file[k]
    print(f"\nTotal entries: {total}")
    print(f"Distinct files counted: {len(by_file)}")

if __name__ == "__main__":
    main()
