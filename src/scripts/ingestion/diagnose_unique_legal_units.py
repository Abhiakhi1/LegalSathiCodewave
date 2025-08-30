# src/ingestion/diagnose_unique_legal_units.py
import json, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
rows = json.loads((ROOT/"data"/"sections_index.json").read_text("utf-8"))

is_chunk = lambda r: "#" in str(r.get("doc_id",""))
prefix = lambda s: ("Section" if s.startswith("Section ") else
                    "Article" if s.startswith("Article ") else
                    "Rule" if s.startswith("Rule ") else "")

# unique by (Act, Section/Article/Rule + number)
uniq_legal = set()
for r in rows:
    sec = (r.get("section") or "").strip()
    pre = prefix(sec)
    if not pre: 
        continue
    act = (r.get("act") or "").strip()
    # normalize like “Section 2(1)(j)” → (“Section”, “2(1)(j)”)
    num = sec.split(" ",1)[1] if " " in sec else sec
    uniq_legal.add((act, pre, num))

by_kind = Counter([k[1] for k in uniq_legal])
print("Unique legal units (by Act + kind + number):", len(uniq_legal))
print("Breakdown:", dict(by_kind))
