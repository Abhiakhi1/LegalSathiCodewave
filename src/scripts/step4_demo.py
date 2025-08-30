# src/step4_demo.py
from __future__ import annotations
from src.keywords.extractor import extract_keywords
from src.keywords.meta import build_meta_filters
from src.router.ranker import rank_categories
from src.search.retriever import search_top_categories

def run(prompt: str, top_cats: int = 3, per_cat_k: int = 6, global_k: int = 8, use_reranker: bool = False):
    # 1) Keywords (noise‑free)
    kws = extract_keywords(prompt, k=12, tau=0.55)
    meta = build_meta_filters(kws)

    # 2) Category router
    ranked = rank_categories(prompt, top_k=max(top_cats, 1))
    top_for_search = ranked[: top_cats]

    # 3) Retrieval
    result = search_top_categories(
        prompt=prompt,
        top_categories=top_for_search,
        meta_filter=meta,
        per_category_k=per_cat_k,
        global_k=global_k,
        use_reranker=use_reranker,
    )

    # --- Pretty print ---
    print("\n" + "="*88)
    print(f"PROMPT: {prompt}\n")

    print("[KEYWORDS]")
    for k in kws:
        print(f"- {k['term']}  | specificity={k['specificity']:.3f}  | {k['reason']}")

    print("\n[META FILTERS]")
    print(meta)

    print("\n[DB RANKING]")
    for r in ranked[:top_cats]:
        print(f"- {r['category']}  | score={r['score']:.3f}  | collection={r['sanitized_collection']}")

    print("\n[HITS (pointers to embeddings)]")
    for h in result["hits"]:
        print(f"- {h['category']}  | {h['id']}  | Sim {h['sim']:.3f}  | Sections {h.get('sections','')}  | Acts {h.get('acts','')}")

if __name__ == "__main__":
    run("boundary encroachment neighbour land")
