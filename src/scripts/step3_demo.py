# src/step3_demo.py
from src.keywords.extractor import extract_keywords
from src.keywords.meta import build_meta_filters
from src.router.ranker import rank_categories

def run(prompt: str):
    kws = extract_keywords(prompt, k=12, tau=0.55)
    meta = build_meta_filters(kws)
    ranks = rank_categories(prompt, top_k=5)

    print("\n=== KEYWORDS (ranked, no useless terms) ===")
    for k in kws:
        print(f"- {k['term']}  | specificity={k['specificity']:.3f}  | {k['reason']}")

    print("\n=== META FILTERS (for retrieval) ===")
    print(meta)

    print("\n=== DB RANKING (Top-5 categories) ===")
    for r in ranks:
        print(f"- {r['category']}  | score={r['score']:.3f}  | collection={r['sanitized_collection']}")

if __name__ == "__main__":
    run("Pune me mere khet par padosi ne boundary cross karke encroachment kiya. Tehsildar ko complaint kaise? Which section applies in Maharashtra?")
    run("A murder took place. Which law and which Section applies for homicide? What is the punishment?")
    run("Dowry harassment under IPC / BNS – what remedies are available and which sections to cite in Delhi?")
