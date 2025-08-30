# src/keywords/__init__.py
from .extractor import extract_keywords  # existing

# Re‑export build_meta_filters so callers can do: `from keywords import build_meta_filters`
try:
    from .meta import build_meta_filters
except Exception:
    def build_meta_filters(_terms, **kwargs):
        return {"acts": [], "sections": [], "jurisdictions": []}
