"""Core orchestrator: file -> full explainable plagiarism report.
This is the single function everyone's module plugs into.
"""
from backend.extraction.preprocess import extract_passages_from_file
from backend.retrieval.search_client import make_query, search_web
from backend.retrieval.scraper import scrape_main_text
from backend.similarity.lexical import tfidf_similarity, word_overlap_ratio
from backend.similarity.semantic import sbert_similarity
from backend.similarity.fusion import fused_score
from backend.explain.explainer import build_evidence_record


def scan_document(file_path: str, threshold: float = 0.55, alpha: float = 0.6,
                   top_k_sources: int = 3) -> dict:
    """Run the full pipeline on one document and return a structured report."""
    passages = extract_passages_from_file(file_path)

    flagged_evidence = []
    for passage in passages:
        query = make_query(passage)
        try:
            urls = search_web(query, num_results=top_k_sources)
        except RuntimeError:
            urls = []  # no API key set - extraction-only mode

        best = None
        for url in urls:
            source_text = scrape_main_text(url)
            if not source_text:
                continue
            lexical = tfidf_similarity(passage, source_text)
            semantic = sbert_similarity(passage, source_text)
            fused = fused_score(lexical, semantic, alpha)
            if best is None or fused > best["fused"]:
                best = {"url": url, "text": source_text, "lexical": lexical,
                        "semantic": semantic, "fused": fused}

        if best and best["fused"] > threshold:
            flagged_evidence.append(build_evidence_record(
                passage=passage,
                source_url=best["url"],
                source_excerpt=best["text"],
                lexical=best["lexical"],
                semantic=best["semantic"],
                fused=best["fused"],
            ))

    total = len(passages)
    flagged_count = len(flagged_evidence)
    plagiarism_percentage = round(100 * flagged_count / total, 1) if total else 0.0

    return {
        "total_passages": total,
        "flagged_count": flagged_count,
        "plagiarism_percentage": plagiarism_percentage,
        "flagged_passages": flagged_evidence,
    }
