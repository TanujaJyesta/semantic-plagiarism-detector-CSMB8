from backend.database.database import SessionLocal
from backend.database.models import Analysis


def get_all_analyses():
    """
    Retrieve all previous plagiarism analyses.
    """

    db = SessionLocal()

    try:
        analyses = (
            db.query(Analysis)
            .order_by(Analysis.created_at.desc())
            .all()
        )

        results = []

        for analysis in analyses:

            results.append({
                "id": analysis.id,

                "source_document":
                    analysis.source_document,

                "submitted_document":
                    analysis.submitted_document,

                "source_sentence_count":
                    analysis.source_sentence_count,

                "submitted_sentence_count":
                    analysis.submitted_sentence_count,

                "matched_sentence_count":
                    analysis.matched_sentence_count,

                "created_at":
                    analysis.created_at.isoformat() if analysis.created_at else None,

                "filename": analysis.filename or analysis.submitted_document,

                "status": analysis.status,

                "suspicious_matches": analysis.suspicious_match_count,

                "sources_found": analysis.source_count,

                "overall_risk": analysis.overall_risk,
            })

        return results

    finally:
        db.close()


def get_analysis_by_id(analysis_id):
    """
    Retrieve one analysis and all of its
    sentence-level matches.
    """

    db = SessionLocal()

    try:

        analysis = (
            db.query(Analysis)
            .filter(Analysis.id == analysis_id)
            .first()
        )

        if analysis is None:
            return None

        result = {
            "id": analysis.id,

            "source_document":
                analysis.source_document,

            "submitted_document":
                analysis.submitted_document,

            "source_sentence_count":
                analysis.source_sentence_count,

            "submitted_sentence_count":
                analysis.submitted_sentence_count,

            "matched_sentence_count":
                analysis.matched_sentence_count,

            "created_at":
                analysis.created_at.isoformat() if analysis.created_at else None,

            "filename": analysis.filename or analysis.submitted_document,

            "status": analysis.status,

            "summary": {
                "total_passages": analysis.total_passages,
                "searched_passages": analysis.searched_passages,
                "sources_found": analysis.source_count,
                "sources_retrieved": analysis.retrieved_source_count,
                "suspicious_matches": analysis.suspicious_match_count,
                "overall_risk": analysis.overall_risk,
            },

            "warnings": [analysis.error_message] if analysis.error_message else [],

            "matches": []
        }


        for match in analysis.matches:

            result["matches"].append({

                "source_sentence_id":
                    match.source_sentence_id,

                "submitted_sentence_id":
                    match.submitted_sentence_id,

                "source_page":
                    match.source_page,

                "submitted_page":
                    match.submitted_page,

                "source_text":
                    match.source_text,

                "submitted_text":
                    match.submitted_text,

                "tfidf_score":
                    match.tfidf_score,

                "sbert_score":
                    match.sbert_score,

                "match_type":
                    match.match_type,

                "explanation":
                    match.explanation
                ,

                "page": match.page_number or match.submitted_page,

                "source": {
                    "title": match.source_title or "Retrieved source",
                    "url": match.source_url,
                },

                "submitted_passage": match.submitted_text,

                "matched_passage": match.source_text,

                "similarity": {
                    "tfidf": match.tfidf_score,
                    "sbert": match.sbert_score,
                    "combined": match.combined_score,
                },

                "risk": match.risk_level,
            })

        result["sources"] = [
            {
                "title": source.title,
                "url": source.url,
                "retrieval_status": source.retrieval_status,
                "error": source.error_message,
                "search_score": source.search_score,
            }
            for source in analysis.sources
        ]


        return result

    finally:
        db.close()


def clear_all_analyses():
    """Delete all stored analyses, match results, and source records."""
    db = SessionLocal()
    try:
        db.query(Analysis).delete()
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_analysis_by_id(analysis_id: int):
    """Delete a single analysis and its related records by ID."""
    db = SessionLocal()
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            return False
        db.delete(analysis)
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

