from backend.database.database import SessionLocal
from backend.database.models import Analysis, MatchResult, SourceRecord


def save_analysis_to_database(result):
    """
    Save a complete plagiarism analysis result
    and all sentence-level matches into SQLite.
    """

    db = SessionLocal()

    try:

        # -----------------------------------------
        # Create analysis record
        # -----------------------------------------

        analysis = Analysis(
            source_document=result["source_document"],
            submitted_document=result["submitted_document"],
            source_sentence_count=result["source_sentence_count"],
            submitted_sentence_count=result["submitted_sentence_count"],
            matched_sentence_count=result["matched_sentence_count"]
        )

        db.add(analysis)

        # Flush so SQLAlchemy generates analysis.id
        db.flush()


        # -----------------------------------------
        # Save every sentence match
        # -----------------------------------------

        for match in result["matches"]:

            match_record = MatchResult(
                analysis_id=analysis.id,

                source_sentence_id=
                    match["source_sentence_id"],

                submitted_sentence_id=
                    match["submitted_sentence_id"],

                source_page=
                    match.get("source_page"),

                submitted_page=
                    match.get("submitted_page"),

                source_text=
                    match["source_text"],

                submitted_text=
                    match["submitted_text"],

                tfidf_score=
                    match["tfidf_score"],

                sbert_score=
                    match["sbert_score"],

                match_type=
                    match["match_type"],

                explanation=
                    match["explanation"]
            )

            db.add(match_record)


        # -----------------------------------------
        # Commit everything
        # -----------------------------------------

        db.commit()

        return analysis.id


    except Exception:

        db.rollback()

        raise


    finally:

        db.close()


def save_web_analysis_to_database(result):
    """Persist the compact web-analysis report and its source evidence."""
    db = SessionLocal()
    try:
        summary = result["summary"]
        analysis = Analysis(
            filename=result["filename"],
            source_document=result["filename"],
            submitted_document=result["filename"],
            status=result["status"],
            total_passages=summary["total_passages"],
            searched_passages=summary["searched_passages"],
            source_count=summary["sources_found"],
            retrieved_source_count=summary["sources_retrieved"],
            suspicious_match_count=summary["suspicious_matches"],
            overall_risk=summary["overall_risk"],
            error_message="; ".join(result.get("warnings", [])) or None,
            source_sentence_count=summary["total_passages"],
            submitted_sentence_count=summary["total_passages"],
            matched_sentence_count=summary["suspicious_matches"],
        )
        db.add(analysis)
        db.flush()
        for source in result.get("sources", []):
            db.add(SourceRecord(
                analysis_id=analysis.id,
                title=source["title"],
                url=source["url"],
                retrieval_status=source["retrieval_status"],
                error_message=source.get("error"),
                search_score=source.get("search_score", 0.0),
            ))
        for match in result.get("matches", []):
            similarity = match["similarity"]
            db.add(MatchResult(
                analysis_id=analysis.id,
                source_sentence_id=match.get("source_position"),
                submitted_sentence_id=match.get("position"),
                source_page=None,
                submitted_page=match.get("page"),
                page_number=match.get("page"),
                source_text=match["matched_passage"],
                submitted_text=match["submitted_passage"],
                source_title=match["source"]["title"],
                source_url=match["source"]["url"],
                tfidf_score=similarity["tfidf"],
                sbert_score=similarity["sbert"],
                combined_score=similarity["combined"],
                match_type=match["match_type"],
                risk_level=match["risk"],
                explanation=match["explanation"],
            ))
        db.commit()
        return analysis.id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
