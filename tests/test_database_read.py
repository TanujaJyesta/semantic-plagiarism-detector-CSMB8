from backend.database.database import SessionLocal
from backend.database.models import Analysis, MatchResult


db = SessionLocal()


print("\n========================================")
print("   DATABASE CONTENT")
print("========================================\n")


analyses = db.query(Analysis).all()


print(
    f"Total analyses: {len(analyses)}"
)


for analysis in analyses:

    print("\n----------------------------------------")

    print(
        f"Analysis ID: {analysis.id}"
    )

    print(
        f"Source: {analysis.source_document}"
    )

    print(
        f"Submitted: {analysis.submitted_document}"
    )

    print(
        f"Source sentences: "
        f"{analysis.source_sentence_count}"
    )

    print(
        f"Submitted sentences: "
        f"{analysis.submitted_sentence_count}"
    )

    print(
        f"Matched sentences: "
        f"{analysis.matched_sentence_count}"
    )


    print("\nMatches:")


    for match in analysis.matches:

        print(
            f"\n  Source S{match.source_sentence_id}"
            f" <-> "
            f"Submitted S{match.submitted_sentence_id}"
        )

        print(
            f"  TF-IDF: {match.tfidf_score}"
        )

        print(
            f"  SBERT: {match.sbert_score}"
        )

        print(
            f"  Type: {match.match_type}"
        )

        print(
            f"  Explanation: {match.explanation}"
        )


db.close()