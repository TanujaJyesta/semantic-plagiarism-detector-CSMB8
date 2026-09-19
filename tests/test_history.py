from backend.services.history_service import (
    get_all_analyses,
    get_analysis_by_id
)


print("\n========================================")
print("       ANALYSIS HISTORY TEST")
print("========================================\n")


# -----------------------------------------
# Get all analyses
# -----------------------------------------

analyses = get_all_analyses()


print(
    f"Total analyses: {len(analyses)}"
)


for analysis in analyses:

    print("\n----------------------------------------")

    print(
        f"Analysis ID: "
        f"{analysis['id']}"
    )

    print(
        f"Source: "
        f"{analysis['source_document']}"
    )

    print(
        f"Submitted: "
        f"{analysis['submitted_document']}"
    )

    print(
        f"Matched sentences: "
        f"{analysis['matched_sentence_count']}"
    )

    print(
        f"Created at: "
        f"{analysis['created_at']}"
    )


# -----------------------------------------
# Get one analysis
# -----------------------------------------

if analyses:

    analysis_id = analyses[0]["id"]

    print("\n========================================")
    print(
        f"DETAILS OF ANALYSIS {analysis_id}"
    )
    print("========================================")


    analysis = get_analysis_by_id(
        analysis_id
    )


    print(
        f"\nSource: "
        f"{analysis['source_document']}"
    )

    print(
        f"Submitted: "
        f"{analysis['submitted_document']}"
    )

    print(
        f"Total matches: "
        f"{len(analysis['matches'])}"
    )


    for match in analysis["matches"]:

        print("\n----------------------------------------")

        print(
            f"Source S"
            f"{match['source_sentence_id']}"
            f" <-> "
            f"Submitted S"
            f"{match['submitted_sentence_id']}"
        )

        print(
            f"TF-IDF: "
            f"{match['tfidf_score']}"
        )

        print(
            f"SBERT: "
            f"{match['sbert_score']}"
        )

        print(
            f"Match Type: "
            f"{match['match_type']}"
        )

        print(
            f"Explanation: "
            f"{match['explanation']}"
        )

else:

    print("\nNo analyses found.")