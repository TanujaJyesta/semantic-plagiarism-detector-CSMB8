from backend.services.plagiarism_pipeline import (
    analyze_documents,
    save_analysis_result
)


source_file = "data/source.pdf"
submitted_file = "data/submitted.pdf"


print("\n========================================")
print("   EXPLAINABLE PLAGIARISM ANALYSIS")
print("========================================\n")


# Run complete analysis
result = analyze_documents(
    source_file,
    submitted_file,
    minimum_similarity=0.0
)


# Save result
save_analysis_result(
    result
)


# Display summary
print("Source sentences:",
      result["source_sentence_count"])

print(
    "Submitted sentences:",
    result["submitted_sentence_count"]
)

print(
    "Matched sentences:",
    result["matched_sentence_count"]
)


print("\n===== MATCH RESULTS =====\n")


for match in result["matches"]:

    print(
        f"Submitted S"
        f"{match['submitted_sentence_id']}"
    )

    print(
        f"Source S"
        f"{match['source_sentence_id']}"
    )

    print(
        f"TF-IDF: "
        f"{match['tfidf_percentage']}%"
    )

    print(
        f"SBERT: "
        f"{match['sbert_percentage']}%"
    )

    print(
        f"Match Type: "
        f"{match['match_type']}"
    )

    print(
        f"Explanation: "
        f"{match['explanation']}"
    )

    print("-" * 80)


print(
    "\nComplete result saved to:"
)

print(
    "results/final_analysis.json"
)