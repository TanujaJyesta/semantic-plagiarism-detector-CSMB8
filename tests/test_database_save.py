from backend.services.plagiarism_pipeline import analyze_documents
from backend.services.database_service import save_analysis_to_database


source_file = "data/source.pdf"
submitted_file = "data/submitted.pdf"


print("\n========================================")
print("   DATABASE SAVE TEST")
print("========================================\n")


# Run the existing ML pipeline
result = analyze_documents(
    source_file,
    submitted_file,
    minimum_similarity=0.0
)


# Save result into SQLite
analysis_id = save_analysis_to_database(
    result
)


print(
    f"Analysis successfully saved!"
)

print(
    f"Analysis ID: {analysis_id}"
)