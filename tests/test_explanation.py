from backend.services.explanation import generate_explanation


# Example 1: Direct match
result = generate_explanation(
    tfidf_score=0.95,
    sbert_score=0.97,
    match_type="direct_match"
)

print("\n===== EXPLANATION 1 =====\n")

print(f"TF-IDF Similarity: {result['tfidf_percentage']}%")
print(f"SBERT Similarity: {result['sbert_percentage']}%")
print(f"Explanation: {result['explanation']}")


# Example 2: Semantic match
result = generate_explanation(
    tfidf_score=0.30,
    sbert_score=0.88,
    match_type="semantic_match"
)

print("\n===== EXPLANATION 2 =====\n")

print(f"TF-IDF Similarity: {result['tfidf_percentage']}%")
print(f"SBERT Similarity: {result['sbert_percentage']}%")
print(f"Explanation: {result['explanation']}")


# Example 3: Lexical overlap
result = generate_explanation(
    tfidf_score=0.82,
    sbert_score=0.55,
    match_type="lexical_overlap"
)

print("\n===== EXPLANATION 3 =====\n")

print(f"TF-IDF Similarity: {result['tfidf_percentage']}%")
print(f"SBERT Similarity: {result['sbert_percentage']}%")
print(f"Explanation: {result['explanation']}")