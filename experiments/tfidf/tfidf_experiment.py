
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

pairs = [
    ("The students submitted their assignments before the deadline.",
     "The students submitted their assignments before the deadline.",
     "Exact match"),

    ("Climate change increases the Earth's average temperature.",
     "Global warming raises the average temperature of the planet.",
     "Paraphrased"),

    ("Machine learning helps computers learn patterns from data.",
     "Computers can identify patterns in data using machine learning techniques.",
     "Reworded"),

    ("The university library contains thousands of research books.",
     "Photosynthesis allows plants to convert sunlight into energy.",
     "Unrelated"),

    ("The treatment improved the patients' condition.",
     "The treatment did not improve the patients' condition.",
     "Hard negative")
]

sentences = []
for first, second, label in pairs:
    sentences.extend([first, second])

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences)

results = []

print("\nTF-IDF SIMILARITY EXPERIMENT")
print("-" * 65)

for i, (first, second, label) in enumerate(pairs):
    vector1 = tfidf_matrix[2 * i]
    vector2 = tfidf_matrix[2 * i + 1]

    score = cosine_similarity(vector1, vector2)[0][0]

    print(f"\nPair: {label}")
    print(f"Sentence 1: {first}")
    print(f"Sentence 2: {second}")
    print(f"Similarity Score: {score:.4f}")

    results.append([label, first, second, round(score, 4)])

with open("experiments/tfidf/tfidf_results.csv", "w",
          newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Pair Type", "Sentence 1", "Sentence 2", "TF-IDF Score"])
    writer.writerows(results)

print("\nResults saved to experiments/tfidf/tfidf_results.csv")
