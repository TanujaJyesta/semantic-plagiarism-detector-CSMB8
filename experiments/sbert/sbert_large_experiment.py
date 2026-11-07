
from sentence_transformers import SentenceTransformer, util
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

print("Loading larger SBERT model...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

results = []

print("\nLARGER SBERT EXPERIMENT")
print("-" * 65)

for first, second, label in pairs:
    embeddings = model.encode([first, second], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()

    print(f"\nPair: {label}")
    print(f"Similarity Score: {score:.4f}")

    results.append([label, round(score, 4)])

with open("experiments/sbert/sbert_large_results.csv", "w",
          newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Pair Type", "MPNet Score"])
    writer.writerows(results)

print("\nResults saved to experiments/sbert/sbert_large_results.csv")
