
import csv

files = {
    "TF-IDF": "experiments/tfidf/tfidf_results.csv",
    "MiniLM": "experiments/sbert/sbert_results.csv",
    "MPNet": "experiments/sbert/sbert_large_results.csv"
}

scores = {}

for model, filename in files.items():
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            label = row["Pair Type"]

            if label not in scores:
                scores[label] = {}

            score_column = {
                "TF-IDF": "TF-IDF Score",
                "MiniLM": "SBERT Score",
                "MPNet": "MPNet Score"
            }[model]

            scores[label][model] = row[score_column]

output = "experiments/comparison/model_comparison.csv"

with open(output, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Pair Type", "TF-IDF", "MiniLM", "MPNet"])

    for label, values in scores.items():
        writer.writerow([
            label,
            values["TF-IDF"],
            values["MiniLM"],
            values["MPNet"]
        ])

print("MODEL COMPARISON")
print("-" * 55)

for label, values in scores.items():
    print(f"\n{label}")
    for model, score in values.items():
        print(f"{model}: {score}")

print(f"\nComparison saved to {output}")
