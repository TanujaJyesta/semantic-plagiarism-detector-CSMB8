
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sentence_transformers import SentenceTransformer

DATA_FILE = "threshold_dataset.csv"
THRESHOLDS = np.arange(0.30, 1.00, 0.05)

df = pd.read_csv(DATA_FILE).fillna("")
required = ["source_doc_id", "submitted_doc_id", "submitted_passage",
            "source_passage", "is_plagiarized", "split"]

if not all(col in df.columns for col in required):
    raise ValueError("CSV is missing required columns.")

df["is_plagiarized"] = pd.to_numeric(df["is_plagiarized"], errors="coerce")
if df["is_plagiarized"].isna().any() or not df["is_plagiarized"].isin([0, 1]).all():
    raise ValueError("is_plagiarized must contain only 0 or 1.")

df["split"] = df["split"].str.strip().str.lower()
if not {"train", "validation"}.issubset(set(df["split"])):
    raise ValueError("Add labelled train and validation rows before running.")

train = df[df["split"] == "train"]
validation = df[df["split"] == "validation"]
test = df[df["split"] == "test"]

if validation["is_plagiarized"].nunique() < 2:
    raise ValueError("Validation data must contain both classes: 0 and 1.")

if train.empty or train[["submitted_passage", "source_passage"]].eq("").any().any():
    raise ValueError("Training passages cannot be empty.")

if df[["submitted_passage", "source_passage"]].eq("").any().any():
    raise ValueError("Passages cannot be empty.")

# Fit TF-IDF using training passages only.
train_texts = train["submitted_passage"].tolist() + train["source_passage"].tolist()
tfidf = TfidfVectorizer()
tfidf.fit(train_texts)

def tfidf_scores(rows):
    submitted = tfidf.transform(rows["submitted_passage"].tolist())
    source = tfidf.transform(rows["source_passage"].tolist())
    return np.array([
        cosine_similarity(submitted[i], source[i])[0, 0]
        for i in range(len(rows))
    ])

print("Loading MiniLM model...")
minilm = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading MPNet model...")
mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def sbert_scores(model, rows):
    submitted = model.encode(
        rows["submitted_passage"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    source = model.encode(
        rows["source_passage"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return np.sum(submitted * source, axis=1)

models = {
    "TF-IDF": tfidf_scores,
    "MiniLM": lambda rows: sbert_scores(minilm, rows),
    "MPNet": lambda rows: sbert_scores(mpnet, rows)
}

results = []
selected = {}

for name, score_function in models.items():
    val_scores = score_function(validation)
    y_val = validation["is_plagiarized"].astype(int).to_numpy()

    best = None

    for threshold in THRESHOLDS:
        predictions = (val_scores >= threshold).astype(int)
        precision = precision_score(y_val, predictions, zero_division=0)
        recall = recall_score(y_val, predictions, zero_division=0)
        f1 = f1_score(y_val, predictions, zero_division=0)
        accuracy = accuracy_score(y_val, predictions)

        results.append({
            "model": name,
            "threshold": round(float(threshold), 2),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        })

        candidate = (f1, precision, float(threshold))
        if best is None or candidate > best:
            best = candidate

    selected[name] = best[2]

    print(f"\n{name}")
    print(f"Selected validation threshold: {best[2]:.2f}")
    print(f"Validation F1: {best[0]:.4f}")

    if not test.empty:
        if test["is_plagiarized"].nunique() < 2:
            print("Test set needs both classes for a meaningful evaluation.")
        else:
            test_scores = score_function(test)
            y_test = test["is_plagiarized"].astype(int).to_numpy()
            test_predictions = (test_scores >= best[2]).astype(int)

            print("Held-out test results:")
            print(f"Precision: {precision_score(y_test, test_predictions, zero_division=0):.4f}")
            print(f"Recall:    {recall_score(y_test, test_predictions, zero_division=0):.4f}")
            print(f"F1-score:  {f1_score(y_test, test_predictions, zero_division=0):.4f}")
            print(f"Accuracy:  {accuracy_score(y_test, test_predictions):.4f}")

pd.DataFrame(results).to_csv("threshold_results.csv", index=False)

with open("selected_thresholds.txt", "w") as file:
    for model, threshold in selected.items():
        file.write(f"{model}: {threshold:.2f}\n")

print("\nThreshold analysis completed.")
print("Saved threshold_results.csv and selected_thresholds.txt")
