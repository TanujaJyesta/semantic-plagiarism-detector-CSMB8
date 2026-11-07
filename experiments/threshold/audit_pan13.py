
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(r"D:\pan13-text-alignment-test-corpus2-2013-01-21")
ANNOTATIONS = ROOT / "pan13-text-alignment-test-corpus2-2013-01-21"
CSV_FILE = Path(r"D:\meeting_preparation\experiments\threshold\threshold_dataset.csv")

folders = [
    "02-no-obfuscation",
    "03-random-obfuscation",
    "04-translation-obfuscation",
    "05-summary-obfuscation"
]

print("\nOBFUSCATION ATTRIBUTE AUDIT")
print("=" * 45)

for folder in folders:
    counts = Counter()
    missing_examples = []

    for xml_file in (ANNOTATIONS / folder).glob("*.xml"):
        try:
            root = ET.parse(xml_file).getroot()
            for feature in root.findall(".//feature[@name='plagiarism']"):
                value = feature.attrib.get("obfuscation")
                if value is None:
                    counts["MISSING ATTRIBUTE"] += 1
                    if len(missing_examples) < 3:
                        missing_examples.append(xml_file.name)
                else:
                    counts[value] += 1
        except ET.ParseError:
            continue

    print(f"\n{folder}:")
    for name, count in counts.items():
        print(f"  {name}: {count}")

    if missing_examples:
        print("  Examples missing attribute:")
        for name in missing_examples:
            print("   ", name)

print("\nCSV DATA VALIDATION")
print("=" * 45)

with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))

split_docs = defaultdict(set)
empty_passages = 0
label_counts = Counter()
split_counts = Counter()

for row in rows:
    split = row["split"]
    split_counts[split] += 1
    label_counts[row["is_plagiarized"]] += 1

    split_docs[split].add(row["submitted_doc_id"])
    split_docs[split].add(row["source_doc_id"])

    if not row["submitted_passage"].strip() or not row["source_passage"].strip():
        empty_passages += 1

print("Total rows:", len(rows))
print("Labels:", dict(label_counts))
print("Empty passages:", empty_passages)
print("Split counts:", dict(split_counts))

splits = list(split_docs)
leakage = False

for i in range(len(splits)):
    for j in range(i + 1, len(splits)):
        overlap = split_docs[splits[i]] & split_docs[splits[j]]
        print(f"{splits[i]}-{splits[j]} shared documents:", len(overlap))
        if overlap:
            leakage = True

print("Document leakage:", "FOUND" if leakage else "NOT FOUND")
print("\nAudit completed.")
