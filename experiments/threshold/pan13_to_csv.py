
import csv
import re
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter
from statistics import median

ROOT = Path(r"D:\pan13-text-alignment-test-corpus2-2013-01-21")
ANNOTATIONS = ROOT / "pan13-text-alignment-test-corpus2-2013-01-21"
SRC = ROOT / "src"
SUSP = ROOT / "susp"
OUTPUT = Path(r"D:\meeting_preparation\experiments\threshold\threshold_dataset.csv")

FOLDER_TYPES = {
    "02-no-obfuscation": "none",
    "03-random-obfuscation": "random",
    "04-translation-obfuscation": "translation",
    "05-summary-obfuscation": "summary"
}
NEGATIVE_FOLDER = "01-no-plagiarism"
RANDOM_SEED = 42


def read_document(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return file.read()


def get_document_id(filename):
    match = re.search(r"(?:source|suspicious)-document\d+", filename)
    return match.group(0) if match else None


def get_source_id(filename):
    match = re.search(r"(source-document\d+)", filename)
    return match.group(1) if match else None


def add_pair(rows, suspicious_name, source_name, submitted, source,
             label, obfuscation, pair_type):
    if not submitted.strip() or not source.strip():
        return False

    submitted_id = get_document_id(suspicious_name)
    source_id = get_document_id(source_name)

    if not submitted_id or not source_id:
        return False

    rows.append({
        "pair_id": "",
        "source_doc_id": source_id,
        "submitted_doc_id": submitted_id,
        "submitted_passage": submitted.strip(),
        "source_passage": source.strip(),
        "is_plagiarized": label,
        "split": "",
        "obfuscation_type": obfuscation,
        "pair_type": pair_type
    })
    return True


def build_positive_pairs():
    rows = []
    skipped = 0
    lengths = []

    for folder, folder_type in FOLDER_TYPES.items():
        folder_path = ANNOTATIONS / folder

        if not folder_path.exists():
            print("Missing positive folder:", folder_path)
            continue

        for xml_file in folder_path.glob("*.xml"):
            try:
                root = ET.parse(xml_file).getroot()
                suspicious_name = root.attrib.get("reference")

                if not suspicious_name:
                    skipped += 1
                    continue

                suspicious_path = SUSP / suspicious_name

                if not suspicious_path.exists():
                    skipped += 1
                    continue

                suspicious_text = read_document(suspicious_path)
                features = root.findall(".//feature[@name='plagiarism']")

                for feature in features:
                    source_name = feature.attrib.get("source_reference")

                    if not source_name:
                        skipped += 1
                        continue

                    source_path = SRC / source_name

                    if not source_path.exists():
                        skipped += 1
                        continue

                    try:
                        submitted_offset = int(feature.attrib["this_offset"])
                        submitted_length = int(feature.attrib["this_length"])
                        source_offset = int(feature.attrib["source_offset"])
                        source_length = int(feature.attrib["source_length"])
                    except (KeyError, ValueError):
                        skipped += 1
                        continue

                    submitted = suspicious_text[
                        submitted_offset:submitted_offset + submitted_length
                    ]
                    source_text = read_document(source_path)
                    source = source_text[
                        source_offset:source_offset + source_length
                    ]

                    obfuscation = feature.attrib.get("obfuscation") or folder_type

                    if add_pair(
                        rows, suspicious_name, source_name,
                        submitted, source, 1, obfuscation, "positive"
                    ):
                        lengths.append((len(submitted), len(source)))
                    else:
                        skipped += 1

            except (ET.ParseError, OSError, ValueError) as error:
                print("Skipped positive annotation:", xml_file.name, error)
                skipped += 1

    return rows, skipped, lengths


def build_negative_pairs(target_length):
    rows = []
    skipped = 0
    folder_path = ANNOTATIONS / NEGATIVE_FOLDER

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Negative annotation folder not found: {folder_path}"
        )

    for xml_file in folder_path.glob("*.xml"):
        try:
            root = ET.parse(xml_file).getroot()
            suspicious_name = root.attrib.get("reference")
            source_id = get_source_id(xml_file.name)

            if not suspicious_name or not source_id:
                skipped += 1
                continue

            source_name = source_id + ".txt"
            suspicious_path = SUSP / suspicious_name
            source_path = SRC / source_name

            if not suspicious_path.exists() or not source_path.exists():
                skipped += 1
                continue

            suspicious_text = read_document(suspicious_path)
            source_text = read_document(source_path)

            if (len(suspicious_text.strip()) < target_length or
                    len(source_text.strip()) < target_length):
                skipped += 1
                continue

            suspicious_start = max(
                0, (len(suspicious_text) - target_length) // 2
            )
            source_start = max(
                0, (len(source_text) - target_length) // 2
            )

            submitted = suspicious_text[
                suspicious_start:suspicious_start + target_length
            ]
            source = source_text[
                source_start:source_start + target_length
            ]

            if not add_pair(
                rows, suspicious_name, source_name,
                submitted, source, 0, "none", "negative"
            ):
                skipped += 1

        except (ET.ParseError, OSError, ValueError) as error:
            print("Skipped negative annotation:", xml_file.name, error)
            skipped += 1

    return rows, skipped


class DisjointSet:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a


def assign_splits(rows):
    dsu = DisjointSet()

    for row in rows:
        dsu.union(row["submitted_doc_id"], row["source_doc_id"])

    components = defaultdict(list)

    for index, row in enumerate(rows):
        root = dsu.find(row["submitted_doc_id"])
        components[root].append(index)

    component_list = list(components.values())
    random.Random(RANDOM_SEED).shuffle(component_list)
    component_list.sort(key=len, reverse=True)

    targets = {
        "train": len(rows) * 0.70,
        "validation": len(rows) * 0.15,
        "test": len(rows) * 0.15
    }

    counts = Counter()
    split_order = ["train", "validation", "test"]

    for component in component_list:
        split = min(
            split_order,
            key=lambda name: counts[name] / max(targets[name], 1)
        )

        for index in component:
            rows[index]["split"] = split
            counts[split] += 1

    return len(component_list)


def main():
    print("Reading PAN 2013 annotations...")

    positive_rows, positive_skipped, lengths = build_positive_pairs()

    if not positive_rows:
        raise RuntimeError("No positive pairs extracted.")

    target_length = max(
        1, int(median((a + b) // 2 for a, b in lengths))
    )

    negative_rows, negative_skipped = build_negative_pairs(target_length)

    if not negative_rows:
        raise RuntimeError("No negative pairs extracted.")

    rows = positive_rows + negative_rows
    random.Random(RANDOM_SEED).shuffle(rows)

    # Assign globally unique pair IDs after combining both classes.
    for index, row in enumerate(rows, 1):
        row["pair_id"] = f"pair_{index:05d}"

    component_count = assign_splits(rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "pair_id", "source_doc_id", "submitted_doc_id",
        "submitted_passage", "source_passage",
        "is_plagiarized", "split", "obfuscation_type", "pair_type"
    ]

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\nPAN 2013 CONVERSION SUMMARY")
    print("=" * 45)
    print("Positive pairs:", len(positive_rows))
    print("Negative pairs:", len(negative_rows))
    print("Total pairs:", len(rows))
    print("Skipped positive annotations:", positive_skipped)
    print("Skipped negative annotations:", negative_skipped)
    print("Connected document groups:", component_count)

    print("\nSplit distribution:")

    for split in ["train", "validation", "test"]:
        split_rows = [r for r in rows if r["split"] == split]
        positives = sum(r["is_plagiarized"] == 1 for r in split_rows)
        negatives = sum(r["is_plagiarized"] == 0 for r in split_rows)

        print(
            f"{split}: {len(split_rows)} pairs | "
            f"Positive: {positives} | Negative: {negatives}"
        )

        if positives == 0 or negatives == 0:
            print("WARNING: This split does not contain both classes.")

    print("\nObfuscation distribution:")
    for name, count in Counter(
        r["obfuscation_type"] for r in positive_rows
    ).items():
        print(f"{name}: {count}")

    pair_ids = [row["pair_id"] for row in rows]
    print("\nUnique pair IDs:", len(set(pair_ids)) == len(pair_ids))

    print("\nCSV saved to:", OUTPUT)
    print("Note: Preliminary pairwise dataset, not official PAN PlagDet evaluation.")


if __name__ == "__main__":
    main()
