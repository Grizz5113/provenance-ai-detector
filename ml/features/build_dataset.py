from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.detector.language_model import LanguageModelAnalyzer
from backend.app.features.extractor import EssayFeatureExtractor


RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_FILE = RAW_DIR / "metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "features.csv"


def load_metadata() -> dict[str, dict[str, str]]:
    with METADATA_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        return {
            row["essay_id"]: row
            for row in reader
            if row.get("essay_id")
        }


def find_essay_file(essay_id: str) -> Path | None:
    for label in ("human", "ai", "hybrid"):
        path = RAW_DIR / label / f"{essay_id}.txt"

        if path.exists():
            return path

    return None


def main() -> None:
    print("=" * 70)
    print("PROVENANCE — BUILD FEATURE DATASET")
    print("=" * 70)

    metadata = load_metadata()

    print(f"Metadata records: {len(metadata)}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Loading language model...")

    language_model = LanguageModelAnalyzer()

    extractor = EssayFeatureExtractor(
        language_model=language_model,
    )

    feature_rows = []

    for index, (essay_id, meta) in enumerate(
        sorted(metadata.items()),
        start=1,
    ):
        essay_file = find_essay_file(essay_id)

        if essay_file is None:
            print(
                f"SKIPPED: {essay_id} "
                "has no essay file"
            )
            continue

        text = essay_file.read_text(
            encoding="utf-8",
        )

        print(
            f"[{index}/{len(metadata)}] "
            f"Extracting {essay_id}..."
        )

        features = extractor.extract(text)

        row = {
            "essay_id": essay_id,
            "label": meta.get("label", ""),
            "topic": meta.get("topic", ""),
            "source": meta.get("source", ""),
            "source_group": meta.get(
                "source_group",
                "",
            ),
            **features.to_dict(),
        }

        # Preserve hybrid provenance information
        if meta.get("label") == "hybrid":
            notes = meta.get("notes", "")

            try:
                parsed_notes = json.loads(notes)
            except json.JSONDecodeError:
                parsed_notes = {}

            row["edit_ratio"] = parsed_notes.get(
                "edit_ratio",
            )
        else:
            row["edit_ratio"] = ""

        feature_rows.append(row)

    if not feature_rows:
        raise RuntimeError(
            "No feature rows were generated."
        )

    fieldnames = list(feature_rows[0].keys())

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(feature_rows)

    print()
    print("-" * 70)
    print(
        f"Feature rows written: {len(feature_rows)}"
    )
    print(
        f"Feature columns:      {len(fieldnames)}"
    )
    print(
        f"Saved to:             {OUTPUT_FILE}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()