from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TRAINING_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)

METADATA_FILE = (
    EVALUATION_DIR
    / "metadata.csv"
)


def load_external_metadata():
    with METADATA_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def main():

    print("=" * 70)
    print("PROVENANCE — TRAINING vs EXTERNAL DISTRIBUTION")
    print("=" * 70)

    training = pd.read_csv(
        TRAINING_FEATURES
    )

    metadata = load_external_metadata()

    external_rows = []

    for row in metadata:

        essay_id = row["essay_id"]
        label = row["label"]

        path = (
            EVALUATION_DIR
            / label
            / f"{essay_id}.txt"
        )

        text = path.read_text(
            encoding="utf-8"
        )

        external_rows.append(
            {
                "essay_id": essay_id,
                "label": label,
                "word_count": len(
                    text.split()
                ),
                "character_count": len(text),
            }
        )

    external = pd.DataFrame(
        external_rows
    )

    print()
    print("TRAINING DATASET")
    print("-" * 70)

    print(
        training.groupby("label")[
            "token_count"
        ].agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        ).round(2)
    )

    print()
    print("EXTERNAL DATASET")
    print("-" * 70)

    print(
        external.groupby("label")[
            "word_count"
        ].agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        ).round(2)
    )

    print()
    print("EXTERNAL SAMPLES")
    print("-" * 70)

    print(
        external.to_string(
            index=False
        )
    )

    print()
    print("TRAINING CLASS DISTRIBUTION")
    print("-" * 70)

    print(
        training["label"].value_counts()
    )

    print()
    print("EXTERNAL CLASS DISTRIBUTION")
    print("-" * 70)

    print(
        external["label"].value_counts()
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
