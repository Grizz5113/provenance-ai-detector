from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.models.predict import ProvenanceDetector


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "metadata.csv"
)

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "inference_evaluation.csv"
)


def find_essay_file(essay_id: str) -> Path | None:

    for label in (
        "human",
        "ai",
        "hybrid",
    ):
        path = (
            RAW_DIR
            / label
            / f"{essay_id}.txt"
        )

        if path.exists():
            return path

    return None


def main() -> None:

    print("=" * 70)
    print("PROVENANCE — INFERENCE SANITY EVALUATION")
    print("=" * 70)

    metadata = pd.read_csv(
        METADATA_FILE
    )

    print(
        f"Dataset samples: {len(metadata)}"
    )
    print()

    detector = ProvenanceDetector()

    results = []

    for index, row in metadata.iterrows():

        essay_id = row["essay_id"]
        actual = row["label"]

        essay_file = find_essay_file(
            essay_id
        )

        if essay_file is None:
            print(
                f"SKIPPED: {essay_id} "
                "essay file missing"
            )
            continue

        text = essay_file.read_text(
            encoding="utf-8"
        )

        result = detector.predict(
            text
        )

        prediction = result[
            "prediction"
        ]

        probabilities = result[
            "probabilities"
        ]

        correct = (
            actual == prediction
        )

        results.append(
            {
                "essay_id": essay_id,
                "actual": actual,
                "predicted": prediction,
                "correct": correct,
                "ai_probability": probabilities.get(
                    "ai",
                    0.0,
                ),
                "human_probability": probabilities.get(
                    "human",
                    0.0,
                ),
                "hybrid_probability": probabilities.get(
                    "hybrid",
                    0.0,
                ),
                "ai_intervention_level": row.get(
                    "ai_intervention_level",
                    "",
                ),
                "source_group": row.get(
                    "source_group",
                    "",
                ),
            }
        )

        status = "✓" if correct else "✗"

        print(
            f"[{index + 1}/{len(metadata)}] "
            f"{status} "
            f"{essay_id:25s} "
            f"actual={actual:6s} "
            f"predicted={prediction:6s}"
        )

    result_df = pd.DataFrame(
        results
    )

    if result_df.empty:
        raise RuntimeError(
            "No inference results were generated."
        )

    accuracy = (
        result_df["correct"].mean()
    )

    print()
    print("=" * 70)
    print("INFERENCE RESULTS")
    print("=" * 70)

    print(
        f"Samples evaluated: "
        f"{len(result_df)}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.4f}"
    )

    print()

    print("Predictions by actual label:")

    print(
        pd.crosstab(
            result_df["actual"],
            result_df["predicted"],
        ).to_string()
    )

    print()
    print("Misclassified essays:")

    errors = result_df[
        ~result_df["correct"]
    ]

    if errors.empty:
        print("None")
    else:
        print(
            errors[
                [
                    "essay_id",
                    "actual",
                    "predicted",
                    "ai_probability",
                    "human_probability",
                    "hybrid_probability",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

    print()
    print("Accuracy by label:")

    for label in (
        "human",
        "ai",
        "hybrid",
    ):

        subset = result_df[
            result_df["actual"] == label
        ]

        if subset.empty:
            continue

        label_accuracy = (
            subset["correct"].mean()
        )

        print(
            f"{label:8s}: "
            f"{label_accuracy:.4f} "
            f"({subset['correct'].sum()}/"
            f"{len(subset)})"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Results saved to: {OUTPUT_FILE}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
