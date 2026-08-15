from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EVALUATION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)


def get_benchmark_paths():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark",
        default=None,
        help=(
            "Evaluation benchmark directory "
            "inside data/evaluation/"
        ),
    )

    args = parser.parse_args()

    if args.benchmark:
        evaluation_dir = (
            EVALUATION_ROOT
            / args.benchmark
        )
    else:
        evaluation_dir = EVALUATION_ROOT

    metadata_file = (
        evaluation_dir
        / "metadata.csv"
    )

    return evaluation_dir, metadata_file


VALID_LABELS = {
    "human",
    "ai",
    "hybrid",
}


def main():

    print("=" * 70)
    print("PROVENANCE — EXTERNAL DATASET VALIDATION")
    print("=" * 70)
    evaluation_dir, metadata_file = (
        get_benchmark_paths()
    )
    if not metadata_file.exists():
        raise RuntimeError(
            f"Missing metadata file: {metadata_file}"
        )

    with metadata_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    if not rows:
        raise RuntimeError(
            "External metadata is empty."
        )

    ids = set()

    counts = {
        "human": 0,
        "ai": 0,
        "hybrid": 0,
    }

    for row in rows:

        essay_id = row.get(
            "essay_id",
            "",
        ).strip()

        label = row.get(
            "label",
            "",
        ).strip()

        if not essay_id:
            raise RuntimeError(
                "Found record without essay_id."
            )

        if essay_id in ids:
            raise RuntimeError(
                f"Duplicate essay_id: {essay_id}"
            )

        ids.add(essay_id)

        if label not in VALID_LABELS:
            raise RuntimeError(
                f"{essay_id}: invalid label '{label}'."
            )

        counts[label] += 1

        expected_file = (
            evaluation_dir
            / label
            / f"{essay_id}.txt"
        )

        if not expected_file.exists():
            raise RuntimeError(
                f"Missing file: {expected_file}"
            )

        text = expected_file.read_text(
            encoding="utf-8"
        )

        if len(text.strip()) < 20:
            raise RuntimeError(
                f"{essay_id}: text is too short."
            )

    print()
    print(
        f"Records: {len(rows)}"
    )

    print()
    print("Class distribution:")

    for label, count in counts.items():
        print(
            f"  {label:7s}: {count}"
        )

    print()
    print("VALIDATION PASSED")
    print("-" * 70)
    print(
        "All metadata records have matching "
        "essay files and valid labels."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
