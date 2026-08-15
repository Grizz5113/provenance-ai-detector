from __future__ import annotations

import csv
import random
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIP_FILE = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "persuade"
    / "persaude-corpus-2.zip"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "external_training"
    / "persuade"
)

OUTPUT_METADATA = OUTPUT_DIR / "metadata.csv"

SEED = 42
TARGET_ESSAYS = 40

MIN_WORDS = 100
MAX_WORDS = 500

CSV_NAME = (
    "persuade_2.0_human_scores_demo_id_github.csv"
)


# Existing PERSUADE essays already present
# in data/raw/metadata.csv.
EXISTING_IDS = {
    "B7468C4DD17C",
    "B2C131C2B3AB",
    "74D0BA756BD1",
    "AD97126ECA8A",
    "FEB736959AD7",
    "31F25A58675C",
    "36F90F86ED01",
    "CF165F9553A4",
    "4B18A3D32AA3",
    "4B759A5F8294",
}


def normalize_id(value: str) -> str:
    return str(value).strip()


def main() -> None:

    print("=" * 70)
    print("PROVENANCE — SELECT PERSUADE HUMAN ESSAYS")
    print("=" * 70)

    if not ZIP_FILE.exists():
        raise RuntimeError(
            f"PERSUADE ZIP not found: {ZIP_FILE}"
        )

    print(f"Source: {ZIP_FILE}")
    print(f"Existing essays excluded: {len(EXISTING_IDS)}")

    candidates = []

    with zipfile.ZipFile(ZIP_FILE) as archive:

        with archive.open(CSV_NAME) as raw:

            # Decode CSV stream without extracting
            # the 75 MB file to disk.
            import io

            text = io.TextIOWrapper(
                raw,
                encoding="utf-8",
                newline="",
            )

            reader = csv.DictReader(text)

            for row in reader:

                essay_id = normalize_id(
                    row.get("essay_id_comp", "")
                )

                if not essay_id:
                    continue

                if essay_id in EXISTING_IDS:
                    continue

                text_value = (
                    row.get("full_text") or ""
                ).strip()

                word_count = int(
                    row.get("word_count") or 0
                )

                prompt = (
                    row.get("prompt_name") or ""
                ).strip()

                task = (
                    row.get("task") or ""
                ).strip()

                if not text_value:
                    continue

                if not (
                    MIN_WORDS
                    <= word_count
                    <= MAX_WORDS
                ):
                    continue

                if not prompt:
                    continue

                candidates.append(
                    {
                        "essay_id": essay_id,
                        "text": text_value,
                        "word_count": word_count,
                        "prompt_name": prompt,
                        "task": task,
                    }
                )

    print(
        f"Eligible candidates: {len(candidates)}"
    )

    if len(candidates) < TARGET_ESSAYS:
        raise RuntimeError(
            "Not enough eligible PERSUADE essays."
        )

    # ------------------------------------------------------------
    # Prefer prompt diversity.
    # ------------------------------------------------------------

    random.seed(SEED)
    random.shuffle(candidates)

    selected = []
    prompt_counts = {}

    # First pass: maximum 4 essays per prompt.
    for row in candidates:

        prompt = row["prompt_name"]

        if prompt_counts.get(prompt, 0) >= 4:
            continue

        selected.append(row)

        prompt_counts[prompt] = (
            prompt_counts.get(prompt, 0) + 1
        )

        if len(selected) >= TARGET_ESSAYS:
            break

    # Fallback if necessary.
    if len(selected) < TARGET_ESSAYS:

        selected_ids = {
            row["essay_id"]
            for row in selected
        }

        for row in candidates:

            if row["essay_id"] in selected_ids:
                continue

            selected.append(row)

            if len(selected) >= TARGET_ESSAYS:
                break

    if len(selected) != TARGET_ESSAYS:
        raise RuntimeError(
            "Could not select requested essays."
        )

    # ------------------------------------------------------------
    # Write selected essays.
    # ------------------------------------------------------------

    human_dir = OUTPUT_DIR / "human"
    human_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_rows = []

    for index, row in enumerate(
        selected,
        start=1,
    ):

        essay_id = (
            f"persuade_train_human_{index:03d}"
        )

        path = (
            human_dir
            / f"{essay_id}.txt"
        )

        path.write_text(
            row["text"],
            encoding="utf-8",
        )

        metadata_rows.append(
            {
                "essay_id": essay_id,
                "persuade_id": row["essay_id"],
                "label": "human",
                "topic": row["prompt_name"],
                "source": "PERSUADE_2.0",
                "source_group": (
                    f"persuade_new_"
                    f"{row['essay_id']}"
                ),
                "ai_intervention_level": "none",
                "word_count": row["word_count"],
                "notes": (
                    "Independent human essay "
                    "selected from PERSUADE 2.0"
                ),
            }
        )

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    fieldnames = [
        "essay_id",
        "persuade_id",
        "label",
        "topic",
        "source",
        "source_group",
        "ai_intervention_level",
        "word_count",
        "notes",
    ]

    with OUTPUT_METADATA.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    print()
    print("-" * 70)
    print("PERSUADE HUMAN SELECTION COMPLETE")
    print("-" * 70)

    print(f"Selected essays: {len(selected)}")
    print(
        f"Unique prompts:  "
        f"{len(prompt_counts)}"
    )

    print()
    print("Prompt distribution:")

    for prompt, count in sorted(
        prompt_counts.items()
    ):
        print(
            f"  {prompt:45s}: {count}"
        )

    print()
    print(f"Essays:   {human_dir}")
    print(f"Metadata: {OUTPUT_METADATA}")
    print("=" * 70)


if __name__ == "__main__":
    main()