from __future__ import annotations

import csv
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

HC3_DIR = (
    PROJECT_ROOT
    / "data"
    / "external_training"
    / "hc3"
)

PERSUADE_DIR = (
    PROJECT_ROOT
    / "data"
    / "external_training"
    / "persuade"
)

PERSUADE_HUMAN_DIR = (
    PERSUADE_DIR / "human"
)

PERSUADE_HYBRID_DIR = (
    PERSUADE_DIR / "hybrid"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw_augmented"
)

OUTPUT_METADATA = (
    OUTPUT_DIR / "metadata.csv"
)

LABELS = (
    "human",
    "ai",
    "hybrid",
)


def load_csv(
    path: Path,
) -> list[dict[str, str]]:

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def validate_file(
    path: Path,
) -> None:

    if not path.exists():

        raise RuntimeError(
            f"Missing file: {path}"
        )


def main() -> None:

    print("=" * 70)
    print(
        "PROVENANCE — BUILD FULL AUGMENTED "
        "TRAINING DATASET"
    )
    print("=" * 70)

    # ------------------------------------------------------------
    # Metadata files
    # ------------------------------------------------------------

    original_metadata_file = (
        RAW_DIR / "metadata.csv"
    )

    hc3_metadata_file = (
        HC3_DIR / "metadata.csv"
    )

    persuade_metadata_file = (
        PERSUADE_DIR / "metadata.csv"
    )

    validate_file(
        original_metadata_file
    )

    validate_file(
        hc3_metadata_file
    )

    validate_file(
        persuade_metadata_file
    )

    # ------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------

    original_rows = load_csv(
        original_metadata_file
    )

    hc3_rows = load_csv(
        hc3_metadata_file
    )

    persuade_rows = load_csv(
        persuade_metadata_file
    )

    print(
        f"Original samples:   "
        f"{len(original_rows)}"
    )

    print(
        f"HC3 samples:         "
        f"{len(hc3_rows)}"
    )

    print(
        f"PERSUADE humans:     "
        f"{len(persuade_rows)}"
    )

    # ------------------------------------------------------------
    # Discover PERSUADE hybrid provenance files
    # ------------------------------------------------------------

    persuade_hybrid_json = sorted(
        PERSUADE_HYBRID_DIR.glob(
            "*.json"
        )
    )

    persuade_hybrid_txt = sorted(
        PERSUADE_HYBRID_DIR.glob(
            "*.txt"
        )
    )

    print(
        f"PERSUADE hybrids:    "
        f"{len(persuade_hybrid_txt)}"
    )

    print(
        f"PERSUADE provenance: "
        f"{len(persuade_hybrid_json)}"
    )

    if len(persuade_hybrid_txt) != 120:

        raise RuntimeError(
            "Expected 120 PERSUADE hybrid "
            f"TXT files, found "
            f"{len(persuade_hybrid_txt)}"
        )

    if len(persuade_hybrid_json) != 120:

        raise RuntimeError(
            "Expected 120 PERSUADE hybrid "
            f"JSON files, found "
            f"{len(persuade_hybrid_json)}"
        )

    # ------------------------------------------------------------
    # Validate original files
    # ------------------------------------------------------------

    for row in original_rows:

        essay_id = row["essay_id"]
        label = row["label"]

        source_file = (
            RAW_DIR
            / label
            / f"{essay_id}.txt"
        )

        validate_file(source_file)

    # ------------------------------------------------------------
    # Validate HC3 files
    # ------------------------------------------------------------

    for row in hc3_rows:

        essay_id = row["essay_id"]
        label = row["label"]

        source_file = (
            HC3_DIR
            / label
            / f"{essay_id}.txt"
        )

        validate_file(source_file)

    # ------------------------------------------------------------
    # Validate PERSUADE human files
    # ------------------------------------------------------------

    for row in persuade_rows:

        essay_id = row["essay_id"]

        source_file = (
            PERSUADE_HUMAN_DIR
            / f"{essay_id}.txt"
        )

        validate_file(source_file)

    # ------------------------------------------------------------
    # Validate PERSUADE hybrid files
    # ------------------------------------------------------------

    hybrid_provenance = []

    for json_file in persuade_hybrid_json:

        with json_file.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = __import__(
                "json"
            ).load(f)

        essay_id = data["essay_id"]

        txt_file = (
            PERSUADE_HYBRID_DIR
            / f"{essay_id}.txt"
        )

        validate_file(txt_file)

        hybrid_provenance.append(
            data
        )

    # ------------------------------------------------------------
    # Validate source human references
    # ------------------------------------------------------------

    persuade_human_ids = {
        row["essay_id"]
        for row in persuade_rows
    }

    for data in hybrid_provenance:

        source_human = data[
            "source_human_essay"
        ]

        if (
            source_human
            not in persuade_human_ids
        ):

            raise RuntimeError(
                "Hybrid references unknown "
                f"human essay: {source_human}"
            )

    # ------------------------------------------------------------
    # Prevent ID collisions
    # ------------------------------------------------------------

    all_existing_ids = (
        {
            row["essay_id"]
            for row in original_rows
        }
        |
        {
            row["essay_id"]
            for row in hc3_rows
        }
        |
        persuade_human_ids
    )

    hybrid_ids = {
        data["essay_id"]
        for data in hybrid_provenance
    }

    collisions = (
        all_existing_ids
        & hybrid_ids
    )

    if collisions:

        raise RuntimeError(
            "Essay ID collisions detected: "
            + ", ".join(
                sorted(collisions)
            )
        )

    # ------------------------------------------------------------
    # Recreate output dataset
    # ------------------------------------------------------------

    if OUTPUT_DIR.exists():

        shutil.rmtree(
            OUTPUT_DIR
        )

    for label in LABELS:

        (
            OUTPUT_DIR / label
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------
    # Metadata schema
    # ------------------------------------------------------------

    fieldnames = [
        "essay_id",
        "label",
        "source",
        "topic",
        "generation_model",
        "generation_prompt",
        "human_editing",
        "ai_intervention_level",
        "source_group",
        "split",
        "author_language_background",
        "notes",
    ]

    merged_rows = []

    # ------------------------------------------------------------
    # Original data
    # ------------------------------------------------------------

    for row in original_rows:

        essay_id = row["essay_id"]
        label = row["label"]

        source = (
            RAW_DIR
            / label
            / f"{essay_id}.txt"
        )

        destination = (
            OUTPUT_DIR
            / label
            / f"{essay_id}.txt"
        )

        shutil.copy2(
            source,
            destination,
        )

        merged_rows.append(
            {
                field: row.get(
                    field,
                    "",
                )
                for field in fieldnames
            }
        )

    # ------------------------------------------------------------
    # HC3 data
    # ------------------------------------------------------------

    for row in hc3_rows:

        essay_id = row["essay_id"]
        label = row["label"]

        source = (
            HC3_DIR
            / label
            / f"{essay_id}.txt"
        )

        destination = (
            OUTPUT_DIR
            / label
            / f"{essay_id}.txt"
        )

        shutil.copy2(
            source,
            destination,
        )

        merged_rows.append(
            {
                "essay_id": essay_id,
                "label": label,
                "source": "HC3",
                "topic": row.get(
                    "topic",
                    "",
                ),
                "generation_model": (
                    "ChatGPT"
                    if label == "ai"
                    else ""
                ),
                "generation_prompt": "",
                "human_editing": "",
                "ai_intervention_level": row.get(
                    "ai_intervention_level",
                    "",
                ),
                "source_group": row[
                    "source_group"
                ],
                "split": "train",
                "author_language_background": (
                    "unknown"
                ),
                "notes": (
                    "External HC3 augmentation; "
                    f"pair_id="
                    f"{row.get('pair_id', '')}; "
                    f"word_count="
                    f"{row.get('word_count', '')}"
                ),
            }
        )

    # ------------------------------------------------------------
    # PERSUADE human data
    # ------------------------------------------------------------

    for row in persuade_rows:

        essay_id = row["essay_id"]

        source = (
            PERSUADE_HUMAN_DIR
            / f"{essay_id}.txt"
        )

        destination = (
            OUTPUT_DIR
            / "human"
            / f"{essay_id}.txt"
        )

        shutil.copy2(
            source,
            destination,
        )

        merged_rows.append(
            {
                "essay_id": essay_id,
                "label": "human",
                "source": "PERSUADE_2.0",
                "topic": row.get(
                    "topic",
                    "",
                ),
                "generation_model": "",
                "generation_prompt": "",
                "human_editing": "",
                "ai_intervention_level": "none",
                "source_group": row.get(
                    "source_group",
                    "",
                ),
                "split": "train",
                "author_language_background": (
                    "unknown"
                ),
                "notes": (
                    "External PERSUADE human "
                    "training sample; "
                    f"persuade_id="
                    f"{row.get('persuade_id', '')}"
                ),
            }
        )

    # ------------------------------------------------------------
    # PERSUADE hybrid data
    # ------------------------------------------------------------

    for data in hybrid_provenance:

        essay_id = data[
            "essay_id"
        ]

        source_human = data[
            "source_human_essay"
        ]

        level = data[
            "ai_intervention_level"
        ]

        source = (
            PERSUADE_HYBRID_DIR
            / f"{essay_id}.txt"
        )

        destination = (
            OUTPUT_DIR
            / "hybrid"
            / f"{essay_id}.txt"
        )

        shutil.copy2(
            source,
            destination,
        )

        # All three intervention levels
        # derived from the same human essay
        # share one source group.
        source_group = (
            f"persuade_hybrid_"
            f"{source_human}"
        )

        merged_rows.append(
            {
                "essay_id": essay_id,
                "label": "hybrid",
                "source": "PERSUADE_2.0",
                "topic": "",
                "generation_model": data.get(
                    "model",
                    "",
                ),
                "generation_prompt": data.get(
                    "prompt",
                    "",
                ),
                "human_editing": "",
                "ai_intervention_level": level,
                "source_group": source_group,
                "split": "train",
                "author_language_background": (
                    "unknown"
                ),
                "notes": (
                    "PERSUADE human essay "
                    "AI-edited with local LLM; "
                    f"source_human="
                    f"{source_human}; "
                    f"edit_ratio="
                    f"{data.get('edit_ratio', '')}"
                ),
            }
        )

    # ------------------------------------------------------------
    # Validate final counts
    # ------------------------------------------------------------

    expected_total = (
        len(original_rows)
        + len(hc3_rows)
        + len(persuade_rows)
        + len(hybrid_provenance)
    )

    if len(merged_rows) != expected_total:

        raise RuntimeError(
            "Merged metadata count mismatch: "
            f"{len(merged_rows)} != "
            f"{expected_total}"
        )

    # ------------------------------------------------------------
    # Write metadata
    # ------------------------------------------------------------

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
        writer.writerows(
            merged_rows
        )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print()
    print("-" * 70)
    print("FULL AUGMENTED DATASET CREATED")
    print("-" * 70)

    print(
        f"Original:           "
        f"{len(original_rows)}"
    )

    print(
        f"HC3:                "
        f"{len(hc3_rows)}"
    )

    print(
        f"PERSUADE human:     "
        f"{len(persuade_rows)}"
    )

    print(
        f"PERSUADE hybrid:    "
        f"{len(hybrid_provenance)}"
    )

    print(
        f"Total:              "
        f"{len(merged_rows)}"
    )

    print()
    print("Class distribution:")

    counts = {}

    for row in merged_rows:

        label = row["label"]

        counts[label] = (
            counts.get(
                label,
                0,
            )
            + 1
        )

    for label in LABELS:

        print(
            f"  {label:8s}: "
            f"{counts.get(label, 0)}"
        )

    print()
    print(
        "PERSUADE hybrid levels:"
    )

    level_counts = {}

    for data in hybrid_provenance:

        level = data[
            "ai_intervention_level"
        ]

        level_counts[level] = (
            level_counts.get(
                level,
                0,
            )
            + 1
        )

    for level in (
        "light",
        "moderate",
        "heavy",
    ):

        print(
            f"  {level:8s}: "
            f"{level_counts.get(level, 0)}"
        )

    print()
    print(
        f"Dataset:  {OUTPUT_DIR}"
    )

    print(
        f"Metadata: {OUTPUT_METADATA}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()