from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_FILE = RAW_DIR / "metadata.csv"

VALID_LABELS = {"human", "ai", "hybrid"}

VALID_INTERVENTION_LEVELS = {
    "none",
    "light",
    "moderate",
    "heavy",
    "full",
}

VALID_SPLITS = {
    "",
    "train",
    "validation",
    "test",
}

REQUIRED_COLUMNS = {
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
}


def validate_metadata() -> list[str]:
    errors: list[str] = []

    if not METADATA_FILE.exists():
        return [
            f"Metadata file does not exist: {METADATA_FILE}"
        ]

    with METADATA_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            return ["metadata.csv has no header."]

        actual_columns = set(reader.fieldnames)

        missing_columns = REQUIRED_COLUMNS - actual_columns

        if missing_columns:
            errors.append(
                "Missing columns: "
                + ", ".join(sorted(missing_columns))
            )

            return errors

        rows = list(reader)

    print(f"Metadata rows found: {len(rows)}")

    if not rows:
        print(
            "No dataset records yet. "
            "Metadata structure is valid."
        )
        return errors

    essay_ids: set[str] = set()
    source_groups: dict[str, list[dict[str, str]]] = {}

    for row_number, row in enumerate(rows, start=2):

        essay_id = row["essay_id"].strip()
        label = row["label"].strip().lower()
        source_group = row["source_group"].strip()
        split = row["split"].strip().lower()
        intervention = (
            row["ai_intervention_level"]
            .strip()
            .lower()
        )

        # ---------------------------------------------
        # Essay ID
        # ---------------------------------------------

        if not essay_id:
            errors.append(
                f"Row {row_number}: essay_id is empty."
            )

        elif essay_id in essay_ids:
            errors.append(
                f"Row {row_number}: duplicate essay_id "
                f"'{essay_id}'."
            )

        else:
            essay_ids.add(essay_id)

        # ---------------------------------------------
        # Label
        # ---------------------------------------------

        if label not in VALID_LABELS:
            errors.append(
                f"Row {row_number}: invalid label "
                f"'{label}'."
            )

        # ---------------------------------------------
        # Source group
        # ---------------------------------------------

        if not source_group:
            errors.append(
                f"Row {row_number}: source_group is empty."
            )
        else:
            source_groups.setdefault(
                source_group,
                [],
            ).append(row)

        # ---------------------------------------------
        # Split
        # ---------------------------------------------

        if split not in VALID_SPLITS:
            errors.append(
                f"Row {row_number}: invalid split "
                f"'{split}'."
            )

        # ---------------------------------------------
        # AI intervention
        # ---------------------------------------------

        if intervention not in VALID_INTERVENTION_LEVELS:
            errors.append(
                f"Row {row_number}: invalid "
                f"ai_intervention_level "
                f"'{intervention}'."
            )

        # ---------------------------------------------
        # Required metadata
        # ---------------------------------------------

        for field in (
            "source",
            "topic",
        ):
            if not row[field].strip():
                errors.append(
                    f"Row {row_number}: {field} is empty."
                )

    # -------------------------------------------------
    # Check source-group split leakage
    # -------------------------------------------------

    for source_group, group_rows in source_groups.items():

        assigned_splits = {
            row["split"].strip().lower()
            for row in group_rows
            if row["split"].strip()
        }

        if len(assigned_splits) > 1:
            errors.append(
                f"Source group '{source_group}' "
                f"appears in multiple splits: "
                f"{sorted(assigned_splits)}"
            )

    return errors


def main() -> None:

    print("=" * 70)
    print("PROVENANCE — DATASET VALIDATION")
    print("=" * 70)

    errors = validate_metadata()

    print()

    if errors:

        print("VALIDATION FAILED")
        print("-" * 70)

        for error in errors:
            print(f"ERROR: {error}")

        print()
        print(f"Total errors: {len(errors)}")

        raise SystemExit(1)

    print("VALIDATION PASSED")
    print("-" * 70)
    print("Metadata schema and constraints are valid.")

    print("=" * 70)


if __name__ == "__main__":
    main()