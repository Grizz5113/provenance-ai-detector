from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_FILE = RAW_DIR / "metadata.csv"

MIN_WORDS = 50
MAX_WORDS = 5000

VALID_LABELS = {"human", "ai", "hybrid"}


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def text_hash(text: str) -> str:
    normalized = normalize_text(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def count_words(text: str) -> int:
    return len(text.split())


def load_metadata() -> dict[str, dict[str, str]]:
    with METADATA_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        return {
            row["essay_id"].strip(): row
            for row in reader
            if row["essay_id"].strip()
        }


def main() -> None:

    print("=" * 70)
    print("PROVENANCE — ESSAY FILE VALIDATION")
    print("=" * 70)

    if not METADATA_FILE.exists():
        print("ERROR: metadata.csv does not exist.")
        raise SystemExit(1)

    metadata = load_metadata()

    print(f"Metadata records: {len(metadata)}")

    errors: list[str] = []
    warnings: list[str] = []

    discovered_files: dict[str, Path] = {}
    hashes: defaultdict[str, list[str]] = defaultdict(list)

    # --------------------------------------------------
    # Discover essay files
    # --------------------------------------------------

    for label in VALID_LABELS:

        directory = RAW_DIR / label

        if not directory.exists():
            continue

        for path in directory.glob("*.txt"):

            essay_id = path.stem

            if essay_id in discovered_files:
                errors.append(
                    f"Duplicate essay file ID: {essay_id}"
                )

            discovered_files[essay_id] = path

    print(f"Essay files found: {len(discovered_files)}")

    # --------------------------------------------------
    # Check metadata -> file
    # --------------------------------------------------

    for essay_id, row in metadata.items():

        if essay_id not in discovered_files:

            errors.append(
                f"Metadata entry '{essay_id}' "
                f"has no corresponding .txt file."
            )

    # --------------------------------------------------
    # Check file -> metadata
    # --------------------------------------------------

    for essay_id in discovered_files:

        if essay_id not in metadata:

            errors.append(
                f"Essay file '{essay_id}.txt' "
                f"has no metadata entry."
            )

    # --------------------------------------------------
    # Validate individual files
    # --------------------------------------------------

    for essay_id, path in discovered_files.items():

        if essay_id not in metadata:
            continue

        row = metadata[essay_id]

        label = row["label"].strip().lower()

        expected_directory = RAW_DIR / label

        if path.parent != expected_directory:

            errors.append(
                f"{essay_id}: file is in "
                f"'{path.parent.name}/' but metadata label "
                f"is '{label}'."
            )

        if label not in VALID_LABELS:

            errors.append(
                f"{essay_id}: invalid label '{label}'."
            )

        try:
            text = path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError:

            errors.append(
                f"{essay_id}: file is not valid UTF-8."
            )

            continue

        normalized = normalize_text(text)

        if not normalized:

            errors.append(
                f"{essay_id}: essay is empty."
            )

            continue

        word_count = count_words(text)

        if word_count < MIN_WORDS:

            errors.append(
                f"{essay_id}: only {word_count} words "
                f"(minimum {MIN_WORDS})."
            )

        if word_count > MAX_WORDS:

            errors.append(
                f"{essay_id}: {word_count} words "
                f"(maximum {MAX_WORDS})."
            )

        digest = text_hash(text)

        hashes[digest].append(essay_id)

    # --------------------------------------------------
    # Detect duplicate text
    # --------------------------------------------------

    for digest, essay_ids in hashes.items():

        if len(essay_ids) <= 1:
            continue

        labels = {
            metadata[essay_id]["label"]
            for essay_id in essay_ids
            if essay_id in metadata
        }

        if len(labels) > 1:

            errors.append(
                "Identical normalized text appears under "
                f"multiple labels: {essay_ids}"
            )

        else:

            warnings.append(
                "Duplicate normalized text: "
                f"{essay_ids}"
            )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()

    if warnings:

        print("WARNINGS")
        print("-" * 70)

        for warning in warnings:
            print(f"WARNING: {warning}")

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

    if not discovered_files:

        print(
            "No essay files found yet. "
            "File structure is ready for ingestion."
        )

    else:

        print(
            "All essay files have valid metadata "
            "and passed integrity checks."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()