from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_FILE = RAW_DIR / "metadata.csv"

VALID_LABELS = {"human", "ai", "hybrid"}

VALID_TOPICS = {
    "challenge",
    "academic_curiosity",
    "leadership",
    "community",
    "personal_growth",
}

VALID_INTERVENTION_LEVELS = {
    "none",
    "light",
    "moderate",
    "heavy",
    "full",
}


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def calculate_hash(text: str) -> str:
    normalized = normalize_text(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def load_metadata() -> list[dict[str, str]]:
    if not METADATA_FILE.exists():
        return []

    with METADATA_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        return list(csv.DictReader(file))


def check_duplicate(
    text: str,
    metadata: list[dict[str, str]],
) -> str | None:

    target_hash = calculate_hash(text)

    for row in metadata:

        essay_id = row.get("essay_id", "").strip()

        if not essay_id:
            continue

        essay_path = (
            RAW_DIR
            / row["label"].strip().lower()
            / f"{essay_id}.txt"
        )

        if not essay_path.exists():
            continue

        existing_text = essay_path.read_text(
            encoding="utf-8"
        )

        if calculate_hash(existing_text) == target_hash:
            return essay_id

    return None


def append_metadata(row: dict[str, str]) -> None:

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

    existing_rows = load_metadata()

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for existing_row in existing_rows:
            writer.writerow(existing_row)

        writer.writerow(row)


def validate_arguments(args: argparse.Namespace) -> None:

    if args.label not in VALID_LABELS:
        raise ValueError(
            f"Invalid label: {args.label}. "
            f"Choose from {sorted(VALID_LABELS)}."
        )

    if args.topic not in VALID_TOPICS:
        raise ValueError(
            f"Invalid topic: {args.topic}. "
            f"Choose from {sorted(VALID_TOPICS)}."
        )

    if (
        args.ai_intervention_level
        not in VALID_INTERVENTION_LEVELS
    ):
        raise ValueError(
            "Invalid AI intervention level: "
            f"{args.ai_intervention_level}"
        )

    if not args.file.exists():
        raise FileNotFoundError(
            f"Essay file does not exist: {args.file}"
        )

    if not args.source_group.strip():
        raise ValueError(
            "source_group cannot be empty."
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Add an essay to the Provenance detector "
            "dataset."
        )
    )

    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the essay .txt file.",
    )

    parser.add_argument(
        "--label",
        required=True,
        choices=sorted(VALID_LABELS),
    )

    parser.add_argument(
        "--topic",
        required=True,
        choices=sorted(VALID_TOPICS),
    )

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--generation-model",
        default="",
    )

    parser.add_argument(
        "--generation-prompt",
        default="",
    )

    parser.add_argument(
        "--human-editing",
        default="",
    )

    parser.add_argument(
        "--ai-intervention-level",
        default="none",
        choices=sorted(VALID_INTERVENTION_LEVELS),
    )

    parser.add_argument(
        "--source-group",
        required=True,
    )

    parser.add_argument(
        "--author-language-background",
        default="unknown",
    )

    parser.add_argument(
        "--notes",
        default="",
    )

    args = parser.parse_args()

    validate_arguments(args)

    text = args.file.read_text(
        encoding="utf-8"
    )

    if not text.strip():
        raise ValueError(
            "Essay file is empty."
        )

    word_count = len(text.split())

    if word_count < 50:
        raise ValueError(
            f"Essay contains only {word_count} words. "
            "Minimum is 50."
        )

    metadata = load_metadata()

    essay_id = args.file.stem

    if any(
        row.get("essay_id", "").strip() == essay_id
        for row in metadata
    ):
        raise ValueError(
            f"essay_id '{essay_id}' already exists."
        )

    duplicate_id = check_duplicate(
        text,
        metadata,
    )

    if duplicate_id:
        raise ValueError(
            "This essay has identical normalized text "
            f"to existing essay '{duplicate_id}'."
        )

    expected_directory = RAW_DIR / args.label

    expected_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        expected_directory
        / f"{essay_id}.txt"
    )

    if destination.exists():
        raise ValueError(
            f"Destination already exists: {destination}"
        )

    destination.write_text(
        text,
        encoding="utf-8",
    )

    metadata_row = {
        "essay_id": essay_id,
        "label": args.label,
        "source": args.source,
        "topic": args.topic,
        "generation_model": args.generation_model,
        "generation_prompt": args.generation_prompt,
        "human_editing": args.human_editing,
        "ai_intervention_level": (
            args.ai_intervention_level
        ),
        "source_group": args.source_group,
        "split": "",
        "author_language_background": (
            args.author_language_background
        ),
        "notes": args.notes,
    }

    append_metadata(metadata_row)

    print()
    print("=" * 70)
    print("PROVENANCE — ESSAY INGESTION")
    print("=" * 70)

    print(f"Essay ID:       {essay_id}")
    print(f"Label:          {args.label}")
    print(f"Topic:          {args.topic}")
    print(f"Words:          {word_count}")
    print(f"Source group:   {args.source_group}")
    print(f"Stored at:      {destination}")

    print()
    print("Essay successfully added.")
    print("=" * 70)


if __name__ == "__main__":
    main()