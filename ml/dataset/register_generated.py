from __future__ import annotations

import json
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
AI_DIR = RAW_DIR / "ai"
METADATA_FILE = RAW_DIR / "metadata.csv"


FIELDNAMES = [
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


def load_metadata() -> list[dict[str, str]]:
    if not METADATA_FILE.exists():
        return []

    with METADATA_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def save_metadata(rows: list[dict[str, str]]) -> None:
    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:

    rows = load_metadata()

    existing_ids = {
        row["essay_id"].strip()
        for row in rows
    }

    json_files = sorted(
        AI_DIR.glob("ai_*.json")
    )

    print("=" * 70)
    print("PROVENANCE — REGISTER GENERATED ESSAYS")
    print("=" * 70)
    print(f"Provenance files found: {len(json_files)}")
    print()

    registered = 0

    for json_path in json_files:

        data = json.loads(
            json_path.read_text(
                encoding="utf-8"
            )
        )

        essay_id = data["essay_id"]

        essay_path = (
            AI_DIR / f"{essay_id}.txt"
        )

        if not essay_path.exists():
            print(
                f"SKIP: {essay_id} — "
                "essay file missing"
            )
            continue

        if essay_id in existing_ids:
            print(
                f"SKIP: {essay_id} — "
                "already registered"
            )
            continue

        row = {
            "essay_id": essay_id,
            "label": "ai",
            "source": "synthetic_local_llm",
            "topic": data["topic"],
            "generation_model": data["model"],
            "generation_prompt": data["prompt"],
            "human_editing": "",
            "ai_intervention_level": "full",
            "source_group": f"ai_{essay_id}",
            "split": "",
            "author_language_background": "not_applicable",
            "notes": (
                f"Generated locally with Ollama. "
                f"Temperature={data['temperature']}; "
                f"Seed={data['seed']}."
            ),
        }

        rows.append(row)
        existing_ids.add(essay_id)

        registered += 1

        print(
            f"REGISTERED: {essay_id} "
            f"({data['topic']}, "
            f"{data['model']}, "
            f"{data['word_count']} words)"
        )

    save_metadata(rows)

    print()
    print("-" * 70)
    print(f"New records registered: {registered}")
    print(f"Total metadata records: {len(rows)}")
    print("=" * 70)


if __name__ == "__main__":
    main()