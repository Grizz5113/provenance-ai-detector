from __future__ import annotations

import csv
import io
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

RAW_HUMAN_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "human"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "metadata.csv"
)

CORPUS_FILE = (
    "persuade_2.0_human_scores_demo_id_github.csv"
)


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


def load_corpus() -> dict[str, dict[str, str]]:

    with zipfile.ZipFile(
        ZIP_FILE,
        "r",
    ) as archive:

        with archive.open(
            CORPUS_FILE,
            "r",
        ) as binary_file:

            reader = csv.DictReader(
                io.TextIOWrapper(
                    binary_file,
                    encoding="utf-8",
                    newline="",
                )
            )

            return {
                row["essay_id_comp"].strip(): row
                for row in reader
                if row.get("essay_id_comp")
            }


def load_metadata() -> list[dict[str, str]]:

    if not METADATA_FILE.exists():
        return []

    with METADATA_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def save_metadata(
    rows: list[dict[str, str]],
) -> None:

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

    corpus = load_corpus()
    rows = load_metadata()

    existing_ids = {
        row["essay_id"]
        for row in rows
    }

    # Read the selection made by import_persuade.py.
    # The importer currently doesn't save its source IDs,
    # so we identify them by matching essay text.
    selected_files = sorted(
        RAW_HUMAN_DIR.glob(
            "human_*.txt"
        )
    )

    print("=" * 70)
    print("PROVENANCE — REGISTER PERSUADE ESSAYS")
    print("=" * 70)

    print(
        f"Corpus records: {len(corpus)}"
    )

    print(
        f"Human files:    {len(selected_files)}"
    )

    registered = 0

    for human_file in selected_files:

        essay_id = human_file.stem

        if essay_id in existing_ids:
            print(
                f"SKIP: {essay_id} "
                f"(already registered)"
            )
            continue

        text = human_file.read_text(
            encoding="utf-8"
        ).strip()

        matches = [
            row
            for row in corpus.values()
            if row.get(
                "full_text",
                "",
            ).strip()
            == text
        ]

        if len(matches) != 1:
            print(
                f"ERROR: Could not uniquely "
                f"match {essay_id} "
                f"to PERSUADE source."
            )
            continue

        source_row = matches[0]

        original_id = source_row[
            "essay_id_comp"
        ].strip()

        prompt = source_row.get(
            "prompt_name",
            "",
        ).strip()

        grade = source_row.get(
            "grade_level",
            "",
        ).strip()

        ell_status = source_row.get(
            "ell_status",
            "",
        ).strip()

        word_count = source_row.get(
            "word_count",
            "",
        ).strip()

        row = {
            "essay_id": essay_id,
            "label": "human",
            "source": "PERSUADE_2.0",
            "topic": prompt,
            "generation_model": "",
            "generation_prompt": "",
            "human_editing": "",
            "ai_intervention_level": "none",
            "source_group": f"persuade_{original_id}",
            "split": "",
            "author_language_background": (
                "ELL"
                if ell_status.lower() == "yes"
                else "non-ELL"
            ),
            "notes": (
                f"PERSUADE 2.0 human-writing baseline. "
                f"Original ID={original_id}; "
                f"grade={grade}; "
                f"word_count={word_count}. "
                f"Licensed source: CC BY-NC-SA 4.0."
            ),
        }

        rows.append(row)
        existing_ids.add(essay_id)
        registered += 1

        print(
            f"REGISTERED: {essay_id} | "
            f"{prompt} | "
            f"grade={grade} | "
            f"{'ELL' if ell_status.lower() == 'yes' else 'non-ELL'}"
        )

    save_metadata(rows)

    print()
    print(
        f"New records registered: "
        f"{registered}"
    )

    print(
        f"Total metadata records: "
        f"{len(rows)}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()