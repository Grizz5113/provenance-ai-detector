from __future__ import annotations

import csv
import io
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

RAW_HUMAN_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "human"
)

CORPUS_FILE = "persuade_2.0_human_scores_demo_id_github.csv"

TARGET_TOTAL = 10
TARGET_PER_ELL_GROUP = 5

MIN_WORDS = 250
MAX_WORDS = 800

RANDOM_SEED = 42


def normalize_ell_status(value: str) -> str:

    value = value.strip().lower()

    if value == "yes":
        return "ELL"

    if value == "no":
        return "non-ELL"

    return "unknown"


def load_rows() -> list[dict[str, str]]:

    with zipfile.ZipFile(
        ZIP_FILE,
        "r",
    ) as archive:

        with archive.open(
            CORPUS_FILE,
            "r",
        ) as binary_file:

            text_stream = io.TextIOWrapper(
                binary_file,
                encoding="utf-8",
                newline="",
            )

            reader = csv.DictReader(
                text_stream
            )

            return list(reader)


def prepare_candidates(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:

    candidates = []

    for row in rows:

        essay_id = row.get(
            "essay_id_comp",
            "",
        ).strip()

        text = row.get(
            "full_text",
            "",
        ).strip()

        prompt_name = row.get(
            "prompt_name",
            "",
        ).strip()

        if not essay_id:
            continue

        if not text:
            continue

        if not prompt_name:
            continue

        try:
            word_count = int(
                row.get(
                    "word_count",
                    "0",
                )
            )
        except ValueError:
            continue

        if not (
            MIN_WORDS
            <= word_count
            <= MAX_WORDS
        ):
            continue

        ell_status = normalize_ell_status(
            row.get(
                "ell_status",
                "",
            )
        )

        if ell_status not in {
            "ELL",
            "non-ELL",
        }:
            continue

        candidates.append(
            {
                "original_id": essay_id,
                "text": text,
                "word_count": str(word_count),
                "prompt_name": prompt_name,
                "task": row.get(
                    "task",
                    "",
                ).strip(),
                "assignment": row.get(
                    "assignment",
                    "",
                ).strip(),
                "ell_status": ell_status,
            }
        )

    return candidates


def select_group(
    candidates: list[dict[str, str]],
    target: int,
    used_ids: set[str],
) -> list[dict[str, str]]:

    random.shuffle(candidates)

    selected = []

    # First pass: maximize prompt diversity.
    used_prompts: set[str] = set()

    for candidate in candidates:

        if candidate["original_id"] in used_ids:
            continue

        if candidate["prompt_name"] in used_prompts:
            continue

        selected.append(candidate)

        used_ids.add(
            candidate["original_id"]
        )

        used_prompts.add(
            candidate["prompt_name"]
        )

        if len(selected) >= target:
            return selected

    # Second pass: fill remaining slots.
    for candidate in candidates:

        if candidate["original_id"] in used_ids:
            continue

        selected.append(candidate)

        used_ids.add(
            candidate["original_id"]
        )

        if len(selected) >= target:
            break

    return selected


def select_balanced(
    candidates: list[dict[str, str]],
) -> list[dict[str, str]]:

    random.seed(RANDOM_SEED)

    ell_candidates = [
        row
        for row in candidates
        if row["ell_status"] == "ELL"
    ]

    non_ell_candidates = [
        row
        for row in candidates
        if row["ell_status"] == "non-ELL"
    ]

    print(
        f"ELL candidates:     "
        f"{len(ell_candidates)}"
    )

    print(
        f"non-ELL candidates: "
        f"{len(non_ell_candidates)}"
    )

    used_ids: set[str] = set()

    selected_ell = select_group(
        ell_candidates,
        TARGET_PER_ELL_GROUP,
        used_ids,
    )

    selected_non_ell = select_group(
        non_ell_candidates,
        TARGET_PER_ELL_GROUP,
        used_ids,
    )

    return selected_ell + selected_non_ell


def save_essays(
    selected: list[dict[str, str]],
) -> None:

    RAW_HUMAN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, row in enumerate(
        selected,
        start=1,
    ):

        essay_id = (
            f"human_{index:03d}"
        )

        destination = (
            RAW_HUMAN_DIR
            / f"{essay_id}.txt"
        )

        destination.write_text(
            row["text"] + "\n",
            encoding="utf-8",
        )


def main() -> None:

    if not ZIP_FILE.exists():
        raise FileNotFoundError(
            f"Dataset archive not found:\n{ZIP_FILE}"
        )

    print("=" * 70)
    print("PROVENANCE — PERSUADE HUMAN IMPORT")
    print("=" * 70)

    rows = load_rows()

    print(
        f"Corpus records loaded: {len(rows)}"
    )

    candidates = prepare_candidates(
        rows
    )

    print(
        f"Eligible candidates: "
        f"{len(candidates)}"
    )

    selected = select_balanced(
        candidates
    )

    if len(selected) != TARGET_TOTAL:
        raise RuntimeError(
            "Could not construct a balanced "
            f"{TARGET_TOTAL}-essay subset."
        )

    save_essays(selected)

    print()
    print("Selected essays:")
    print("-" * 70)

    for index, row in enumerate(
        selected,
        start=1,
    ):

        print(
            f"human_{index:03d} | "
            f"{row['word_count']} words | "
            f"{row['ell_status']} | "
            f"{row['prompt_name']} | "
            f"source={row['original_id']}"
        )

    print("-" * 70)

    print(
        f"Imported: {len(selected)} essays"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()