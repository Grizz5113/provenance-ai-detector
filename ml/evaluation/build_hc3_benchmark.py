from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "external_sources"
    / "hc3"
    / "all.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "hc3_external"
)
TRAINING_METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "external_training"
    / "hc3"
    / "metadata.csv"
)

SEED = 42
TARGET_PAIRS = 250
MAX_PAIRS_PER_SOURCE = 50
MIN_WORDS = 100
MAX_WORDS = 400
MAX_LENGTH_RATIO = 1.20


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def text_hash(text: str) -> str:
    return hashlib.sha256(
        normalize(text).encode("utf-8")
    ).hexdigest()


def word_count(text: str) -> int:
    return len(text.split())


def main() -> None:

    print("=" * 70)
    print("PROVENANCE — BUILD HC3 EXTERNAL BENCHMARK")
    print("=" * 70)

    if not SOURCE_FILE.exists():
        raise RuntimeError(
            f"HC3 source not found: {SOURCE_FILE}"
        )

    random.seed(SEED)

    records = []
        # ------------------------------------------------------------
    # Load HC3 pairs already used for training.
    # These must never appear in the external benchmark.
    # ------------------------------------------------------------

    training_pair_ids = set()

    if TRAINING_METADATA_FILE.exists():

        with TRAINING_METADATA_FILE.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                pair_id = (
                    row.get("pair_id") or ""
                ).strip()

                if pair_id:
                    training_pair_ids.add(
                        pair_id
                    )

    print(
        f"Training pairs excluded: "
        f"{len(training_pair_ids)}"
    )
    with SOURCE_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            row = json.loads(line)

            question = (
                row.get("question") or ""
            ).strip()

            humans = row.get(
                "human_answers",
                [],
            )

            ais = row.get(
                "chatgpt_answers",
                [],
            )

            source = row.get(
                "source",
                "unknown",
            )

            if not question or not humans or not ais:
                continue

            human_candidates = [
                {
                    "text": text.strip(),
                    "words": word_count(text),
                }
                for text in humans
                if MIN_WORDS
                <= word_count(text)
                <= MAX_WORDS
                and text.strip()
            ]

            ai_candidates = [
                {
                    "text": text.strip(),
                    "words": word_count(text),
                }
                for text in ais
                if MIN_WORDS
                <= word_count(text)
                <= MAX_WORDS
                and text.strip()
            ]

            if not human_candidates or not ai_candidates:
                continue

            # Find the closest human/AI length pair
            best_pair = None

            for human in human_candidates:
                for ai in ai_candidates:

                    shorter = min(
                        human["words"],
                        ai["words"],
                    )

                    longer = max(
                        human["words"],
                        ai["words"],
                    )

                    ratio = (
                        longer / shorter
                    )

                    if ratio <= MAX_LENGTH_RATIO:

                        difference = abs(
                            human["words"]
                            - ai["words"]
                        )

                        candidate = (
                            difference,
                            human,
                            ai,
                        )

                        if (
                            best_pair is None
                            or candidate[0]
                            < best_pair[0]
                        ):
                            best_pair = candidate

                       # No suitable human/AI length-matched pair.
            if best_pair is None:
                continue

            # Identify the HC3 question/pair.
            pair_id = (
                hashlib.sha256(
                    question.encode("utf-8")
                ).hexdigest()[:16]
            )

            # Never allow a pair already used for training
            # to enter the external benchmark.
            if pair_id in training_pair_ids:
                continue

            _, human, ai = best_pair

            records.append(
                {
                    "question": question,
                    "source": source,
                    "human_text": human["text"],
                    "human_words": human["words"],
                    "ai_text": ai["text"],
                    "ai_words": ai["words"],
                }
            )

    print(
    f"Eligible pairs after training exclusion: "
    f"{len(records)}"
)
    if len(records) < TARGET_PAIRS:
        raise RuntimeError(
            f"Only {len(records)} suitable pairs found."
        )

    # Shuffle deterministically.
    random.shuffle(records)

    # Prefer source diversity.
    selected = []
    source_counts = {}

    for record in records:

        source = record["source"]

        # Don't let one source dominate.
                # Keep the benchmark reasonably balanced
        # across HC3 source categories.
        if (
            source_counts.get(source, 0)
            >= MAX_PAIRS_PER_SOURCE
        ):
            continue
        selected.append(record)

        source_counts[source] = (
            source_counts.get(source, 0)
            + 1
        )

        if len(selected) >= TARGET_PAIRS:
            break

    if len(selected) < TARGET_PAIRS:
        # Fall back to remaining records.
        for record in records:

            if record in selected:
                continue

            selected.append(record)

            if len(selected) >= TARGET_PAIRS:
                break

    if len(selected) != TARGET_PAIRS:
        raise RuntimeError(
            "Could not select the requested "
            f"{TARGET_PAIRS} pairs."
        )

    human_dir = OUTPUT_DIR / "human"
    ai_dir = OUTPUT_DIR / "ai"

    human_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ai_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_rows = []

    for index, record in enumerate(
        selected,
        start=1,
    ):

        human_id = (
            f"hc3_human_{index:03d}"
        )

        ai_id = (
            f"hc3_ai_{index:03d}"
        )

        human_path = (
            human_dir
            / f"{human_id}.txt"
        )

        ai_path = (
            ai_dir
            / f"{ai_id}.txt"
        )

        human_path.write_text(
            record["human_text"],
            encoding="utf-8",
        )

        ai_path.write_text(
            record["ai_text"],
            encoding="utf-8",
        )

        pair_id = (
            hashlib.sha256(
                record["question"]
                .encode("utf-8")
            ).hexdigest()[:16]
        )

        metadata_rows.append(
            {
                "essay_id": human_id,
                "label": "human",
                "topic": record["question"],
                "source": record["source"],
                "source_group": (
                    f"hc3_pair_{pair_id}"
                ),
                "ai_intervention_level": "none",
                "notes": (
                    "HC3 human response; "
                    "external evaluation only"
                ),
            }
        )

        metadata_rows.append(
            {
                "essay_id": ai_id,
                "label": "ai",
                "topic": record["question"],
                "source": record["source"],
                "source_group": (
                    f"hc3_pair_{pair_id}"
                ),
                "ai_intervention_level": "full",
                "notes": (
                    "HC3 ChatGPT response; "
                    "external evaluation only"
                ),
            }
        )
    metadata_file = (
        OUTPUT_DIR / "metadata.csv"
    )

    with metadata_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        fieldnames = [
            "essay_id",
            "label",
            "topic",
            "source",
            "source_group",
            "ai_intervention_level",
            "notes",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(metadata_rows)

    print()
    print("-" * 70)
    print("BENCHMARK CREATED")
    print("-" * 70)

    print(
        f"Pairs:              {TARGET_PAIRS}"
    )

    print(
        f"Human samples:      {TARGET_PAIRS}"
    )

    print(
        f"AI samples:         {TARGET_PAIRS}"
    )

    print(
        f"Total samples:      {TARGET_PAIRS * 2}"
    )

    print(
        f"Sources:            {source_counts}"
    )

    print(
        f"Metadata:           {metadata_file}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
