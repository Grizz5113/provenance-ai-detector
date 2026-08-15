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

BENCHMARK_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "hc3_external"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "external_training"
    / "hc3"
)

SEED = 123
TARGET_PAIRS = 100

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


def load_benchmark_hashes() -> set[str]:

    hashes = set()

    for label in ["human", "ai"]:

        directory = (
            BENCHMARK_DIR / label
        )

        if not directory.exists():
            continue

        for path in directory.glob("*.txt"):

            text = path.read_text(
                encoding="utf-8"
            )

            hashes.add(
                text_hash(text)
            )

    return hashes


def main():

    print("=" * 70)
    print("PROVENANCE — BUILD HC3 TRAINING AUGMENTATION")
    print("=" * 70)

    if not SOURCE_FILE.exists():
        raise RuntimeError(
            f"HC3 source not found: {SOURCE_FILE}"
        )

    random.seed(SEED)

    benchmark_hashes = (
        load_benchmark_hashes()
    )

    print(
        f"Benchmark texts excluded: "
        f"{len(benchmark_hashes)}"
    )

    records = []

    seen_texts = set(
        benchmark_hashes
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

            human_candidates = []

            for text in humans:

                text = text.strip()

                if not text:
                    continue

                words = word_count(text)

                if not (
                    MIN_WORDS
                    <= words
                    <= MAX_WORDS
                ):
                    continue

                if text_hash(text) in seen_texts:
                    continue

                human_candidates.append(
                    {
                        "text": text,
                        "words": words,
                    }
                )

            ai_candidates = []

            for text in ais:

                text = text.strip()

                if not text:
                    continue

                words = word_count(text)

                if not (
                    MIN_WORDS
                    <= words
                    <= MAX_WORDS
                ):
                    continue

                if text_hash(text) in seen_texts:
                    continue

                ai_candidates.append(
                    {
                        "text": text,
                        "words": words,
                    }
                )

            if not human_candidates:
                continue

            if not ai_candidates:
                continue

            # Find the closest human/AI length pair.
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

                    if ratio > MAX_LENGTH_RATIO:
                        continue

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

            if best_pair is None:
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
        f"Eligible pairs: {len(records)}"
    )

    if len(records) < TARGET_PAIRS:
        raise RuntimeError(
            f"Only {len(records)} "
            f"eligible pairs available."
        )

    random.shuffle(records)

    # Keep the source distribution reasonably balanced.
    selected = []
    source_counts = {}

    max_per_source = max(
        20,
        TARGET_PAIRS // 4,
    )

    for record in records:

        source = record["source"]

        if (
            source_counts.get(source, 0)
            >= max_per_source
        ):
            continue

        selected.append(record)

        source_counts[source] = (
            source_counts.get(source, 0)
            + 1
        )

        if len(selected) >= TARGET_PAIRS:
            break

    # Fill remaining slots if necessary.
    if len(selected) < TARGET_PAIRS:

        for record in records:

            if record in selected:
                continue

            selected.append(record)

            if len(selected) >= TARGET_PAIRS:
                break

    if len(selected) != TARGET_PAIRS:
        raise RuntimeError(
            "Could not create requested "
            f"{TARGET_PAIRS} training pairs."
        )

    human_dir = (
        OUTPUT_DIR / "human"
    )

    ai_dir = (
        OUTPUT_DIR / "ai"
    )

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
            f"hc3_train_human_{index:03d}"
        )

        ai_id = (
            f"hc3_train_ai_{index:03d}"
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

        pair_id = hashlib.sha256(
            (
                normalize(
                    record["question"]
                )
            ).encode("utf-8")
        ).hexdigest()[:16]

        metadata_rows.append(
            {
                "essay_id": human_id,
                "label": "human",
                "topic": record["source"],
                "source": "HC3",
                "source_group": (
                    f"hc3_{pair_id}"
                ),
                "ai_intervention_level": "none",
                "pair_id": pair_id,
                "word_count": record[
                    "human_words"
                ],
            }
        )

        metadata_rows.append(
            {
                "essay_id": ai_id,
                "label": "ai",
                "topic": record["source"],
                "source": "HC3",
                "source_group": (
                    f"hc3_{pair_id}"
                ),
                "ai_intervention_level": "full",
                "pair_id": pair_id,
                "word_count": record[
                    "ai_words"
                ],
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

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "essay_id",
                "label",
                "topic",
                "source",
                "source_group",
                "ai_intervention_level",
                "pair_id",
                "word_count",
            ],
        )

        writer.writeheader()

        writer.writerows(
            metadata_rows
        )

    print()
    print("-" * 70)
    print("TRAINING AUGMENTATION CREATED")
    print("-" * 70)

    print(
        f"Pairs:        {TARGET_PAIRS}"
    )

    print(
        f"Human essays: {TARGET_PAIRS}"
    )

    print(
        f"AI essays:    {TARGET_PAIRS}"
    )

    print(
        f"Metadata:     {metadata_file}"
    )

    print()
    print("Sources:")

    for source, count in sorted(
        source_counts.items()
    ):
        print(
            f"  {source:20s}: {count}"
        )

    print()
    print(
        "HC3 benchmark samples were "
        "excluded from this training set."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()