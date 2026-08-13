from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HYBRID_DIR = PROJECT_ROOT / "data" / "raw" / "hybrid"
METADATA_FILE = PROJECT_ROOT / "data" / "raw" / "metadata.csv"


def main() -> None:

    print("=" * 70)
    print("PROVENANCE — REGISTER HYBRID ESSAYS")
    print("=" * 70)

    if not HYBRID_DIR.exists():
        raise FileNotFoundError(
            f"Hybrid directory not found: {HYBRID_DIR}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    with METADATA_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        metadata_by_id = {
    row["essay_id"]: row
    for row in rows
    if row.get("essay_id")
}
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise RuntimeError("Metadata CSV has no header.")

    existing_ids = {
        row["essay_id"]
        for row in rows
        if row.get("essay_id")
    }

    provenance_files = sorted(
        HYBRID_DIR.glob("*.json")
    )

    registered = 0

    for provenance_file in provenance_files:

        with provenance_file.open(
            "r",
            encoding="utf-8",
        ) as f:
            provenance = json.load(f)

        essay_id = provenance["essay_id"]

        source_human_id = provenance.get(
            "source_human_essay",
            "",
        )

        source_metadata = metadata_by_id.get(
            source_human_id,
            {},
        )
        
        

        if essay_id in existing_ids:
            continue

        text_file = (
            HYBRID_DIR
            / f"{essay_id}.txt"
        )

        if not text_file.exists():
            print(
                f"SKIPPED: {essay_id} "
                "has no .txt file"
            )
            continue

        row = {
            "essay_id": essay_id,
            "label": "hybrid",
            "source": "synthetic_local_llm",
            "topic": source_metadata.get("topic", ""),
            "generation_model": provenance.get(
                "model",
                "",
            ),
            "generation_prompt": provenance.get(
                "prompt",
                "",
            ),
            "human_editing": "ai_polished",
            "ai_intervention_level": provenance.get(
                "ai_intervention_level",
                "unknown",
            ),
            "source_group": provenance.get(
                "source_human_essay",
                "",
            ),
            "split": "",
            "author_language_background": "",
            "notes": json.dumps(
                {
                    "operation": provenance.get(
                        "operation",
                        "",
                    ),
                    "temperature": provenance.get(
                        "temperature",
                    ),
                    "seed": provenance.get(
                        "seed",
                    ),
                    "original_word_count": provenance.get(
                        "original_word_count",
                    ),
                    "hybrid_word_count": provenance.get(
                        "hybrid_word_count",
                    ),
                    "edit_ratio": provenance.get(
                        "edit_ratio",
                    ),
                },
                ensure_ascii=False,
            ),
        }

        rows.append(row)
        existing_ids.add(essay_id)
        registered += 1

        print(
            f"REGISTERED: {essay_id} | "
            f"source={row['source_group']} | "
            f"edit_ratio="
            f"{provenance.get('edit_ratio', 'N/A')}"
        )

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Registered: {registered} hybrid essays")
    print("=" * 70)


if __name__ == "__main__":
    main()
    