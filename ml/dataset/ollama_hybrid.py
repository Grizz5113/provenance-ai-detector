from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.features.text_similarity import (
    calculate_edit_ratio,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HUMAN_DIR = PROJECT_ROOT / "data" / "raw" / "human"
HYBRID_DIR = PROJECT_ROOT / "data" / "raw" / "hybrid"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

MODEL = "gemma3:4b"


POLISH_PROMPT = """You are editing a student's college-style essay.

Lightly polish the essay while preserving the student's:

- meaning
- personal experiences
- opinions
- facts
- first-person voice
- paragraph structure
- approximate length

Do NOT invent experiences or facts.

Do NOT substantially rewrite the essay.

Make only the kinds of changes a language model might make during
ordinary proofreading or polishing:

- improve awkward phrasing
- improve grammar
- improve sentence flow
- improve word choice where appropriate
- remove obvious repetition

Some original wording should remain unchanged.

Return ONLY the polished essay.
Do not explain your changes.
Do not add a title.
Do not add commentary.

ORIGINAL ESSAY:

"""


def generate_with_ollama(
    prompt: str,
    temperature: float,
    seed: int,
) -> str:

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(
            "utf-8"
        ),
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=600,
    ) as response:

        result = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

    text = result.get(
        "response",
        "",
    ).strip()

    if not text:
        raise RuntimeError(
            "Ollama returned empty output."
        )

    return text


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create an AI-polished "
            "hybrid essay."
        )
    )

    parser.add_argument(
        "--essay-id",
        required=True,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2001,
    )

    args = parser.parse_args()

    source_file = (
        HUMAN_DIR
        / f"{args.essay_id}.txt"
    )

    if not source_file.exists():
        raise FileNotFoundError(
            f"Human essay not found: "
            f"{source_file}"
        )

    HYBRID_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_text = source_file.read_text(
        encoding="utf-8"
    ).strip()

    prompt = (
        POLISH_PROMPT
        + original_text
    )

    print("=" * 70)
    print("PROVENANCE — HYBRID GENERATION")
    print("=" * 70)

    print(
        f"Source essay: {args.essay_id}"
    )

    print(
        f"Model:        {MODEL}"
    )

    print(
        f"Temperature:  {args.temperature}"
    )

    print(
        f"Seed:         {args.seed}"
    )

    print()
    print("Generating...")

    polished_text = generate_with_ollama(
        prompt=prompt,
        temperature=args.temperature,
        seed=args.seed,
    )

    hybrid_id = (
        args.essay_id.replace(
            "human_",
            "hybrid_",
        )
    )

    output_file = (
        HYBRID_DIR
        / f"{hybrid_id}.txt"
    )

    provenance_file = (
        HYBRID_DIR
        / f"{hybrid_id}.json"
    )

    output_file.write_text(
        polished_text + "\n",
        encoding="utf-8",
    )

    provenance = {
    "essay_id": hybrid_id,
    "label": "hybrid",
    "source_human_essay": args.essay_id,
    "model": MODEL,
    "temperature": args.temperature,
    "seed": args.seed,
    "operation": "light_ai_polish",
    "generated_at": datetime.now(
        timezone.utc
    ).isoformat(),
    "original_word_count": len(
        original_text.split()
    ),
    "hybrid_word_count": len(
        polished_text.split()
    ),
    "edit_ratio": calculate_edit_ratio(
        original_text,
        polished_text,
    ),
    "prompt": POLISH_PROMPT,
}

    provenance_file.write_text(
        json.dumps(
            provenance,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Original words: "
        f"{len(original_text.split())}"
    )

    print(
        f"Hybrid words:   "
        f"{len(polished_text.split())}"
    )

    print(
        f"Essay saved:    "
        f"{output_file}"
    )

    print(
        f"Provenance:     "
        f"{provenance_file}"
    )

    print()
    print(
    f"Edit ratio:     "
    f"{calculate_edit_ratio(original_text, polished_text):.4f}"
)
    print("Generation successful.")
    print("=" * 70)


if __name__ == "__main__":
    main()