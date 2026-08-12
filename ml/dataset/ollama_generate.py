from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


TOPICS = {
    "challenge": (
        "Describe a significant challenge or failure you experienced "
        "and explain how it changed the way you approach difficult situations."
    ),
    "academic_curiosity": (
        "Describe an academic subject or question that genuinely "
        "interests you and explain why you want to explore it further."
    ),
    "leadership": (
        "Describe a situation in which you took responsibility or "
        "leadership and explain what you learned from the experience."
    ),
    "community": (
        "Describe an experience involving your community or helping "
        "others and explain how it affected your perspective."
    ),
    "personal_growth": (
        "Describe an experience that significantly changed your "
        "understanding of yourself or the world around you."
    ),
}


VARIATIONS = [
    "Use a natural first-person voice.",
    "Use a reflective and personal tone.",
    "Focus on concrete experiences rather than abstract claims.",
    "Use varied sentence lengths and natural transitions.",
    "Write as a genuine college applicant rather than as an academic article.",
]


def build_prompt(topic: str, variation: int) -> str:
    return f"""
Write a college admissions essay of approximately 500–700 words.

Prompt:
{TOPICS[topic]}

Requirements:
- {VARIATIONS[variation % len(VARIATIONS)]}
- Do not mention artificial intelligence.
- Do not describe the writing process.
- Do not use headings.
- Do not add a title.
- Return ONLY the essay.
- Do not provide an explanation.
- Do not provide notes.
- Do not ask questions.
- Do not surround the essay with quotation marks.
""".strip()


def call_ollama(
    model: str,
    prompt: str,
    temperature: float,
    seed: int,
) -> str:

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=600,
    ) as response:

        result = json.loads(
            response.read().decode("utf-8")
        )

    return result["response"].strip()


def validate_essay(text: str) -> None:

    if not text:
        raise ValueError("Model returned empty text.")

    word_count = len(text.split())

    if word_count < 100:
        raise ValueError(
            f"Generated essay is too short: {word_count} words."
        )

    suspicious_phrases = [
        "here's a draft",
        "here is a draft",
        "here's an essay",
        "here is an essay",
        "breakdown:",
        "this essay",
        "i hope this helps",
        "let me know if",
    ]

    lower_text = text.lower()

    for phrase in suspicious_phrases:
        if phrase in lower_text:
            raise ValueError(
                "Generated output appears to contain "
                f"model commentary: '{phrase}'"
            )


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Generate an AI essay using Ollama."
    )

    parser.add_argument(
        "--model",
        default="gemma3:4b",
    )

    parser.add_argument(
        "--topic",
        required=True,
        choices=sorted(TOPICS),
    )

    parser.add_argument(
        "--variation",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1001,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--essay-id",
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    prompt = build_prompt(
        args.topic,
        args.variation,
    )

    print()
    print("=" * 70)
    print("PROVENANCE — AI ESSAY GENERATION")
    print("=" * 70)

    print(f"Model:        {args.model}")
    print(f"Essay ID:     {args.essay_id}")
    print(f"Topic:        {args.topic}")
    print(f"Variation:    {args.variation}")
    print(f"Temperature:  {args.temperature}")
    print(f"Seed:         {args.seed}")
    print()

    print("Generating...")

    essay = call_ollama(
        model=args.model,
        prompt=prompt,
        temperature=args.temperature,
        seed=args.seed,
    )

    validate_essay(essay)

    word_count = len(essay.split())

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        essay + "\n",
        encoding="utf-8",
    )

    provenance = {
        "essay_id": args.essay_id,
        "label": "ai",
        "topic": args.topic,
        "model": args.model,
        "prompt": prompt,
        "temperature": args.temperature,
        "seed": args.seed,
        "word_count": word_count,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    provenance_path = args.output.with_suffix(
        ".json"
    )

    provenance_path.write_text(
        json.dumps(
            provenance,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Generated words: {word_count}")
    print(f"Essay saved:     {args.output}")
    print(f"Provenance:      {provenance_path}")
    print()
    print("Generation successful.")
    print("=" * 70)


if __name__ == "__main__":
    main()