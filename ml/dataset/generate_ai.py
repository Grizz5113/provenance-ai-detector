from __future__ import annotations

import argparse
from pathlib import Path


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


def build_prompt(topic: str, variation: int) -> str:

    topic_prompt = TOPICS[topic]

    variations = [
        "Write in a natural first-person voice.",
        "Use a reflective and personal tone.",
        "Focus on concrete experiences rather than abstract claims.",
        "Use varied sentence lengths and natural transitions.",
        "Write as a genuine college applicant rather than as an academic article.",
    ]

    instruction = variations[
        variation % len(variations)
    ]

    return f"""
Write a college admissions essay of approximately 500–700 words.

Prompt:
{topic_prompt}

Requirements:
- {instruction}
- Do not mention artificial intelligence.
- Do not describe the writing process.
- Do not use headings.
- Do not add a title.
- Return only the essay.
""".strip()


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Generate reproducible prompts for AI essays."
    )

    parser.add_argument(
        "--topic",
        choices=sorted(TOPICS),
        required=True,
    )

    parser.add_argument(
        "--variation",
        type=int,
        default=0,
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

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        prompt,
        encoding="utf-8",
    )

    print("=" * 70)
    print("PROVENANCE — AI DATASET PROMPT")
    print("=" * 70)
    print()
    print(f"Topic:     {args.topic}")
    print(f"Variation: {args.variation}")
    print()
    print(prompt)
    print()
    print(f"Saved prompt to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()