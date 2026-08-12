from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re


@dataclass
class PunctuationFeatures:
    character_count: int
    word_count: int

    punctuation_count: int
    punctuation_density: float
    punctuation_types: int

    comma_count: int
    period_count: int
    semicolon_count: int
    colon_count: int
    question_count: int
    exclamation_count: int

    parentheses_count: int
    quotation_count: int

    contraction_count: int


def calculate_punctuation_features(
    text: str,
) -> PunctuationFeatures:

    if not text or not text.strip():
        raise ValueError(
            "Cannot calculate punctuation features from empty text."
        )

    characters = len(text)

    words = re.findall(
        r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b",
        text,
    )

    word_count = len(words)

    punctuation = Counter(
        character
        for character in text
        if character in ",.;:?!()\"'"
    )

    punctuation_count = sum(punctuation.values())

    punctuation_density = (
        punctuation_count / characters
        if characters > 0
        else 0.0
    )

    contraction_count = len(
        re.findall(
            r"\b[a-zA-Z]+['’][a-zA-Z]+\b",
            text,
        )
    )

    # Count actual double-quote characters.
    # Apostrophes are handled separately as contractions.
    quotation_count = punctuation['"']

    return PunctuationFeatures(
        character_count=characters,
        word_count=word_count,

        punctuation_count=punctuation_count,
        punctuation_density=punctuation_density,
        punctuation_types=len(punctuation),

        comma_count=punctuation[","],
        period_count=punctuation["."],
        semicolon_count=punctuation[";"],
        colon_count=punctuation[":"],
        question_count=punctuation["?"],
        exclamation_count=punctuation["!"],

        parentheses_count=(
            punctuation["("]
            + punctuation[")"]
        ),

        quotation_count=quotation_count,

        contraction_count=contraction_count,
    )