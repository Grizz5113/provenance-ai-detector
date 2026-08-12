from __future__ import annotations

from dataclasses import dataclass
from math import log

import re
from collections import Counter


@dataclass
class LexicalFeatures:
    token_count: int
    unique_token_count: int
    type_token_ratio: float
    hapax_ratio: float
    vocabulary_entropy: float
    repeated_token_ratio: float


def tokenize_words(text: str) -> list[str]:
    """
    Extract normalized word tokens from text.
    """

    return re.findall(
        r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b",
        text.lower(),
    )


def calculate_lexical_features(
    text: str,
) -> LexicalFeatures:

    words = tokenize_words(text)

    if not words:
        raise ValueError(
            "Cannot calculate lexical features from empty text."
        )

    counts = Counter(words)

    token_count = len(words)

    unique_token_count = len(counts)

    type_token_ratio = (
        unique_token_count / token_count
    )

    hapax_count = sum(
        1
        for count in counts.values()
        if count == 1
    )

    hapax_ratio = (
        hapax_count / token_count
    )

    vocabulary_entropy = 0.0

    for count in counts.values():

        probability = count / token_count

        vocabulary_entropy -= (
            probability * log(probability)
        )

    repeated_tokens = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    repeated_token_ratio = (
        repeated_tokens / token_count
    )

    return LexicalFeatures(
        token_count=token_count,
        unique_token_count=unique_token_count,
        type_token_ratio=type_token_ratio,
        hapax_ratio=hapax_ratio,
        vocabulary_entropy=vocabulary_entropy,
        repeated_token_ratio=repeated_token_ratio,
    )