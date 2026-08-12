from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from backend.app.features.lexical import tokenize_words


@dataclass
class RepetitionFeatures:
    token_count: int

    unique_bigrams: int
    repeated_bigrams: int
    bigram_repetition_ratio: float

    unique_trigrams: int
    repeated_trigrams: int
    trigram_repetition_ratio: float

    most_common_bigram_count: int
    most_common_trigram_count: int


def build_ngrams(
    words: list[str],
    n: int,
) -> list[tuple[str, ...]]:

    if len(words) < n:
        return []

    return [
        tuple(words[index:index + n])
        for index in range(len(words) - n + 1)
    ]


def calculate_repetition_features(
    text: str,
) -> RepetitionFeatures:

    words = tokenize_words(text)

    if not words:
        raise ValueError(
            "Cannot calculate repetition features from empty text."
        )

    bigrams = build_ngrams(words, 2)
    trigrams = build_ngrams(words, 3)

    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)

    repeated_bigram_occurrences = sum(
        count - 1
        for count in bigram_counts.values()
        if count > 1
    )

    repeated_trigram_occurrences = sum(
        count - 1
        for count in trigram_counts.values()
        if count > 1
    )

    bigram_repetition_ratio = (
        repeated_bigram_occurrences / len(bigrams)
        if bigrams
        else 0.0
    )

    trigram_repetition_ratio = (
        repeated_trigram_occurrences / len(trigrams)
        if trigrams
        else 0.0
    )

    most_common_bigram_count = (
        bigram_counts.most_common(1)[0][1]
        if bigram_counts
        else 0
    )

    most_common_trigram_count = (
        trigram_counts.most_common(1)[0][1]
        if trigram_counts
        else 0
    )

    return RepetitionFeatures(
        token_count=len(words),

        unique_bigrams=len(bigram_counts),
        repeated_bigrams=sum(
            1
            for count in bigram_counts.values()
            if count > 1
        ),
        bigram_repetition_ratio=bigram_repetition_ratio,

        unique_trigrams=len(trigram_counts),
        repeated_trigrams=sum(
            1
            for count in trigram_counts.values()
            if count > 1
        ),
        trigram_repetition_ratio=trigram_repetition_ratio,

        most_common_bigram_count=most_common_bigram_count,
        most_common_trigram_count=most_common_trigram_count,
    )