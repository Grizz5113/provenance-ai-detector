from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median, pstdev

from backend.app.features.sentence import Sentence


@dataclass
class RhythmFeatures:
    sentence_count: int
    mean_sentence_length: float
    median_sentence_length: float
    std_sentence_length: float
    min_sentence_length: int
    max_sentence_length: int
    coefficient_of_variation: float


def calculate_rhythm_features(
    sentences: list[Sentence],
) -> RhythmFeatures:

    if not sentences:
        raise ValueError(
            "Cannot calculate rhythm features from empty text."
        )

    lengths = [
        len(sentence.text.split())
        for sentence in sentences
    ]

    mean_length = mean(lengths)

    standard_deviation = (
        pstdev(lengths)
        if len(lengths) > 1
        else 0.0
    )

    coefficient_of_variation = (
        standard_deviation / mean_length
        if mean_length > 0
        else 0.0
    )

    return RhythmFeatures(
        sentence_count=len(lengths),
        mean_sentence_length=mean_length,
        median_sentence_length=median(lengths),
        std_sentence_length=standard_deviation,
        min_sentence_length=min(lengths),
        max_sentence_length=max(lengths),
        coefficient_of_variation=coefficient_of_variation,
    )