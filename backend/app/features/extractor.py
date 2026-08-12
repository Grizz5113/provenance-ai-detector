from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from backend.app.detector.language_model import LanguageModelAnalyzer
from backend.app.features.lexical import calculate_lexical_features
from backend.app.features.punctuation import calculate_punctuation_features
from backend.app.features.repetition import calculate_repetition_features
from backend.app.features.rhythm import calculate_rhythm_features
from backend.app.features.sentence import split_sentences


@dataclass
class EssayFeatureVector:
    """
    Unified representation of measurable characteristics
    extracted from an essay.

    These features describe the text. They do not make an
    AI/human classification.
    """

    # Language-model features
    token_count: int
    mean_nll: float
    perplexity: float
    nll_std: float
    nll_min: float
    nll_max: float
    nll_median: float
    nll_p90: float

    # Rhythm
    sentence_count: int
    mean_sentence_length: float
    median_sentence_length: float
    sentence_length_std: float
    min_sentence_length: int
    max_sentence_length: int
    sentence_length_cv: float

    # Lexical
    unique_token_count: int
    type_token_ratio: float
    hapax_ratio: float
    vocabulary_entropy: float
    repeated_token_ratio: float

    # Repetition
    unique_bigrams: int
    repeated_bigrams: int
    bigram_repetition_ratio: float
    unique_trigrams: int
    repeated_trigrams: int
    trigram_repetition_ratio: float
    most_common_bigram_count: int
    most_common_trigram_count: int

    # Punctuation
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EssayFeatureExtractor:
    """
    Coordinates all feature extraction components.

    This class is the orchestration layer. Individual feature
    modules remain responsible for their own calculations.
    """

    def __init__(
        self,
        language_model: LanguageModelAnalyzer,
    ) -> None:
        self.language_model = language_model

    def extract(self, text: str) -> EssayFeatureVector:

        if not text or not text.strip():
            raise ValueError("Essay text cannot be empty.")

        # Sentence analysis
        sentences = split_sentences(text)

        if not sentences:
            raise ValueError(
                "Could not identify any sentences."
            )

        # Language-model measurement
        raw_result = self.language_model.score_text(text)

        nll_values = [
            token.negative_log_likelihood
            for token in raw_result.tokens
        ]

        if not nll_values:
            raise ValueError(
                "Language model produced no token scores."
            )

        import numpy as np
        from math import exp

        nll_array = np.asarray(
            nll_values,
            dtype=np.float64,
        )

        mean_nll = float(np.mean(nll_array))

        mean_perplexity = float(exp(mean_nll))

        nll_std = float(np.std(nll_array))
        nll_min = float(np.min(nll_array))
        nll_max = float(np.max(nll_array))
        nll_median = float(np.median(nll_array))
        nll_p90 = float(np.percentile(nll_array, 90))

        # Traditional text features
        rhythm = calculate_rhythm_features(sentences)
        lexical = calculate_lexical_features(text)
        repetition = calculate_repetition_features(text)
        punctuation = calculate_punctuation_features(text)

        return EssayFeatureVector(
            # Language model
            token_count=raw_result.token_count,
            mean_nll=mean_nll,
            perplexity=mean_perplexity,
            nll_std=nll_std,
            nll_min=nll_min,
            nll_max=nll_max,
            nll_median=nll_median,
            nll_p90=nll_p90,

            # Rhythm
            sentence_count=rhythm.sentence_count,
            mean_sentence_length=rhythm.mean_sentence_length,
            median_sentence_length=rhythm.median_sentence_length,
            sentence_length_std=rhythm.std_sentence_length,
            min_sentence_length=rhythm.min_sentence_length,
            max_sentence_length=rhythm.max_sentence_length,
            sentence_length_cv=rhythm.coefficient_of_variation,

            # Lexical
            unique_token_count=lexical.unique_token_count,
            type_token_ratio=lexical.type_token_ratio,
            hapax_ratio=lexical.hapax_ratio,
            vocabulary_entropy=lexical.vocabulary_entropy,
            repeated_token_ratio=lexical.repeated_token_ratio,

            # Repetition
            unique_bigrams=repetition.unique_bigrams,
            repeated_bigrams=repetition.repeated_bigrams,
            bigram_repetition_ratio=repetition.bigram_repetition_ratio,
            unique_trigrams=repetition.unique_trigrams,
            repeated_trigrams=repetition.repeated_trigrams,
            trigram_repetition_ratio=repetition.trigram_repetition_ratio,
            most_common_bigram_count=repetition.most_common_bigram_count,
            most_common_trigram_count=repetition.most_common_trigram_count,

            # Punctuation
            punctuation_count=punctuation.punctuation_count,
            punctuation_density=punctuation.punctuation_density,
            punctuation_types=punctuation.punctuation_types,
            comma_count=punctuation.comma_count,
            period_count=punctuation.period_count,
            semicolon_count=punctuation.semicolon_count,
            colon_count=punctuation.colon_count,
            question_count=punctuation.question_count,
            exclamation_count=punctuation.exclamation_count,
            parentheses_count=punctuation.parentheses_count,
            quotation_count=punctuation.quotation_count,
            contraction_count=punctuation.contraction_count,
        )