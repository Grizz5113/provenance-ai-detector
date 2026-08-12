from __future__ import annotations

from dataclasses import dataclass

from backend.app.detector.language_model import LanguageModelAnalyzer
from backend.app.features.perplexity import (
    PerplexityFeatures,
    calculate_perplexity_features,
)
from backend.app.features.sentence import Sentence, split_sentences


@dataclass
class SentenceAnalysis:
    sentence: Sentence
    features: PerplexityFeatures


class SentenceAnalyzer:
    """
    Runs language-model measurements on individual sentences.

    This class does not make an AI/human judgement.
    It only produces measurable linguistic features.
    """

    def __init__(self, language_model: LanguageModelAnalyzer):
        self.language_model = language_model

    def analyze(self, text: str) -> list[SentenceAnalysis]:

        sentences = split_sentences(text)

        results: list[SentenceAnalysis] = []

        for sentence in sentences:

            raw_result = self.language_model.score_text(
                sentence.text
            )

            features = calculate_perplexity_features(
                raw_result.tokens
            )

            results.append(
                SentenceAnalysis(
                    sentence=sentence,
                    features=features,
                )
            )

        return results