from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np

from backend.app.detector.language_model import TokenScore


@dataclass
class PerplexityFeatures:
    token_count: int
    mean_nll: float
    perplexity: float
    nll_std: float
    nll_min: float
    nll_max: float
    nll_median: float
    nll_p90: float


def calculate_perplexity_features(
    tokens: list[TokenScore],
) -> PerplexityFeatures:

    if not tokens:
        raise ValueError(
            "Cannot calculate perplexity from empty token list."
        )

    nll_values = np.asarray(
        [
            token.negative_log_likelihood
            for token in tokens
        ],
        dtype=np.float64,
    )

    mean_nll = float(np.mean(nll_values))

    return PerplexityFeatures(
        token_count=len(nll_values),
        mean_nll=mean_nll,
        perplexity=float(exp(mean_nll)),
        nll_std=float(np.std(nll_values)),
        nll_min=float(np.min(nll_values)),
        nll_max=float(np.max(nll_values)),
        nll_median=float(np.median(nll_values)),
        nll_p90=float(np.percentile(nll_values, 90)),
    )