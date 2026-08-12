from backend.app.detector.language_model import LanguageModelAnalyzer
from backend.app.features.perplexity import calculate_perplexity_features


def main() -> None:

    analyzer = LanguageModelAnalyzer()

    text = (
        "I have always believed that failure is an important "
        "part of success. Every difficult experience has taught "
        "me something that I could not have learned otherwise."
    )

    raw_result = analyzer.score_text(text)

    features = calculate_perplexity_features(
        raw_result.tokens
    )

    print()
    print("=" * 70)
    print("PROVENANCE — LANGUAGE MODEL ANALYSIS")
    print("=" * 70)

    print(f"Text:                {raw_result.text}")
    print(f"Tokens:              {features.token_count}")
    print(f"Mean token NLL:      {features.mean_nll:.4f}")
    print(f"Perplexity:          {features.perplexity:.4f}")
    print(f"NLL standard dev:    {features.nll_std:.4f}")
    print(f"NLL minimum:         {features.nll_min:.4f}")
    print(f"NLL maximum:         {features.nll_max:.4f}")
    print(f"NLL median:          {features.nll_median:.4f}")
    print(f"NLL 90th percentile: {features.nll_p90:.4f}")

    print()
    print("TOKEN ANALYSIS")
    print("-" * 70)

    for index, token in enumerate(
        raw_result.tokens,
        start=1,
    ):
        print(
            f"{index:3}. "
            f"{token.token!r:20} "
            f"probability={token.probability:.6f} "
            f"NLL={token.negative_log_likelihood:.4f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()