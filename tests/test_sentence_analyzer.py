from backend.app.detector.language_model import LanguageModelAnalyzer
from backend.app.detector.sentence_analyzer import SentenceAnalyzer


def main() -> None:

    text = (
        "I have always believed that failure is an important "
        "part of success. Every difficult experience has taught "
        "me something that I could not have learned otherwise. "
        "These experiences changed the way I approach challenges."
    )

    print("Loading language model...")

    language_model = LanguageModelAnalyzer()

    analyzer = SentenceAnalyzer(language_model)

    results = analyzer.analyze(text)

    print()
    print("=" * 70)
    print("PROVENANCE — SENTENCE LEVEL ANALYSIS")
    print("=" * 70)

    print(f"Sentences analyzed: {len(results)}")
    print()

    for result in results:

        sentence = result.sentence
        features = result.features

        print(f"Sentence {sentence.index}")
        print("-" * 70)
        print(sentence.text)
        print()

        print(f"Tokens:              {features.token_count}")
        print(f"Mean NLL:            {features.mean_nll:.4f}")
        print(f"Perplexity:          {features.perplexity:.4f}")
        print(f"NLL std:             {features.nll_std:.4f}")
        print(f"NLL median:          {features.nll_median:.4f}")
        print(f"NLL 90th percentile: {features.nll_p90:.4f}")
        print()

    print("=" * 70)


if __name__ == "__main__":
    main()  