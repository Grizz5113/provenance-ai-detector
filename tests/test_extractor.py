from backend.app.detector.language_model import LanguageModelAnalyzer
from backend.app.features.extractor import EssayFeatureExtractor


def main() -> None:

    text = (
        "I have always believed that failure is an important "
        "part of success. Every difficult experience has taught "
        "me something that I could not have learned otherwise. "
        "These experiences changed the way I approach challenges."
    )

    print("Loading language model...")

    language_model = LanguageModelAnalyzer()

    extractor = EssayFeatureExtractor(
        language_model
    )

    features = extractor.extract(text)

    print()
    print("=" * 70)
    print("PROVENANCE — UNIFIED FEATURE VECTOR")
    print("=" * 70)

    print()
    print("LANGUAGE MODEL")
    print("-" * 70)

    print(f"Token count:          {features.token_count}")
    print(f"Mean NLL:             {features.mean_nll:.4f}")
    print(f"Perplexity:           {features.perplexity:.4f}")
    print(f"NLL std:              {features.nll_std:.4f}")
    print(f"NLL median:           {features.nll_median:.4f}")
    print(f"NLL P90:              {features.nll_p90:.4f}")

    print()
    print("RHYTHM")
    print("-" * 70)

    print(f"Sentence count:       {features.sentence_count}")
    print(
        f"Mean sentence length: "
        f"{features.mean_sentence_length:.2f}"
    )
    print(
        f"Sentence length std:  "
        f"{features.sentence_length_std:.2f}"
    )
    print(
        f"Sentence length CV:   "
        f"{features.sentence_length_cv:.4f}"
    )

    print()
    print("LEXICAL")
    print("-" * 70)

    print(
        f"Unique tokens:        "
        f"{features.unique_token_count}"
    )
    print(
        f"Type-token ratio:     "
        f"{features.type_token_ratio:.4f}"
    )
    print(
        f"Hapax ratio:          "
        f"{features.hapax_ratio:.4f}"
    )
    print(
        f"Vocabulary entropy:   "
        f"{features.vocabulary_entropy:.4f}"
    )

    print()
    print("REPETITION")
    print("-" * 70)

    print(
        f"Bigram repetition:    "
        f"{features.bigram_repetition_ratio:.4f}"
    )
    print(
        f"Trigram repetition:   "
        f"{features.trigram_repetition_ratio:.4f}"
    )

    print()
    print("PUNCTUATION")
    print("-" * 70)

    print(
        f"Punctuation density:  "
        f"{features.punctuation_density:.4f}"
    )
    print(
        f"Punctuation types:    "
        f"{features.punctuation_types}"
    )
    print(
        f"Comma count:          "
        f"{features.comma_count}"
    )
    print(
        f"Period count:         "
        f"{features.period_count}"
    )

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()