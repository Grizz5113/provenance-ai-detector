from backend.app.features.rhythm import calculate_rhythm_features
from backend.app.features.sentence import split_sentences


def main() -> None:

    text = (
        "I have always believed that failure is an important "
        "part of success. Every difficult experience has taught "
        "me something that I could not have learned otherwise. "
        "These experiences changed the way I approach challenges."
    )

    sentences = split_sentences(text)

    features = calculate_rhythm_features(sentences)

    print()
    print("=" * 60)
    print("PROVENANCE — RHYTHM FEATURES")
    print("=" * 60)

    print(f"Sentences:                 {features.sentence_count}")
    print(
        f"Mean sentence length:      "
        f"{features.mean_sentence_length:.2f}"
    )
    print(
        f"Median sentence length:    "
        f"{features.median_sentence_length:.2f}"
    )
    print(
        f"Sentence length std:       "
        f"{features.std_sentence_length:.2f}"
    )
    print(
        f"Minimum sentence length:   "
        f"{features.min_sentence_length}"
    )
    print(
        f"Maximum sentence length:   "
        f"{features.max_sentence_length}"
    )
    print(
        f"Coefficient of variation:  "
        f"{features.coefficient_of_variation:.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
    