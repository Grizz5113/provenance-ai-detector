from backend.app.features.repetition import (
    calculate_repetition_features,
)


def main() -> None:

    text = (
        "I have always believed that failure is an important "
        "part of success. Every difficult experience has taught "
        "me something that I could not have learned otherwise. "
        "These experiences changed the way I approach challenges."
    )

    features = calculate_repetition_features(text)

    print()
    print("=" * 60)
    print("PROVENANCE — REPETITION FEATURES")
    print("=" * 60)

    print(
        f"Token count:                 "
        f"{features.token_count}"
    )

    print(
        f"Unique bigrams:               "
        f"{features.unique_bigrams}"
    )

    print(
        f"Repeated bigrams:             "
        f"{features.repeated_bigrams}"
    )

    print(
        f"Bigram repetition ratio:      "
        f"{features.bigram_repetition_ratio:.4f}"
    )

    print(
        f"Unique trigrams:              "
        f"{features.unique_trigrams}"
    )

    print(
        f"Repeated trigrams:            "
        f"{features.repeated_trigrams}"
    )

    print(
        f"Trigram repetition ratio:     "
        f"{features.trigram_repetition_ratio:.4f}"
    )

    print(
        f"Most common bigram count:     "
        f"{features.most_common_bigram_count}"
    )

    print(
        f"Most common trigram count:    "
        f"{features.most_common_trigram_count}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()