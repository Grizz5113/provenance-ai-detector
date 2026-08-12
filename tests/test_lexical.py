from backend.app.features.lexical import (
    calculate_lexical_features,
)


def main() -> None:

    text = (
        "I have always believed that failure is an important "
        "part of success. Every difficult experience has taught "
        "me something that I could not have learned otherwise. "
        "These experiences changed the way I approach challenges."
    )

    features = calculate_lexical_features(text)

    print()
    print("=" * 60)
    print("PROVENANCE — LEXICAL FEATURES")
    print("=" * 60)

    print(
        f"Token count:              "
        f"{features.token_count}"
    )

    print(
        f"Unique tokens:            "
        f"{features.unique_token_count}"
    )

    print(
        f"Type-token ratio:         "
        f"{features.type_token_ratio:.4f}"
    )

    print(
        f"Hapax ratio:              "
        f"{features.hapax_ratio:.4f}"
    )

    print(
        f"Vocabulary entropy:       "
        f"{features.vocabulary_entropy:.4f}"
    )

    print(
        f"Repeated token ratio:     "
        f"{features.repeated_token_ratio:.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()