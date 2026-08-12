from backend.app.features.punctuation import (
    calculate_punctuation_features,
)


def main() -> None:

    text = (
        "I have always believed that failure is an important "
        "part of success. Every difficult experience has taught "
        "me something that I could not have learned otherwise. "
        "These experiences changed the way I approach challenges."
    )

    features = calculate_punctuation_features(text)

    print()
    print("=" * 60)
    print("PROVENANCE — PUNCTUATION FEATURES")
    print("=" * 60)

    print(f"Characters:             {features.character_count}")
    print(f"Words:                  {features.word_count}")
    print(f"Punctuation count:      {features.punctuation_count}")
    print(f"Punctuation density:    {features.punctuation_density:.4f}")
    print(f"Punctuation types:      {features.punctuation_types}")

    print()
    print("Punctuation breakdown")
    print("-" * 60)

    print(f"Commas:                 {features.comma_count}")
    print(f"Periods:                {features.period_count}")
    print(f"Semicolons:             {features.semicolon_count}")
    print(f"Colons:                 {features.colon_count}")
    print(f"Questions:              {features.question_count}")
    print(f"Exclamations:           {features.exclamation_count}")
    print(f"Parentheses:            {features.parentheses_count}")
    print(f"Quotes:                 {features.quotation_count}")
    print(f"Contractions:           {features.contraction_count}")

    print("=" * 60)


if __name__ == "__main__":
    main()