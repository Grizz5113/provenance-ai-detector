from backend.app.features.text_similarity import (
    calculate_edit_ratio,
)


def main() -> None:

    original = (
        "I struggled with mathematics "
        "during my first year."
    )

    identical = original

    modified = (
        "During my first year, "
        "I struggled with mathematics."
    )

    print(
        "Identical edit ratio:",
        f"{calculate_edit_ratio(original, identical):.4f}",
    )

    print(
        "Modified edit ratio:",
        f"{calculate_edit_ratio(original, modified):.4f}",
    )


if __name__ == "__main__":
    main()