from backend.app.features.sentence import split_sentences


def main() -> None:

    text = (
        "I have always believed that failure is an important "
        "part of success. Every difficult experience has taught "
        "me something that I could not have learned otherwise. "
        "These experiences changed the way I approach challenges."
    )

    sentences = split_sentences(text)

    print()
    print("=" * 60)
    print("PROVENANCE — SENTENCE SEGMENTATION")
    print("=" * 60)

    print(f"Total sentences: {len(sentences)}")
    print()

    for sentence in sentences:
        print(f"{sentence.index}. {sentence.text}")

    print("=" * 60)


if __name__ == "__main__":
    main()