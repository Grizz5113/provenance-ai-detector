from __future__ import annotations

from dataclasses import dataclass

import nltk


@dataclass
class Sentence:
    index: int
    text: str


def split_sentences(text: str) -> list[Sentence]:
    """
    Split an essay into sentences while preserving their order.
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    raw_sentences = nltk.sent_tokenize(text)

    sentences: list[Sentence] = []

    for index, sentence in enumerate(raw_sentences, start=1):

        cleaned = sentence.strip()

        if not cleaned:
            continue

        sentences.append(
            Sentence(
                index=index,
                text=cleaned,
            )
        )

    return sentences