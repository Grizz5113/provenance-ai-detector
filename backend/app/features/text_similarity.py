from __future__ import annotations

from difflib import SequenceMatcher


def calculate_edit_ratio(
    original: str,
    modified: str,
) -> float:
    """
    Estimate how much the modified text differs
    from the original.

    Returns a value between 0 and 1.

    0.0 = identical
    1.0 = completely different
    """

    original_tokens = original.split()
    modified_tokens = modified.split()

    if not original_tokens:
        return 1.0

    matcher = SequenceMatcher(
        None,
        original_tokens,
        modified_tokens,
    )

    similarity = matcher.ratio()

    return 1.0 - similarity