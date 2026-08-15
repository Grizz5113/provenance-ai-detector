from __future__ import annotations

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = "data/processed/features.csv"

META_COLUMNS = {
    "essay_id",
    "label",
    "topic",
    "source",
    "source_group",
    "ai_intervention_level",
    "edit_ratio",
}


COUNT_FEATURES = [
    "unique_token_count",
    "unique_bigrams",
    "repeated_bigrams",
    "unique_trigrams",
    "repeated_trigrams",
    "punctuation_count",
    "comma_count",
    "period_count",
    "semicolon_count",
    "colon_count",
    "question_count",
    "exclamation_count",
    "parentheses_count",
    "quotation_count",
    "contraction_count",
    "sentence_count",
]


LM_FEATURES = [
    "mean_nll",
    "perplexity",
    "nll_std",
    "nll_min",
    "nll_max",
    "nll_median",
    "nll_p90",
]


STYLE_FEATURES = [
    "sentence_length_cv",
    "type_token_ratio",
    "hapax_ratio",
    "vocabulary_entropy",
    "repeated_token_ratio",
    "bigram_repetition_ratio",
    "trigram_repetition_ratio",
    "punctuation_density",
]


def build_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate(
    name,
    X,
    y,
    groups,
    seeds,
):
    results = []

    print("=" * 70)
    print(name)
    print("=" * 70)

    for seed in seeds:

        cv = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )

        predictions = cross_val_predict(
            build_model(),
            X,
            y,
            groups=groups,
            cv=cv,
        )

        accuracy = accuracy_score(
            y,
            predictions,
        )

        macro_f1 = f1_score(
            y,
            predictions,
            average="macro",
        )

        hybrid_mask = y == "hybrid"

        hybrid_recall = (
            (
                predictions[hybrid_mask]
                == "hybrid"
            ).sum()
            / hybrid_mask.sum()
        )

        results.append(
            {
                "seed": seed,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "hybrid_recall": hybrid_recall,
            }
        )

        print(
            f"Seed {seed:>3}: "
            f"accuracy={accuracy:.4f} | "
            f"macro_f1={macro_f1:.4f} | "
            f"hybrid_recall={hybrid_recall:.4f}"
        )

    result_df = pd.DataFrame(results)

    print()
    print("Mean ± standard deviation")

    for metric in [
        "accuracy",
        "macro_f1",
        "hybrid_recall",
    ]:
        print(
            f"{metric:15s}: "
            f"{result_df[metric].mean():.4f} ± "
            f"{result_df[metric].std(ddof=1):.4f}"
        )

    print()

    return result_df


def add_normalized_features(df):
    result = df.copy()

    tokens = result["token_count"].clip(lower=1)

    for feature in COUNT_FEATURES:

        normalized_name = (
            f"{feature}_per_100_tokens"
        )

        result[normalized_name] = (
            result[feature]
            / tokens
            * 100.0
        )

    return result


def main():

    print("=" * 70)
    print("PROVENANCE — NORMALIZED FEATURE EXPERIMENT")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    # ------------------------------------------------------------
    # IMPORTANT:
    # Capture the canonical 40 features BEFORE creating anything.
    # ------------------------------------------------------------

    original_features = [
        c for c in df.columns
        if c not in META_COLUMNS
    ]

    print(
        f"Samples: {len(df)}"
    )

    print(
        f"Original features: "
        f"{len(original_features)}"
    )

    if len(original_features) != 40:
        raise RuntimeError(
            "Expected exactly 40 canonical "
            f"features, found {len(original_features)}."
        )

    # ------------------------------------------------------------
    # Create normalized dataframe separately.
    # ------------------------------------------------------------

    normalized_df = add_normalized_features(df)

    normalized_features = [
        "token_count",
        *LM_FEATURES,
        *STYLE_FEATURES,
        *[
            f"{feature}_per_100_tokens"
            for feature in COUNT_FEATURES
        ],
    ]

    # Remove accidental duplicates.
    normalized_features = list(
        dict.fromkeys(normalized_features)
    )

    # ------------------------------------------------------------
    # Raw + normalized:
    # canonical 40 + normalized rates
    # ------------------------------------------------------------

    raw_plus_normalized_features = [
        *original_features,
        *[
            f"{feature}_per_100_tokens"
            for feature in COUNT_FEATURES
        ],
    ]

    raw_plus_normalized_features = list(
        dict.fromkeys(
            raw_plus_normalized_features
        )
    )

    print(
        f"Normalized + LM features: "
        f"{len(normalized_features)}"
    )

    print(
        f"Raw + normalized features: "
        f"{len(raw_plus_normalized_features)}"
    )

    print()

    y = df["label"]
    groups = df["source_group"]

    seeds = [
        42,
        7,
        21,
        100,
        123,
    ]

    # ------------------------------------------------------------
    # A — Canonical baseline
    # ------------------------------------------------------------

    evaluate(
        "A — ORIGINAL FEATURES",
        df[original_features],
        y,
        groups,
        seeds,
    )

    # ------------------------------------------------------------
    # B — Normalized counts + LM/style features
    # ------------------------------------------------------------

    evaluate(
        "B — NORMALIZED + LM FEATURES",
        normalized_df[
            normalized_features
        ],
        y,
        groups,
        seeds,
    )

    # ------------------------------------------------------------
    # C — Original + normalized rates
    # ------------------------------------------------------------

    evaluate(
        "C — ORIGINAL + NORMALIZED FEATURES",
        normalized_df[
            raw_plus_normalized_features
        ],
        y,
        groups,
        seeds,
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
