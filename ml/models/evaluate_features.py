
from __future__ import annotations

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
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

# Features that are strongly dependent on document length.
LENGTH_FEATURES = {
    "token_count",
    "sentence_count",
    "unique_token_count",
    "unique_bigrams",
    "unique_trigrams",
    "punctuation_count",
    "comma_count",
    "period_count",
    "semicolon_count",
    "colon_count",
    "repeated_bigrams",
    "repeated_trigrams",
}

# Normalized structural + language-model features.
NORMALIZED_LM_FEATURES = {
    "mean_nll",
    "perplexity",
    "nll_std",
    "nll_min",
    "nll_max",
    "nll_median",
    "nll_p90",
    "mean_sentence_length",
    "median_sentence_length",
    "sentence_length_std",
    "sentence_length_cv",
    "min_sentence_length",
    "max_sentence_length",
    "type_token_ratio",
    "hapax_ratio",
    "vocabulary_entropy",
    "repeated_token_ratio",
    "bigram_repetition_ratio",
    "trigram_repetition_ratio",
    "most_common_bigram_count",
    "most_common_trigram_count",
    "punctuation_density",
    "punctuation_types",
    "question_count",
    "exclamation_count",
    "parentheses_count",
    "quotation_count",
    "contraction_count",
}


def evaluate(
    name: str,
    features: list[str],
    df: pd.DataFrame,
    cv: StratifiedGroupKFold,
) -> None:

    X = df[features]
    y = df["label"]
    groups = df["source_group"]

    model = Pipeline(
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

    predictions = cross_val_predict(
        model,
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
        (predictions[hybrid_mask] == "hybrid").sum()
        / hybrid_mask.sum()
    )

    print("=" * 70)
    print(name)
    print("=" * 70)
    print(f"Features:        {len(features)}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Hybrid recall:   {hybrid_recall:.4f}")
    print()


def main() -> None:

    print("=" * 70)
    print("PROVENANCE — FEATURE SET COMPARISON")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    all_features = [
        column
        for column in df.columns
        if column not in META_COLUMNS
    ]

    length_independent = [
        column
        for column in all_features
        if column not in LENGTH_FEATURES
    ]

    normalized_lm = [
        column
        for column in all_features
        if column in NORMALIZED_LM_FEATURES
    ]

    print(f"Samples: {len(df)}")
    print(f"All features: {len(all_features)}")
    print(f"Length-independent: {len(length_independent)}")
    print(f"Normalized + LM: {len(normalized_lm)}")
    print()

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    evaluate(
        "A — ALL FEATURES",
        all_features,
        df,
        cv,
    )

    evaluate(
        "B — LENGTH-INDEPENDENT FEATURES",
        length_independent,
        df,
        cv,
    )

    evaluate(
        "C — NORMALIZED + LANGUAGE-MODEL FEATURES",
        normalized_lm,
        df,
        cv,
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
