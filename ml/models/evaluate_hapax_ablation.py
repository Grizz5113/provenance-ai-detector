from __future__ import annotations

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = "data/processed/features_augmented.csv"

META_COLUMNS = {
    "essay_id",
    "label",
    "topic",
    "source",
    "source_group",
    "ai_intervention_level",
    "edit_ratio",
}

REMOVED_FEATURES = {
    "question_count",
    "quotation_count",
    "semicolon_count",
    "median_sentence_length",
    "exclamation_count",
    "sentence_length_cv",
    "punctuation_density",
    "unique_bigrams",
    "repeated_bigrams",
    "bigram_repetition_ratio",
    "unique_trigrams",
    "repeated_trigrams",
    "trigram_repetition_ratio",
    "most_common_bigram_count",
    "most_common_trigram_count",
}

SEEDS = [42, 7, 21, 100, 123]


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


def evaluate(name, features, df):

    X = df[features]
    y = df["label"]
    groups = df["source_group"]

    results = []

    for seed in SEEDS:

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
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "hybrid_recall": hybrid_recall,
            }
        )

    result_df = pd.DataFrame(results)

    return {
        "accuracy_mean": result_df["accuracy"].mean(),
        "accuracy_std": result_df["accuracy"].std(ddof=1),
        "macro_f1_mean": result_df["macro_f1"].mean(),
        "macro_f1_std": result_df["macro_f1"].std(ddof=1),
        "hybrid_recall_mean": result_df["hybrid_recall"].mean(),
        "hybrid_recall_std": result_df["hybrid_recall"].std(ddof=1),
    }


def main():

    print("=" * 70)
    print("PROVENANCE — HAPAX RATIO 3-CLASS ABLATION")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    all_features = [
        column
        for column in df.columns
        if column not in META_COLUMNS
    ]

    current_features = [
        feature
        for feature in all_features
        if feature not in REMOVED_FEATURES
    ]

    without_hapax = [
        feature
        for feature in current_features
        if feature != "hapax_ratio"
    ]

    print(
        f"Current features:       {len(current_features)}"
    )

    print(
        f"Without hapax_ratio:    {len(without_hapax)}"
    )

    print()

    experiments = [
        (
            "CURRENT 25 FEATURES",
            current_features,
        ),
        (
            "WITHOUT HAPAX_RATIO",
            without_hapax,
        ),
    ]

    results = []

    for name, features in experiments:

        print(
            f"Evaluating: {name} "
            f"({len(features)} features)"
        )

        metrics = evaluate(
            name,
            features,
            df,
        )

        results.append(
            {
                "experiment": name,
                "features": len(features),
                **metrics,
            }
        )

        print(
            f"  Accuracy:       "
            f"{metrics['accuracy_mean']:.4f}"
            f" ± {metrics['accuracy_std']:.4f}"
        )

        print(
            f"  Macro F1:       "
            f"{metrics['macro_f1_mean']:.4f}"
            f" ± {metrics['macro_f1_std']:.4f}"
        )

        print(
            f"  Hybrid recall:  "
            f"{metrics['hybrid_recall_mean']:.4f}"
            f" ± {metrics['hybrid_recall_std']:.4f}"
        )

        print()

    result_df = pd.DataFrame(results)

    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(
        result_df.to_string(
            index=False
        )
    )

    baseline = result_df.iloc[0]
    modified = result_df.iloc[1]

    print()
    print("CHANGE")
    print("-" * 70)

    print(
        f"Accuracy:      "
        f"{modified['accuracy_mean'] - baseline['accuracy_mean']:+.4f}"
    )

    print(
        f"Macro F1:      "
        f"{modified['macro_f1_mean'] - baseline['macro_f1_mean']:+.4f}"
    )

    print(
        f"Hybrid recall: "
        f"{modified['hybrid_recall_mean'] - baseline['hybrid_recall_mean']:+.4f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()