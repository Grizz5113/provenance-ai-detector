from __future__ import annotations

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = "data/processed/features_persuade_augmented.csv"

META_COLUMNS = {
    "essay_id",
    "label",
    "topic",
    "source",
    "source_group",
    "ai_intervention_level",
    "edit_ratio",
}

# Features whose removal improved the individual-ablation experiment.
REDUNDANT_FEATURES = {
    "question_count",
    "quotation_count",
    "semicolon_count",
    "median_sentence_length",
    "exclamation_count",
    "sentence_length_cv",
    "punctuation_density",
}

# Features that showed essentially zero contribution in the
# individual-ablation experiment.
ZERO_CONTRIBUTION_FEATURES = {
    "repeated_trigrams",
    "trigram_repetition_ratio",
    "bigram_repetition_ratio",
    "most_common_bigram_count",
}


SEEDS = [42, 7, 21, 100, 123]


def evaluate(
    df,
    feature_columns,
    y,
    groups,
):
    X = df[feature_columns]

    results = []

    for seed in SEEDS:
        cv = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )

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

        results.append(
            {
                "seed": seed,
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


def main() -> None:

    print("=" * 70)
    print("PROVENANCE — REDUCED FEATURE EXPERIMENT")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    all_features = [
        c
        for c in df.columns
        if c not in META_COLUMNS
    ]

    reduced_features = [
        c
        for c in all_features
        if c not in REDUNDANT_FEATURES
    ]

    aggressive_features = [
        c
        for c in reduced_features
        if c not in ZERO_CONTRIBUTION_FEATURES
    ]

    y = df["label"]
    groups = df["source_group"]

    experiments = {
        "A — ALL FEATURES": all_features,
        "B — REDUCED": reduced_features,
        "C — AGGRESSIVE REDUCED": aggressive_features,
    }

    print(f"Samples: {len(df)}")
    print(f"Groups:   {groups.nunique()}")
    print()

    results = []

    for name, features in experiments.items():

        print("=" * 70)
        print(name)
        print("=" * 70)

        print(f"Features: {len(features)}")
        print()

        metrics = evaluate(
            df=df,
            feature_columns=features,
            y=y,
            groups=groups,
        )

        print(
            f"Accuracy:       "
            f"{metrics['accuracy_mean']:.4f} "
            f"± {metrics['accuracy_std']:.4f}"
        )

        print(
            f"Macro F1:       "
            f"{metrics['macro_f1_mean']:.4f} "
            f"± {metrics['macro_f1_std']:.4f}"
        )

        print(
            f"Hybrid recall:  "
            f"{metrics['hybrid_recall_mean']:.4f} "
            f"± {metrics['hybrid_recall_std']:.4f}"
        )

        results.append(
            {
                "experiment": name,
                "feature_count": len(features),
                **metrics,
            }
        )

    result_df = pd.DataFrame(results)

    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(
        result_df[
            [
                "experiment",
                "feature_count",
                "accuracy_mean",
                "accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "hybrid_recall_mean",
                "hybrid_recall_std",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    output_file = (
        "data/processed/"
        "reduced_feature_results.csv"
    )

    result_df.to_csv(
        output_file,
        index=False,
    )

    print()
    print(f"Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
