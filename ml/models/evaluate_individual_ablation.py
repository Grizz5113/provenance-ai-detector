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


def evaluate_features(
    df,
    feature_columns,
    y,
    groups,
    seeds,
):
    results = []

    X = df[feature_columns]

    for seed in seeds:
        cv = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )

        model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
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
    print("PROVENANCE — INDIVIDUAL FEATURE ABLATION")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    feature_columns = [
        c
        for c in df.columns
        if c not in META_COLUMNS
    ]

    y = df["label"]
    groups = df["source_group"]

    seeds = [
        42,
        7,
        21,
        100,
        123,
    ]

    print(f"Samples:  {len(df)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Groups:   {groups.nunique()}")
    print()

    results = []

    # First evaluate the complete feature set.
    print(
        f"Evaluating: ALL FEATURES "
        f"({len(feature_columns)} features)"
    )

    baseline = evaluate_features(
        df=df,
        feature_columns=feature_columns,
        y=y,
        groups=groups,
        seeds=seeds,
    )

    results.append(
        {
            "removed_feature": "NONE",
            "feature_count": len(feature_columns),
            **baseline,
        }
    )

    # Remove one feature at a time.
    for feature in feature_columns:

        reduced_features = [
            c
            for c in feature_columns
            if c != feature
        ]

        print(
            f"Evaluating: WITHOUT {feature} "
            f"({len(reduced_features)} features)"
        )

        metrics = evaluate_features(
            df=df,
            feature_columns=reduced_features,
            y=y,
            groups=groups,
            seeds=seeds,
        )

        results.append(
            {
                "removed_feature": feature,
                "feature_count": len(reduced_features),
                **metrics,
            }
        )

    result_df = pd.DataFrame(results)

    baseline_accuracy = baseline["accuracy_mean"]
    baseline_f1 = baseline["macro_f1_mean"]
    baseline_hybrid = baseline["hybrid_recall_mean"]

    result_df["accuracy_delta"] = (
        result_df["accuracy_mean"]
        - baseline_accuracy
    )

    result_df["macro_f1_delta"] = (
        result_df["macro_f1_mean"]
        - baseline_f1
    )

    result_df["hybrid_recall_delta"] = (
        result_df["hybrid_recall_mean"]
        - baseline_hybrid
    )

    # A positive delta means removing the feature improved
    # performance. A negative delta means the feature helped.
    result_df = result_df.sort_values(
        "macro_f1_delta",
        ascending=False,
    )

    print()
    print("=" * 70)
    print("INDIVIDUAL ABLATION RESULTS")
    print("=" * 70)

    print(
        result_df[
            [
                "removed_feature",
                "feature_count",
                "accuracy_mean",
                "macro_f1_mean",
                "hybrid_recall_mean",
                "macro_f1_delta",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("MOST USEFUL FEATURES")
    print("=" * 70)

    # Most useful = removing them hurts performance the most.
    useful = result_df[
        result_df["removed_feature"] != "NONE"
    ].sort_values(
        "macro_f1_delta"
    )

    print(
        useful[
            [
                "removed_feature",
                "macro_f1_delta",
                "hybrid_recall_delta",
            ]
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("POTENTIALLY REDUNDANT FEATURES")
    print("=" * 70)

    # If removing a feature improves performance, it may be
    # noisy or redundant.
    redundant = result_df[
        result_df["removed_feature"] != "NONE"
    ].sort_values(
        "macro_f1_delta",
        ascending=False,
    )

    print(
        redundant[
            [
                "removed_feature",
                "macro_f1_delta",
                "hybrid_recall_delta",
            ]
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    output_file = (
        "data/processed/"
        "individual_feature_ablation_results.csv"
    )

    result_df.to_csv(
        output_file,
        index=False,
    )

    print()
    print(
        f"Results saved to: {output_file}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
