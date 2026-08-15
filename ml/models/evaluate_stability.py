
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


def main():
    print("=" * 70)
    print("PROVENANCE — GROUPED CV STABILITY")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    feature_columns = [
        c for c in df.columns
        if c not in META_COLUMNS
    ]

    X = df[feature_columns]
    y = df["label"]
    groups = df["source_group"]

    seeds = [42, 7, 21, 100, 123]

    results = []

    for seed in seeds:
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

        print(
            f"Seed {seed:>3}: "
            f"accuracy={accuracy:.4f} | "
            f"macro_f1={macro_f1:.4f} | "
            f"hybrid_recall={hybrid_recall:.4f}"
        )

    result_df = pd.DataFrame(results)

    print()
    print("=" * 70)
    print("STABILITY SUMMARY")
    print("=" * 70)

    for metric in [
        "accuracy",
        "macro_f1",
        "hybrid_recall",
    ]:
        mean = result_df[metric].mean()
        std = result_df[metric].std(ddof=1)

        print(
            f"{metric:15s}: "
            f"{mean:.4f} ± {std:.4f}"
        )

    print()
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
