from __future__ import annotations

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
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


def evaluate_model(
    name,
    model,
    X,
    y,
    groups,
    df,
    cv,
):
    print("=" * 70)
    print(name)
    print("=" * 70)

    predictions = cross_val_predict(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
    )

    accuracy = accuracy_score(y, predictions)

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print()

    print("Classification report:")
    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y,
            predictions,
            labels=[
                "human",
                "ai",
                "hybrid",
            ],
        )
    )

    result = df[
        [
            "essay_id",
            "label",
            "source_group",
            "ai_intervention_level",
        ]
    ].copy()

    result["predicted"] = predictions
    result["correct"] = (
        result["label"] == result["predicted"]
    )

    errors = result[~result["correct"]]

    print()
    print("Misclassified essays:")

    if errors.empty:
        print("None")
    else:
        print(
            errors.to_string(index=False)
        )

    output_file = (
        f"data/processed/"
        f"predictions_{name.lower().replace(' ', '_')}.csv"
    )

    result.to_csv(
        output_file,
        index=False,
    )

    print()
    print(f"Predictions saved to: {output_file}")
    print()

    return predictions


def main() -> None:

    print("=" * 70)
    print("PROVENANCE — GROUP-AWARE CLASSIFIER EVALUATION")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    feature_columns = [
        column
        for column in df.columns
        if column not in META_COLUMNS
    ]

    X = df[feature_columns]
    y = df["label"]
    groups = df["source_group"]

    print(f"Samples:  {len(df)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Groups:   {groups.nunique()}")
    print()

    print("Class distribution:")
    print(y.value_counts())
    print()

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    models = {
        "Logistic Regression": Pipeline(
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
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
        ),
    }

    for name, model in models.items():
        evaluate_model(
            name=name,
            model=model,
            X=X,
            y=y,
            groups=groups,
            df=df,
            cv=cv,
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
