from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.app.detector.language_model import LanguageModelAnalyzer
from backend.app.features.extractor import EssayFeatureExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_augmented.csv"
)
MODEL_METADATA = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "artifacts"
    / "model_metadata.json"
)

HC3_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "hc3_external"
)

METADATA_FILE = HC3_DIR / "metadata.csv"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hc3_binary_predictions.csv"
)


def main():

    print("=" * 70)
    print("PROVENANCE — HC3 BINARY AI/HUMAN EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load metadata / feature definition
    # ------------------------------------------------------------

    metadata = pd.read_csv(METADATA_FILE)

    with MODEL_METADATA.open(
        "r",
        encoding="utf-8",
    ) as f:
        model_metadata = json.load(f)

    features = model_metadata["features"]

    print(f"Training features: {len(features)}")
    print(f"HC3 samples:       {len(metadata)}")
    print()

       # ------------------------------------------------------------
    # Training data — ONLY AI + HUMAN
    # ------------------------------------------------------------

    train_df = pd.read_csv(TRAIN_FEATURES)

    print(f"Training dataset: {TRAIN_FEATURES}")
    print(f"Training rows:     {len(train_df)}")

    train_df = train_df[
        train_df["label"].isin(["ai", "human"])
    ].copy()

    print("Binary training distribution:")
    print(train_df["label"].value_counts())
    print()

    X_train = train_df[features]
    y_train = train_df["label"]

    # ------------------------------------------------------------
    # Train binary model
    # ------------------------------------------------------------
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

    print("Training binary AI/Human model...")

    model.fit(
        X_train,
        y_train,
    )

    print("Binary model trained.")
    print()

    # ------------------------------------------------------------
    # Load language model + feature extractor
    # ------------------------------------------------------------

    print("Loading language model...")

    language_model = LanguageModelAnalyzer()

    extractor = EssayFeatureExtractor(
        language_model=language_model,
    )

    # ------------------------------------------------------------
    # Evaluate HC3
    # ------------------------------------------------------------

    predictions = []

    print()
    print("Evaluating HC3 samples...")

    for index, row in metadata.iterrows():

        essay_id = row["essay_id"]
        actual = row["label"]

        essay_file = (
            HC3_DIR
            / actual
            / f"{essay_id}.txt"
        )

        if not essay_file.exists():
            raise RuntimeError(
                f"Missing essay file: {essay_file}"
            )

        text = essay_file.read_text(
            encoding="utf-8"
        )

        extracted = extractor.extract(text)

        feature_dict = extracted.to_dict()

        missing = [
            feature
            for feature in features
            if feature not in feature_dict
        ]

        if missing:
            raise RuntimeError(
                f"{essay_id}: missing features: "
                + ", ".join(missing)
            )

        X = pd.DataFrame(
            [
                {
                    feature: feature_dict[feature]
                    for feature in features
                }
            ]
        )

        prediction = model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]

        probability_map = dict(
            zip(
                model.classes_,
                probabilities,
            )
        )

        predictions.append(
            {
                "essay_id": essay_id,
                "actual": actual,
                "predicted": prediction,
                "correct": (
                    actual == prediction
                ),
                "ai_probability": probability_map.get(
                    "ai",
                    0.0,
                ),
                "human_probability": probability_map.get(
                    "human",
                    0.0,
                ),
            }
        )

        print(
            f"[{index + 1:2d}/{len(metadata)}] "
            f"{essay_id:25s} "
            f"actual={actual:5s} "
            f"predicted={prediction:5s}"
        )

    # ------------------------------------------------------------
    # Results
    # ------------------------------------------------------------

    result_df = pd.DataFrame(
        predictions
    )

    y_true = result_df["actual"]
    y_pred = result_df["predicted"]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    print()
    print("=" * 70)
    print("HC3 BINARY RESULTS")
    print("=" * 70)

    print(
        f"Samples:   {len(result_df)}"
    )

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Macro F1:  {macro_f1:.4f}"
    )

    print()
    print("Classification report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=["human", "ai"],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")

    print(
        pd.DataFrame(
            confusion_matrix(
                y_true,
                y_pred,
                labels=[
                    "human",
                    "ai",
                ],
            ),
            index=[
                "actual_human",
                "actual_ai",
            ],
            columns=[
                "pred_human",
                "pred_ai",
            ],
        )
        .to_string()
    )

    print()
    print("Predictions by class:")

    print(
        pd.crosstab(
            result_df["actual"],
            result_df["predicted"],
        ).to_string()
    )

    print()
    print("Misclassified samples:")

    errors = result_df[
        ~result_df["correct"]
    ]

    if errors.empty:
        print("None")
    else:
        print(
            errors[
                [
                    "essay_id",
                    "actual",
                    "predicted",
                    "ai_probability",
                    "human_probability",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Results saved to: {OUTPUT_FILE}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
