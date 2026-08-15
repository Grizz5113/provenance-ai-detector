from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features_persuade_augmented.csv"
)
MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "artifacts"

MODEL_FILE = MODEL_DIR / "provenance_detector.joblib"
METADATA_FILE = MODEL_DIR / "model_metadata.json"


META_COLUMNS = {
    "essay_id",
    "label",
    "topic",
    "source",
    "source_group",
    "ai_intervention_level",
    "edit_ratio",
}


# Features selected from the individual-ablation experiment.
# Final feature selection:
# all 40 extracted features are retained.
REMOVED_FEATURES = set()


def main() -> None:

    print("=" * 70)
    print("PROVENANCE — TRAIN FINAL DETECTOR")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    all_features = [
        column
        for column in df.columns
        if column not in META_COLUMNS
    ]

    feature_columns = [
        column
        for column in all_features
        if column not in REMOVED_FEATURES
    ]

    X = df[feature_columns]
    y = df["label"]

    print(f"Samples:          {len(df)}")
    print(f"All features:     {len(all_features)}")
    print(f"Final features:   {len(feature_columns)}")
    print()

    print("Class distribution:")
    print(y.value_counts())
    print()

    print("Final feature set:")
    for index, feature in enumerate(feature_columns, start=1):
        print(f"{index:2d}. {feature}")

    print()

    model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
)

    print("Training final model...")

    model.fit(
        X,
        y,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    metadata = {
        "model_type": "RandomForestClassifier",
"parameters": {
    "n_estimators": 300,
    "class_weight": "balanced",
    "random_state": 42,
},
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "classes": list(model.classes_),
        "training_samples": len(df),
        "class_distribution": y.value_counts().to_dict(),
        "removed_features": sorted(
            REMOVED_FEATURES
        ),
        "selection_basis": (
    "Random Forest selected using 5-fold StratifiedGroupKFold "
    "evaluation on the 406-sample augmented dataset"
),
    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("-" * 70)
    print("FINAL MODEL TRAINED")
    print("-" * 70)
    print(f"Model:            {MODEL_FILE}")
    print(f"Metadata:         {METADATA_FILE}")
    print(f"Features:         {len(feature_columns)}")
    print(f"Classes:          {list(model.classes_)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
