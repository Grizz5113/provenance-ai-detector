from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from backend.app.detector.language_model import LanguageModelAnalyzer
from backend.app.features.extractor import EssayFeatureExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "artifacts"
    / "provenance_detector.joblib"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "artifacts"
    / "model_metadata.json"
)


class ProvenanceDetector:

    def __init__(self) -> None:

        self.model = joblib.load(
            MODEL_FILE
        )

        with METADATA_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)

        self.feature_columns = metadata[
            "features"
        ]

        # Verify that the trained model and
        # metadata use the exact same feature order.
        model_features = list(
            self.model.feature_names_in_
        )

        if model_features != self.feature_columns:
            raise RuntimeError(
                "Model feature order mismatch between "
                "trained model and metadata."
            )

        self.language_model = (
            LanguageModelAnalyzer()
        )

        self.extractor = EssayFeatureExtractor(
            language_model=self.language_model
        )
    def predict(
        self,
        text: str,
    ) -> dict:

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        features = self.extractor.extract(
            text
        )

        feature_dict = features.to_dict()

        missing = [
            feature
            for feature in self.feature_columns
            if feature not in feature_dict
        ]

        if missing:
            raise RuntimeError(
                "Missing features: "
                + ", ".join(missing)
            )

        X = pd.DataFrame(
            [
                {
                    feature: feature_dict[
                        feature
                    ]
                    for feature in self.feature_columns
                }
            ]
        )

        prediction = self.model.predict(X)[0]

        probabilities = (
            self.model.predict_proba(X)[0]
        )

        classes = self.model.classes_

        probability_map = {
            class_name: float(probability)
            for class_name, probability
            in zip(
                classes,
                probabilities,
            )
        }

        return {
            "prediction": prediction,
            "probabilities": probability_map,
            "features_used": len(
                self.feature_columns
            ),
        }


def main() -> None:

    detector = ProvenanceDetector()

    text = """
    Technology has changed the way people
    communicate, learn, and work. While
    these changes provide many benefits,
    they also create challenges that
    society must address carefully.
    """

    result = detector.predict(text)

    print("=" * 70)
    print("PROVENANCE — DETECTOR INFERENCE")
    print("=" * 70)

    print(
        f"Prediction: {result['prediction']}"
    )

    print()
    print("Probabilities:")

    for label, probability in (
        result["probabilities"].items()
    ):
        print(
            f"  {label:8s}: "
            f"{probability:.4f}"
        )

    print()
    print(
        f"Features used: "
        f"{result['features_used']}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
