from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backend.app.detector.language_model import LanguageModelAnalyzer
from backend.app.features.extractor import EssayFeatureExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
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

HC3_METADATA = HC3_DIR / "metadata.csv"


def main():

    print("=" * 70)
    print("PROVENANCE — HC3 FEATURE DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load feature list
    # ------------------------------------------------------------

    with MODEL_METADATA.open(
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    features = metadata["features"]

    print(
        f"Features analyzed: {len(features)}"
    )

    # ------------------------------------------------------------
    # Training data
    # ------------------------------------------------------------

    training = pd.read_csv(
        TRAIN_FEATURES
    )

    training = training[
        training["label"].isin(
            ["human", "ai"]
        )
    ].copy()

    # ------------------------------------------------------------
    # HC3
    # ------------------------------------------------------------

    hc3_metadata = pd.read_csv(
        HC3_METADATA
    )

    print(
        f"HC3 samples: {len(hc3_metadata)}"
    )

    # ------------------------------------------------------------
    # Extract HC3 features
    # ------------------------------------------------------------

    print()
    print("Loading language model...")

    language_model = LanguageModelAnalyzer()

    extractor = EssayFeatureExtractor(
        language_model=language_model
    )

    hc3_rows = []

    for index, row in hc3_metadata.iterrows():

        essay_id = row["essay_id"]
        label = row["label"]

        path = (
            HC3_DIR
            / label
            / f"{essay_id}.txt"
        )

        text = path.read_text(
            encoding="utf-8"
        )

        extracted = extractor.extract(
            text
        )

        feature_dict = extracted.to_dict()

        result = {
            "essay_id": essay_id,
            "label": label,
        }

        for feature in features:

            if feature not in feature_dict:
                raise RuntimeError(
                    f"Missing feature: {feature}"
                )

            result[feature] = (
                feature_dict[feature]
            )

        hc3_rows.append(result)

        print(
            f"[{index + 1}/{len(hc3_metadata)}] "
            f"{essay_id}"
        )

    hc3 = pd.DataFrame(
        hc3_rows
    )

    # ------------------------------------------------------------
    # Distribution comparison
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("FEATURE DISTRIBUTION COMPARISON")
    print("=" * 70)

    for feature in features:

        train_ai = training.loc[
            training["label"] == "ai",
            feature,
        ]

        train_human = training.loc[
            training["label"] == "human",
            feature,
        ]

        hc3_ai = hc3.loc[
            hc3["label"] == "ai",
            feature,
        ]

        hc3_human = hc3.loc[
            hc3["label"] == "human",
            feature,
        ]

        print()
        print(
            f"{feature}"
        )
        print("-" * 70)

        print(
            f"Training AI:     "
            f"mean={train_ai.mean():.4f} "
            f"median={train_ai.median():.4f}"
        )

        print(
            f"Training Human:  "
            f"mean={train_human.mean():.4f} "
            f"median={train_human.median():.4f}"
        )

        print(
            f"HC3 AI:          "
            f"mean={hc3_ai.mean():.4f} "
            f"median={hc3_ai.median():.4f}"
        )

        print(
            f"HC3 Human:       "
            f"mean={hc3_human.mean():.4f} "
            f"median={hc3_human.median():.4f}"
        )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    output = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "hc3_feature_analysis.csv"
    )

    combined = pd.concat(
        [
            training[
                ["essay_id", "label"]
                + features
            ].assign(
                dataset="training"
            ),
            hc3.assign(
                dataset="hc3"
            ),
        ],
        ignore_index=True,
    )

    combined.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 70)
    print(
        f"Saved: {output}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
