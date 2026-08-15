from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


LM_FEATURES = [
    "mean_nll",
    "perplexity",
    "nll_std",
    "nll_min",
    "nll_max",
    "nll_median",
    "nll_p90",
]

LEXICAL_FEATURES = [
    "unique_token_count",
    "type_token_ratio",
    "hapax_ratio",
    "vocabulary_entropy",
    "repeated_token_ratio",
]


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


def evaluate(
    name,
    features,
    train,
    hc3,
):

    model = build_model()

    model.fit(
        train[features],
        train["label"],
    )

    predictions = model.predict(
        hc3[features]
    )

    accuracy = accuracy_score(
        hc3["label"],
        predictions,
    )

    macro_f1 = f1_score(
        hc3["label"],
        predictions,
        average="macro",
    )

    ai_recall = recall_score(
        hc3["label"],
        predictions,
        labels=["ai"],
        average=None,
        zero_division=0,
    )[0]

    human_recall = recall_score(
        hc3["label"],
        predictions,
        labels=["human"],
        average=None,
        zero_division=0,
    )[0]

    return {
        "experiment": name,
        "features": len(features),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "ai_recall": ai_recall,
        "human_recall": human_recall,
    }


def main():

    print("=" * 70)
    print("PROVENANCE — HC3 LM + LEXICAL ABLATION")
    print("=" * 70)

    with MODEL_METADATA.open(
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    available_features = set(
        metadata["features"]
    )

    train = pd.read_csv(
        TRAIN_FEATURES
    )

    train = train[
        train["label"].isin(
            ["ai", "human"]
        )
    ].copy()

    hc3_metadata = pd.read_csv(
        HC3_METADATA
    )

    print(
        f"Training samples: {len(train)}"
    )

    print(
        f"HC3 samples:      {len(hc3_metadata)}"
    )

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

        values = extracted.to_dict()

        result = {
            "essay_id": essay_id,
            "label": label,
        }

        for feature in available_features:

            if feature in values:
                result[feature] = values[feature]

        hc3_rows.append(result)

    hc3 = pd.DataFrame(
        hc3_rows
    )

    # ------------------------------------------------------------
    # Baseline: all LM + lexical
    # ------------------------------------------------------------

    baseline_features = (
        LM_FEATURES
        + LEXICAL_FEATURES
    )

    experiments = [
        (
            "ALL LM + LEXICAL",
            baseline_features,
        )
    ]

    # ------------------------------------------------------------
    # Leave one lexical feature out
    # ------------------------------------------------------------

    for removed in LEXICAL_FEATURES:

        features = [
            feature
            for feature in baseline_features
            if feature != removed
        ]

        experiments.append(
            (
                f"WITHOUT {removed}",
                features,
            )
        )

    # ------------------------------------------------------------
    # LM + individual lexical feature
    # ------------------------------------------------------------

    for lexical in LEXICAL_FEATURES:

        experiments.append(
            (
                f"LM + {lexical}",
                LM_FEATURES + [lexical],
            )
        )

    results = []

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for name, features in experiments:

        missing = [
            feature
            for feature in features
            if feature not in train.columns
        ]

        if missing:
            print(
                f"Skipping {name}: "
                f"missing {missing}"
            )
            continue

        result = evaluate(
            name,
            features,
            train,
            hc3,
        )

        results.append(result)

        print(
            f"{name:45s} "
            f"accuracy={result['accuracy']:.4f} "
            f"f1={result['macro_f1']:.4f} "
            f"AI={result['ai_recall']:.4f} "
            f"Human={result['human_recall']:.4f}"
        )

    result_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print("SORTED BY MACRO F1")
    print("=" * 70)

    print(
        result_df.sort_values(
            "macro_f1",
            ascending=False,
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()