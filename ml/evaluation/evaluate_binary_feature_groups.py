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


FEATURE_GROUPS = {
    "lm": [
        "mean_nll",
        "perplexity",
        "nll_std",
        "nll_min",
        "nll_max",
        "nll_median",
        "nll_p90",
    ],

    "sentence": [
        "sentence_count",
        "mean_sentence_length",
        "sentence_length_std",
        "min_sentence_length",
        "max_sentence_length",
    ],

    "lexical": [
        "unique_token_count",
        "type_token_ratio",
        "hapax_ratio",
        "vocabulary_entropy",
        "repeated_token_ratio",
    ],

    "punctuation": [
        "punctuation_count",
        "punctuation_types",
        "comma_count",
        "period_count",
        "colon_count",
        "parentheses_count",
        "contraction_count",
    ],

    "token": [
        "token_count",
    ],
}


EXPERIMENTS = {
    "LM": FEATURE_GROUPS["lm"],

    "SENTENCE": FEATURE_GROUPS["sentence"],

    "LEXICAL": FEATURE_GROUPS["lexical"],

    "PUNCTUATION": FEATURE_GROUPS["punctuation"],

    "TOKEN": FEATURE_GROUPS["token"],

    "LM+SENTENCE": (
        FEATURE_GROUPS["lm"]
        + FEATURE_GROUPS["sentence"]
    ),

    "LM+LEXICAL": (
        FEATURE_GROUPS["lm"]
        + FEATURE_GROUPS["lexical"]
    ),

    "LM+SENTENCE+LEXICAL": (
        FEATURE_GROUPS["lm"]
        + FEATURE_GROUPS["sentence"]
        + FEATURE_GROUPS["lexical"]
    ),
}


def build_model():

    return Pipeline(
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


def main():

    print("=" * 70)
    print("PROVENANCE — HC3 FEATURE GROUP EXPERIMENT")
    print("=" * 70)

    with MODEL_METADATA.open(
        "r",
        encoding="utf-8",
    ) as f:
        model_metadata = json.load(f)

    available_features = set(
        model_metadata["features"]
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

    # ------------------------------------------------------------
    # Extract HC3 features once
    # ------------------------------------------------------------

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

        for feature in available_features:

            if feature in feature_dict:
                result[feature] = (
                    feature_dict[feature]
                )

        hc3_rows.append(result)

    hc3 = pd.DataFrame(
        hc3_rows
    )

    # ------------------------------------------------------------
    # Experiments
    # ------------------------------------------------------------

    results = []

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for name, features in EXPERIMENTS.items():

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

        X_train = train[features]
        y_train = train["label"]

        X_test = hc3[features]
        y_test = hc3["label"]

        model = build_model()

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
        )

        ai_recall = recall_score(
            y_test,
            predictions,
            labels=["ai"],
            average=None,
            zero_division=0,
        )[0]

        human_recall = recall_score(
            y_test,
            predictions,
            labels=["human"],
            average=None,
            zero_division=0,
        )[0]

        results.append(
            {
                "experiment": name,
                "features": len(features),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "ai_recall": ai_recall,
                "human_recall": human_recall,
            }
        )

        print(
            f"{name:25s} "
            f"features={len(features):2d} "
            f"accuracy={accuracy:.4f} "
            f"f1={macro_f1:.4f} "
            f"AI_recall={ai_recall:.4f} "
            f"Human_recall={human_recall:.4f}"
        )

    print()
    print("=" * 70)
    print("SORTED BY MACRO F1")
    print("=" * 70)

    result_df = pd.DataFrame(
        results
    )

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