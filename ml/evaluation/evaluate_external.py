from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVALUATION_ROOT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)


def get_benchmark_paths():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark",
        default=None,
        help=(
            "Evaluation benchmark directory "
            "inside data/evaluation/"
        ),
    )

    args = parser.parse_args()

    if args.benchmark:
        evaluation_dir = (
            EVALUATION_ROOT
            / args.benchmark
        )
    else:
        evaluation_dir = EVALUATION_ROOT

    metadata_file = (
        evaluation_dir
        / "metadata.csv"
    )

    return evaluation_dir, metadata_file

RAW_DIR = PROJECT_ROOT / "data" / "raw"

MODEL_FILE = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "artifacts"
    / "provenance_detector.joblib"
)

MODEL_metadata_file = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "artifacts"
    / "model_metadata.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

PREDICTIONS_FILE = (
    OUTPUT_DIR / "external_evaluation_predictions.csv"
)


# ---------------------------------------------------------------------------
# Detector components
# ---------------------------------------------------------------------------

from backend.app.detector.language_model import LanguageModelAnalyzer
from backend.app.features.extractor import EssayFeatureExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Normalize text for duplicate/leakage detection.
    """
    return " ".join(text.lower().split())


def text_hash(text: str) -> str:
    normalized = normalize_text(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def load_metadata(
    metadata_file: Path,
) -> list[dict[str, str]]:

    if not metadata_file.exists():
        raise RuntimeError(
            f"Evaluation metadata not found: {metadata_file}"
        )

    with metadata_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        return list(csv.DictReader(f))

def find_evaluation_file(
    evaluation_dir: Path,
    essay_id: str,
    label: str,
) -> Path | None:

    path = (
        evaluation_dir
        / label
        / f"{essay_id}.txt"
    )

    if path.exists():
        return path

    return None

def build_training_hashes() -> dict[str, str]:

    hashes = {}

    for label in (
        "human",
        "ai",
        "hybrid",
    ):

        directory = RAW_DIR / label

        if not directory.exists():
            continue

        for path in directory.glob("*.txt"):

            text = path.read_text(
                encoding="utf-8"
            )

            hashes[text_hash(text)] = path.name

    return hashes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=" * 70)
    print("PROVENANCE — EXTERNAL DATASET VALIDATION")
    print("=" * 70)

    evaluation_dir, metadata_file = (
        get_benchmark_paths()
    )
    if not MODEL_FILE.exists():
        raise RuntimeError(
            f"Model not found: {MODEL_FILE}"
        )

    metadata = load_metadata(
    metadata_file
)

    if not metadata:
        raise RuntimeError(
            "No external evaluation records found."
        )

    print(
        f"External metadata records: {len(metadata)}"
    )

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------

    print()
    print("Loading trained classifier...")

    model = joblib.load(MODEL_FILE)

    model_metadata = {}

    if MODEL_metadata_file.exists():

        with MODEL_metadata_file.open(
            "r",
            encoding="utf-8",
        ) as f:
            model_metadata = json.load(f)

    model_features = model_metadata.get(
        "features",
        [],
    )

    if not model_features:
        raise RuntimeError(
            "Model metadata does not contain feature list."
        )

    print(
        f"Model features: {len(model_features)}"
    )

    print()
    print("Loading language model...")

    language_model = LanguageModelAnalyzer()

    extractor = EssayFeatureExtractor(
        language_model=language_model,
    )

    # -----------------------------------------------------------------------
    # Leakage detection
    # -----------------------------------------------------------------------

    print()
    print("Checking for training-data leakage...")

    training_hashes = build_training_hashes()

    print(
        f"Training texts indexed: "
        f"{len(training_hashes)}"
    )

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------

    rows = []

    leakage_count = 0

    for index, meta in enumerate(
        metadata,
        start=1,
    ):

        essay_id = meta.get(
            "essay_id",
            "",
        ).strip()

        label = meta.get(
            "label",
            "",
        ).strip()

        if label not in {
            "human",
            "ai",
            "hybrid",
        }:
            raise RuntimeError(
                f"{essay_id}: invalid label '{label}'. "
                "Expected human, ai, or hybrid."
            )

        essay_file = find_evaluation_file(
        evaluation_dir,
        essay_id,
        label,
    )

        if essay_file is None:
            raise RuntimeError(
                f"Missing essay file for "
                f"{essay_id}: expected "
                f"{evaluation_dir / label / (essay_id + '.txt')}"
            )

        text = essay_file.read_text(
            encoding="utf-8"
        )

        if len(text.strip()) < 20:
            raise RuntimeError(
                f"{essay_id}: text is too short."
            )

        # ---------------------------------------------------------------
        # Leakage check
        # ---------------------------------------------------------------

        current_hash = text_hash(text)

        leaked_from = training_hashes.get(
            current_hash
        )

        if leaked_from is not None:

            leakage_count += 1

            print(
                f"LEAKAGE: {essay_id} "
                f"matches training file {leaked_from}"
            )

            continue

        # ---------------------------------------------------------------
        # Feature extraction
        # ---------------------------------------------------------------

        print(
            f"[{index}/{len(metadata)}] "
            f"Evaluating {essay_id}..."
        )

        extracted = extractor.extract(
            text
        )

        feature_dict = extracted.to_dict()

        missing = [
            feature
            for feature in model_features
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
                    for feature in model_features
                }
            ]
        )

        prediction = model.predict(X)[0]

        probabilities_array = (
            model.predict_proba(X)[0]
        )

        probabilities = {
            class_name: float(probability)
            for class_name, probability
            in zip(
                model.classes_,
                probabilities_array,
            )
        }

        confidence = max(
            probabilities.values()
        )

        rows.append(
            {
                "essay_id": essay_id,
                "actual": label,
                "predicted": prediction,
                "correct": prediction == label,
                "confidence": confidence,
                "ai_probability": probabilities.get(
                    "ai",
                    0.0,
                ),
                "human_probability": probabilities.get(
                    "human",
                    0.0,
                ),
                "hybrid_probability": probabilities.get(
                    "hybrid",
                    0.0,
                ),
                "topic": meta.get(
                    "topic",
                    "",
                ),
                "source": meta.get(
                    "source",
                    "",
                ),
                "source_group": meta.get(
                    "source_group",
                    "",
                ),
                "ai_intervention_level": meta.get(
                    "ai_intervention_level",
                    "",
                ),
                "character_count": len(text),
                "word_count": len(text.split()),
            }
        )

    # -----------------------------------------------------------------------
    # Leakage failure
    # -----------------------------------------------------------------------

    if leakage_count > 0:
        raise RuntimeError(
            f"\nExternal evaluation aborted: "
            f"{leakage_count} sample(s) leaked from training data."
        )

    if not rows:
        raise RuntimeError(
            "No external evaluation rows were generated."
        )

    result_df = pd.DataFrame(rows)

    y_true = result_df["actual"]
    y_pred = result_df["predicted"]

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    hybrid_mask = y_true == "hybrid"

    if hybrid_mask.sum() > 0:
        hybrid_recall = (
            (y_pred[hybrid_mask] == "hybrid").sum()
            / hybrid_mask.sum()
        )
    else:
        hybrid_recall = float("nan")

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("EXTERNAL EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Samples evaluated: {len(result_df)}"
    )

    print(
        f"Accuracy:          {accuracy:.4f}"
    )

    print(
        f"Macro precision:   {macro_precision:.4f}"
    )

    print(
        f"Macro recall:      {macro_recall:.4f}"
    )

    print(
        f"Macro F1:          {macro_f1:.4f}"
    )

    if hybrid_mask.sum() > 0:
        print(
            f"Hybrid recall:     {hybrid_recall:.4f}"
        )

    print()
    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[
                "ai",
                "human",
                "hybrid",
            ],
            zero_division=0,
        )
    )

    print("Confusion matrix:")

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            "ai",
            "human",
            "hybrid",
        ],
    )

    print(
        pd.DataFrame(
            cm,
            index=[
                "actual_ai",
                "actual_human",
                "actual_hybrid",
            ],
            columns=[
                "pred_ai",
                "pred_human",
                "pred_hybrid",
            ],
        )
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
                    "confidence",
                    "ai_probability",
                    "human_probability",
                    "hybrid_probability",
                ]
            ].to_string(index=False)
        )

    print()
    print("Results saved to:")
    print(PREDICTIONS_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()
