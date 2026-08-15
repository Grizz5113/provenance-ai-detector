from __future__ import annotations

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
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


FEATURE_GROUPS = {
    "lm_nll": [
        "mean_nll",
        "perplexity",
        "nll_std",
        "nll_min",
        "nll_max",
        "nll_median",
        "nll_p90",
    ],

    "sentence_structure": [
        "sentence_count",
        "mean_sentence_length",
        "median_sentence_length",
        "sentence_length_std",
        "min_sentence_length",
        "max_sentence_length",
        "sentence_length_cv",
    ],

    "lexical_diversity": [
        "unique_token_count",
        "type_token_ratio",
        "hapax_ratio",
        "vocabulary_entropy",
        "repeated_token_ratio",
    ],

    "repetition": [
        "unique_bigrams",
        "repeated_bigrams",
        "bigram_repetition_ratio",
        "unique_trigrams",
        "repeated_trigrams",
        "trigram_repetition_ratio",
        "most_common_bigram_count",
        "most_common_trigram_count",
    ],

    "punctuation": [
        "punctuation_count",
        "punctuation_density",
        "punctuation_types",
        "comma_count",
        "period_count",
        "semicolon_count",
        "colon_count",
        "question_count",
        "exclamation_count",
        "parentheses_count",
        "quotation_count",
        "contraction_count",
    ],

    "token_length": [
        "token_count",
    ],
}


SEEDS = [42, 7, 21, 100, 123]


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


def evaluate_feature_set(
    name,
    features,
    df,
):
    X = df[features]
    y = df["label"]
    groups = df["source_group"]

    results = []

    for seed in SEEDS:

        cv = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )

        predictions = cross_val_predict(
            build_model(),
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
            (
                predictions[hybrid_mask]
                == "hybrid"
            ).sum()
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
        "feature_count": len(features),
        "accuracy_mean": result_df["accuracy"].mean(),
        "accuracy_std": result_df["accuracy"].std(ddof=1),
        "macro_f1_mean": result_df["macro_f1"].mean(),
        "macro_f1_std": result_df["macro_f1"].std(ddof=1),
        "hybrid_recall_mean": result_df["hybrid_recall"].mean(),
        "hybrid_recall_std": result_df["hybrid_recall"].std(ddof=1),
    }


def main():

    print("=" * 70)
    print("PROVENANCE — FEATURE ABLATION")
    print("=" * 70)

    df = pd.read_csv(DATA_FILE)

    all_features = [
        c
        for c in df.columns
        if c not in META_COLUMNS
    ]

    if len(all_features) != 40:
        raise RuntimeError(
            f"Expected 40 features, "
            f"found {len(all_features)}."
        )

    # Validate feature definitions.
    grouped_features = []

    for group_features in FEATURE_GROUPS.values():
        grouped_features.extend(group_features)

    missing = set(all_features) - set(grouped_features)

    if missing:
        raise RuntimeError(
            "Features not assigned to a group: "
            + ", ".join(sorted(missing))
        )

    duplicated = [
        feature
        for feature in grouped_features
        if grouped_features.count(feature) > 1
    ]

    if duplicated:
        raise RuntimeError(
            "Features assigned to multiple groups: "
            + ", ".join(sorted(set(duplicated)))
        )

    experiments = []

    # ------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------

    experiments.append(
        (
            "ALL FEATURES",
            all_features,
        )
    )

    # ------------------------------------------------------------
    # Leave-one-group-out ablation
    # ------------------------------------------------------------

    for group_name, group_features in FEATURE_GROUPS.items():

        remaining = [
            feature
            for feature in all_features
            if feature not in group_features
        ]

        experiments.append(
            (
                f"WITHOUT {group_name.upper()}",
                remaining,
            )
        )

    # ------------------------------------------------------------
    # Individual groups
    # ------------------------------------------------------------

    for group_name, group_features in FEATURE_GROUPS.items():

        experiments.append(
            (
                f"ONLY {group_name.upper()}",
                group_features,
            )
        )
        # ------------------------------------------------------------
    # Combined candidate selected from augmented ablation
    # ------------------------------------------------------------

    combined_removed = set(
        FEATURE_GROUPS["punctuation"]
        + ["hapax_ratio"]
    )

    combined_features = [
        feature
        for feature in all_features
        if feature not in combined_removed
    ]

    experiments.append(
        (
            "WITHOUT PUNCTUATION + HAPAX_RATIO",
            combined_features,
        )
    )

    
    results = []

    for name, features in experiments:

        print()
        print(
            f"Evaluating: {name} "
            f"({len(features)} features)"
        )

        metrics = evaluate_feature_set(
            name,
            features,
            df,
        )

        results.append(
            {
                "experiment": name,
                **metrics,
            }
        )

    results_df = pd.DataFrame(results)

    print()
    print("=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)

    display_columns = [
        "experiment",
        "feature_count",
        "accuracy_mean",
        "macro_f1_mean",
        "hybrid_recall_mean",
    ]

    print(
        results_df[
            display_columns
        ]
        .sort_values(
            "macro_f1_mean",
            ascending=False,
        )
        .to_string(index=False)
    )

    print()
    print("=" * 70)
    print("STABILITY — MEAN ± STD")
    print("=" * 70)

    for _, row in results_df.sort_values(
        "macro_f1_mean",
        ascending=False,
    ).iterrows():

        print(
            f"{row['experiment']:<35} "
            f"Acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f} | "
            f"F1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f} | "
            f"Hybrid={row['hybrid_recall_mean']:.4f}±{row['hybrid_recall_std']:.4f}"
        )

    output_file = (
        "data/processed/feature_ablation_results.csv"
    )

    results_df.to_csv(
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
