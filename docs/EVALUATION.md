# Evaluation Notes

## Overview

Observed model behavior on local and external test samples: classification behavior, probability distributions, text-length effects, human/AI separation, hybrid detection, external dataset behavior.

## Local AI Samples

| Sample | Words | Prediction | AI | Human | Hybrid |
|---|---:|---|---:|---:|---:|
| ai_001 | 623 | AI | 0.9667 | 0.0033 | 0.0300 |
| ai_002 | 672 | AI | 0.9300 | 0.0100 | 0.0600 |
| ai_003 | 634 | AI | 0.8833 | 0.0167 | 0.1000 |

## Local Human Samples

| Sample | Words | Prediction | AI | Human | Hybrid |
|---|---:|---|---:|---:|---:|
| human_001 | 703 | Human | 0.0467 | 0.7833 | 0.1700 |
| human_002 | 382 | Human | 0.0067 | 0.9400 | 0.0533 |
| human_003 | 399 | Human | 0.0033 | 0.8000 | 0.1967 |

## AI Text Length Experiment

| Words | Prediction | AI | Human | Hybrid |
|---:|---|---:|---:|---:|
| 100 | Human | 0.0233 | 0.6633 | 0.3133 |
| 120 | Human | 0.0100 | 0.6033 | 0.3867 |
| 150 | Human | 0.0067 | 0.5867 | 0.4067 |
| 180 | Human | 0.0100 | 0.5567 | 0.4333 |
| 220 | Human | 0.0100 | 0.5367 | 0.4533 |
| 300 | Hybrid | 0.0133 | 0.4300 | 0.5567 |
| 400 | Hybrid | 0.1467 | 0.2633 | 0.5900 |
| 500 | Hybrid | 0.3067 | 0.1267 | 0.5667 |
| 600 | AI | 0.8267 | 0.0200 | 0.1533 |

## Human Text Length Experiment

| Words | Prediction | AI | Human | Hybrid |
|---:|---|---:|---:|---:|
| 100 | Human | 0.0300 | 0.8833 | 0.0867 |
| 120 | Human | 0.0333 | 0.8700 | 0.0967 |
| 150 | Human | 0.0300 | 0.8967 | 0.0733 |
| 180 | Human | 0.0100 | 0.9600 | 0.0300 |
| 220 | Human | 0.0000 | 0.6900 | 0.3100 |
| 300 | Human | 0.0033 | 0.5833 | 0.4133 |
| 400 | Human | 0.0033 | 0.5967 | 0.4000 |
| 500 | Human | 0.0033 | 0.6100 | 0.3867 |
| 600 | Human | 0.0100 | 0.6067 | 0.3833 |

## Interpretation

Text length has a significant effect on the feature distribution and therefore on classification. Short portions of an AI document can be classified as human or hybrid even when the complete document is confidently classified as AI. This motivated document chunking and probability aggregation in the API.

## External Human Test

An HC3 human sample, tested using local context windows (~116 words, ~125 model tokens):

```text
AI:     0.040
Human:  0.903
Hybrid: 0.057
```

## Model Features

40 features, `RandomForestClassifier`.

## Evaluation Philosophy

These results are not universal accuracy — they describe behavior on the project's available datasets and test samples. A rigorous benchmark should use held-out test data, source-group-aware splitting, multiple independent datasets, class-balanced evaluation, precision, recall, F1, confusion matrix, and calibration analysis.

## Important Limitation

AI detection is inherently difficult because human writing and AI writing can overlap substantially. A prediction should be treated as a statistical classification result, not definitive proof of authorship.
