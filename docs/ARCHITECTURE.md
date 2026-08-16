# System Architecture

## Overview

Provenance AI Detector is a machine-learning system designed to classify text into three provenance categories:

- `human`
- `ai`
- `hybrid`

The system combines linguistic statistics, lexical characteristics, sentence structure, repetition patterns, punctuation patterns, and language-model measurements.

---

## High-Level Architecture

```text
                         INPUT TEXT
                             |
                             v
                    +------------------+
                    |   FastAPI API    |
                    +------------------+
                             |
                             v
                    +------------------+
                    | Text Validation  |
                    +------------------+
                             |
                             v
                    +------------------+
                    | Text Chunking /  |
                    | Context Windows  |
                    +------------------+
                             |
                             v
                    +------------------+
                    | Feature          |
                    | Extraction       |
                    +------------------+
                             |
          +----------------+----------------+
          |                |                |
          v                v                v
    Language Model     Linguistic       Structural
      Features         Features         Features
          |                |                |
          +----------------+----------------+
                           |
                           v
                  40-Dimensional Vector
                           |
                           v
                 +----------------------+
                 | Random Forest        |
                 | Classifier            |
                 +----------------------+
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
           HUMAN          AI           HYBRID
```

## Main Components

### API

Located in `api/`.

The API:

- Receives text.
- Validates the input.
- Splits long documents into appropriate windows.
- Extracts the trained feature set.
- Runs the Random Forest model.
- Calculates class probabilities.
- Returns the predicted provenance class.

### Backend Feature Extraction

Located in `backend/app/features/`. Major modules:

- `lexical.py`
- `perplexity.py`
- `punctuation.py`
- `repetition.py`
- `rhythm.py`
- `sentence.py`
- `text_similarity.py`
- `extractor.py`

### Language Model

The detector uses `EleutherAI/pythia-160m`, computing token-level negative log-likelihood measurements converted into: mean NLL, perplexity, NLL std, min, max, median, and 90th percentile NLL. The language model does not directly decide provenance — it provides measurable features to the classifier.

### Machine Learning Layer

Located in `ml/models/`. Classifier: `RandomForestClassifier`.

- Trained model: `ml/models/artifacts/provenance_detector.joblib`
- Metadata: `ml/models/artifacts/model_metadata.json`

The inference code verifies the trained model and metadata share the same feature ordering.

### Dataset Pipeline

Located in `ml/dataset/`. Handles importing external datasets, selecting human samples, generating AI/hybrid samples, registering provenance metadata, merging, and validating. Major sources: HC3 and PERSUADE 2.0.

### Hybrid Provenance

Hybrid samples represent human-written text later modified by AI, at three intervention levels: light, moderate, heavy. Each hybrid sample stores `source_human_essay`, `ai_intervention_level`, `model`, `temperature`, `seed`, `edit_ratio`.

## Inference Strategy

The classifier was trained on relatively large text samples, so classifying a single short sentence directly can fall outside the training distribution. The API uses context windows sized 70–110 words (target 90) for sentence-level analysis, and chunks long documents the same way, aggregating chunk-level probabilities by word-count weight to produce one document-level prediction.

## Design Principle

The system separates measurement (feature extraction) from machine learning classification. Individual features do not independently determine provenance — the Random Forest learns combinations of these measurements from training data.
