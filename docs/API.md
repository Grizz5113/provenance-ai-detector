# API Documentation

## Overview

The project exposes the provenance detector through a FastAPI application. It accepts text and returns a provenance classification with class probabilities.

## Running the API

```bash
uvicorn api.main:app --reload
```

## Prediction

**Endpoint:** `POST /predict`

**Request:**
```json
{
  "text": "Your text goes here."
}
```

**Response:**
```json
{
  "prediction": "human",
  "probabilities": {
    "ai": 0.02,
    "human": 0.91,
    "hybrid": 0.07
  },
  "confidence": 0.91,
  "feature_count": 40,
  "character_count": 25,
  "word_count": 5
}
```

## Sentence Prediction

**Endpoint:** `POST /predict/sentences`

Because the trained model was not designed as a true single-sentence classifier, this endpoint uses local context windows around each sentence, targeting 90 words (min 70, max 110).

## Long-Document Prediction

Long documents are divided into chunks, each classified independently. Probabilities are aggregated using each chunk's word count as weight; the class with the highest aggregated probability becomes the final prediction.

## Classes

- **human** — primarily human-written
- **ai** — primarily AI-generated
- **hybrid** — human writing with AI modification, or characteristics associated with hybrid provenance

## Confidence

The highest class probability is reported as model confidence. Confidence reflects model probability, not a guarantee of factual correctness.

## Error Handling

The API validates supplied text; invalid or empty input returns an HTTP error response.

## Interactive API Documentation

Available through FastAPI's default documentation routes when running with default configuration.
