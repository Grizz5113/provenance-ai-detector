from __future__ import annotations
from fastapi.middleware.cors import CORSMiddleware
import json
import sys
from pathlib import Path
import re
import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


from backend.app.detector.language_model import LanguageModelAnalyzer
from backend.app.features.extractor import EssayFeatureExtractor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_TEXT_LENGTH = 20
MAX_TEXT_LENGTH = 50_000

ALLOWED_FILE_TYPES = {
    ".txt",
}


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Provenance AI Detector",
    description=(
        "AI, Human, and Hybrid text provenance classifier."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

print("=" * 70)
print("PROVENANCE — API STARTUP")
print("=" * 70)

if not MODEL_FILE.exists():
    raise RuntimeError(
        f"Model not found: {MODEL_FILE}"
    )

print("Loading trained classifier...")

model = joblib.load(MODEL_FILE)

print("Loading language model...")

language_model = LanguageModelAnalyzer()

print("Creating feature extractor...")

extractor = EssayFeatureExtractor(
    language_model=language_model,
)

model_metadata = {}

if METADATA_FILE.exists():
    with METADATA_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        model_metadata = json.load(f)


MODEL_FEATURES = model_metadata.get(
    "features",
    [],
)
MODEL_FEATURES_FROM_MODEL = list(
    model.feature_names_in_
)

if MODEL_FEATURES_FROM_MODEL != MODEL_FEATURES:
    raise RuntimeError(
        "Model feature order mismatch between "
        "trained model and metadata."
    )
print(
    f"Classifier loaded: {MODEL_FILE}"
)

print(
    f"Final model features: "
    f"{len(MODEL_FEATURES)}"
)

print("=" * 70)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=MIN_TEXT_LENGTH,
        description="Text to classify.",
    )


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]
    confidence: float
    feature_count: int
    character_count: int
    word_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_text(text: str) -> str:

    text = text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    if len(text) < MIN_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Text must contain at least "
                f"{MIN_TEXT_LENGTH} characters."
            ),
        )

    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Text exceeds the maximum allowed "
                f"length of {MAX_TEXT_LENGTH} characters."
            ),
        )

    return text


def extract_model_features(
    text: str,
) -> pd.DataFrame:

    features = extractor.extract(text)

    feature_dict = features.to_dict()

    missing = [
        feature
        for feature in MODEL_FEATURES
        if feature not in feature_dict
    ]

    if missing:
        raise RuntimeError(
            "Missing model features: "
            + ", ".join(missing)
        )

    return pd.DataFrame(
        [
            {
                feature: feature_dict[feature]
                for feature in MODEL_FEATURES
            }
        ]
    )
def split_into_sentences(text: str):
    """
    Split text while preserving sentence order.
    """

    pattern = r'[^.!?]+(?:[.!?]+|$)'

    return [
        match.group(0)
        for match in re.finditer(pattern, text, re.DOTALL)
        if match.group(0).strip()
    ]



def chunk_text_into_windows(
    text: str,
    target_words: int = 90,
    min_words: int = 70,
    max_words: int = 110,
) -> list[str]:
    sentences = split_into_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        w = len(s.split())

        if current and current_words + w > max_words and current_words >= min_words:
            chunks.append(" ".join(current))
            current, current_words = [], 0

        current.append(s)
        current_words += w

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


def run_prediction(text: str) -> PredictionResponse:
    text = validate_text(text)

    chunks = chunk_text_into_windows(text)

    probs_list = []
    weights = []

    for chunk in chunks:
        feats = extract_model_features(chunk)
        proba = model.predict_proba(feats)[0]
        probs_list.append(proba)
        weights.append(len(chunk.split()))

    probs_matrix = np.array(probs_list)
    weights_arr = np.array(weights, dtype=float)

    avg_probs = np.average(probs_matrix, axis=0, weights=weights_arr)

    probabilities = {
        class_name: round(float(p), 6)
        for class_name, p in zip(model.classes_, avg_probs)
    }

    prediction = model.classes_[int(avg_probs.argmax())]
    confidence = max(probabilities.values())

    word_count = len(text.split())

    return PredictionResponse(
        prediction=prediction,
        probabilities=probabilities,
        confidence=round(confidence, 6),
        feature_count=len(MODEL_FEATURES),
        character_count=len(text),
        word_count=word_count,
    )

@app.post("/predict/sentences")
def predict_sentences(request: PredictionRequest):

    text = validate_text(request.text)

    try:
        # ---------------------------------------------------------------
        # Split into sentences
        # ---------------------------------------------------------------
        sentences = split_into_sentences(text)

        if not sentences:
            raise HTTPException(
                status_code=400,
                detail="Could not identify any sentences.",
            )

        # ---------------------------------------------------------------
        # IMPORTANT:
        #
        # The trained model was built from text samples of roughly
        # 70-110 words.
        #
        # It is NOT a true single-sentence classifier.
        #
        # Therefore each sentence is classified using a LOCAL CONTEXT
        # WINDOW centered around that sentence.
        #
        # This keeps inference much closer to the distribution seen
        # during training.
        # ---------------------------------------------------------------

        TARGET_WORDS = 90
        MIN_WINDOW_WORDS = 70
        MAX_WINDOW_WORDS = 110

        results = []

        def word_count(value: str) -> int:
            return len(value.split())

        def build_context_window(index: int) -> str:

            selected = [index]

            current_words = word_count(
                sentences[index].strip()
            )

            left = index - 1
            right = index + 1

            # Expand outward around the target sentence.
            # Prefer alternating left/right so the target remains
            # approximately centered in the context.
            while (
                current_words < TARGET_WORDS
                and (
                    left >= 0
                    or right < len(sentences)
                )
            ):

                candidates = []

                if left >= 0:
                    candidates.append(
                        (
                            abs(index - left),
                            left,
                            word_count(
                                sentences[left].strip()
                            ),
                        )
                    )

                if right < len(sentences):
                    candidates.append(
                        (
                            abs(index - right),
                            right,
                            word_count(
                                sentences[right].strip()
                            ),
                        )
                    )

                # Pick the closest sentence to the target.
                candidates.sort(
                    key=lambda item: item[0]
                )

                _, chosen, chosen_words = candidates[0]

                # Don't allow the context to grow excessively.
                if (
                    current_words + chosen_words
                    > MAX_WINDOW_WORDS
                    and current_words >= MIN_WINDOW_WORDS
                ):
                    break

                selected.append(chosen)

                current_words += chosen_words

                if chosen < index:
                    left -= 1
                else:
                    right += 1

            selected.sort()

            return " ".join(
                sentences[i].strip()
                for i in selected
                if sentences[i].strip()
            )

        # ---------------------------------------------------------------
        # Predict each sentence using local context
        # ---------------------------------------------------------------
        for index, sentence in enumerate(sentences):

            sentence_text = sentence.strip()

            if not sentence_text:
                continue

            context_text = build_context_window(index)

            context_word_count = word_count(
                context_text
            )

            # -----------------------------------------------------------
            # Extract exactly the same 40 features used by the model.
            # -----------------------------------------------------------
            context_features = extract_model_features(
                context_text
            )

            proba_array = model.predict_proba(
                context_features
            )[0]

            probabilities = {
                class_name: float(probability)
                for class_name, probability in zip(
                    model.classes_,
                    proba_array,
                )
            }

            prediction = model.classes_[
                proba_array.argmax()
            ]

            confidence = float(
                proba_array.max()
            )

            results.append(
                {
                    "text": sentence_text,

                    "prediction": prediction,

                    "confidence": round(
                        confidence,
                        6,
                    ),

                    "probabilities": {
                        key: round(
                            value,
                            6,
                        )
                        for key, value
                        in probabilities.items()
                    },

                    "window": index + 1,

                    "context_word_count": (
                        context_word_count
                    ),
                }
            )

        return {
            "sentence_count": len(results),

            "window_count": len(results),

            "sentences": results,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():

    return {
        "name": "Provenance AI Detector",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_extractor_loaded": extractor is not None,
        "feature_count": len(MODEL_FEATURES),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
):

    try:

        return run_prediction(
            request.text
        )

    except HTTPException:
        raise

    except Exception as exc:

        print(
            f"Prediction error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        )


@app.post(
    "/predict/file",
    response_model=PredictionResponse,
)
async def predict_file(
    file: UploadFile = File(...),
):

    filename = file.filename or ""

    suffix = Path(
        filename
    ).suffix.lower()

    if suffix not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Only .txt files are allowed."
            ),
        )

    try:

        contents = await file.read()

        if len(contents) > MAX_TEXT_LENGTH * 4:
            raise HTTPException(
                status_code=413,
                detail="File is too large.",
            )

        try:
            text = contents.decode(
                "utf-8"
            )
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail=(
                    "File must be valid UTF-8 text."
                ),
            )

        return run_prediction(text)

    except HTTPException:
        raise

    except Exception as exc:

        print(
            f"File prediction error: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="File prediction failed.",
        )
