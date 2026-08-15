from __future__ import annotations

import io

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


VALID_TEXT = """
Technology has changed the way people communicate, learn, and work.
Digital platforms allow individuals to access information almost instantly
and connect with people across the world. However, this convenience also
creates challenges related to privacy, misinformation, and excessive
dependence on technology. For this reason, technology should be used
responsibly, with individuals and organizations considering both its
benefits and its potential social consequences.
""".strip()


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"
    assert data["name"] == "Provenance AI Detector"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["feature_extractor_loaded"] is True
    assert data["feature_count"] == 40


def test_predict_valid_text():
    response = client.post(
        "/predict",
        json={
            "text": VALID_TEXT,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in {
        "ai",
        "human",
        "hybrid",
    }

    assert set(data["probabilities"]) == {
        "ai",
        "human",
        "hybrid",
    }

    assert data["feature_count"] == 40

    assert 0.0 <= data["confidence"] <= 1.0

    probability_sum = sum(
        data["probabilities"].values()
    )

    assert abs(
        probability_sum - 1.0
    ) < 0.001


def test_predict_short_text():
    response = client.post(
        "/predict",
        json={
            "text": "hello world",
        },
    )

    assert response.status_code == 422


def test_predict_empty_text():
    response = client.post(
        "/predict",
        json={
            "text": "",
        },
    )

    assert response.status_code == 422


def test_predict_file_valid():
    response = client.post(
        "/predict/file",
        files={
            "file": (
                "essay.txt",
                VALID_TEXT.encode("utf-8"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in {
        "ai",
        "human",
        "hybrid",
    }

    assert data["feature_count"] == 40


def test_predict_file_invalid_extension():
    response = client.post(
        "/predict/file",
        files={
            "file": (
                "essay.pdf",
                VALID_TEXT.encode("utf-8"),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400

    assert (
        response.json()["detail"]
        == "Unsupported file type. Only .txt files are allowed."
    )


def test_predict_file_invalid_utf8():
    invalid_bytes = b"\xff\xfe\xfd\xfc"

    response = client.post(
        "/predict/file",
        files={
            "file": (
                "essay.txt",
                invalid_bytes,
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    assert (
        response.json()["detail"]
        == "File must be valid UTF-8 text."
    )


def test_predict_long_text():
    long_text = "technology " * 5000

    response = client.post(
        "/predict",
        json={
            "text": long_text,
        },
    )

    assert response.status_code == 413
