import joblib
import pytest
import requests
import subprocess
import time
import sys

from score import score


# Load trained model
model_bundle = joblib.load("best_model.pkl")


# UNIT TESTS

def test_smoke():
    pred, prob = score("hello world", model_bundle, 0.5)
    assert pred is not None
    assert prob is not None


def test_format():
    pred, prob = score("hello", model_bundle, 0.5)
    assert isinstance(pred, bool)
    assert isinstance(prob, float)


def test_prediction_values():
    pred, _ = score("hello", model_bundle, 0.5)
    assert pred in [True, False]


def test_propensity_range():
    _, prob = score("hello", model_bundle, 0.5)
    assert 0 <= prob <= 1


def test_threshold_zero():
    pred, _ = score("any message", model_bundle, 0)
    assert pred == True


def test_threshold_one():
    pred, prob = score("any message", model_bundle, 1)
    assert prob <= 1


def test_spam_message():
    pred, _ = score("Urgent! You have won a $1000 cash. Claim now!", model_bundle, 0.5)
    assert pred == True


def test_non_spam_message():
    pred, _ = score("Let's meet tomorrow", model_bundle, 0.5)
    assert pred == False


# INTEGRATION TEST

def test_flask_integration():

    process = subprocess.Popen([sys.executable, "app.py"])

    time.sleep(5)

    try:
        for _ in range(5):
            try:
                response = requests.post(
                    "http://127.0.0.1:5000/score",
                    json={"text": "Free lottery! Claim now!"}
                )
                break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            pytest.fail("Flask server did not start.")

        assert response.status_code == 200

        data = response.json()

        assert "prediction" in data
        assert "propensity" in data
        assert isinstance(data["prediction"], bool)
        assert isinstance(data["propensity"], float)
        assert 0.0 <= data["propensity"] <= 1.0

    finally:
        process.terminate()
        process.wait()