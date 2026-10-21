
from fastapi.testclient import TestClient

import api


client = TestClient(api.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate_valid_request(monkeypatch):
    def fake_generate_text(prompt, max_new_tokens, temperature):
        return prompt + " generated text"

    monkeypatch.setattr(
        api.generator,
        "generate_text",
        fake_generate_text,
    )

    response = client.post(
        "/generate",
        json={
            "prompt": "To be",
            "max_new_tokens": 10,
            "temperature": 0.8,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "generated_text": "To be generated text"
    }


def test_empty_prompt():
    response = client.post(
        "/generate",
        json={
            "prompt": "",
            "max_new_tokens": 10,
            "temperature": 0.8,
        },
    )

    assert response.status_code == 422


def test_zero_tokens():
    response = client.post(
        "/generate",
        json={
            "prompt": "To be",
            "max_new_tokens": 0,
            "temperature": 0.8,
        },
    )

    assert response.status_code == 422


def test_too_many_tokens():
    response = client.post(
        "/generate",
        json={
            "prompt": "To be",
            "max_new_tokens": 201,
            "temperature": 0.8,
        },
    )

    assert response.status_code == 422


def test_zero_temperature():
    response = client.post(
        "/generate",
        json={
            "prompt": "To be",
            "max_new_tokens": 10,
            "temperature": 0,
        },
    )

    assert response.status_code == 422


def test_excessive_temperature():
    response = client.post(
        "/generate",
        json={
            "prompt": "To be",
            "max_new_tokens": 10,
            "temperature": 2.1,
        },
    )

    assert response.status_code == 422


def test_generation_failure(monkeypatch):
    def failing_generate_text(prompt, max_new_tokens, temperature):
        raise RuntimeError("Simulated model failure")

    monkeypatch.setattr(
        api.generator,
        "generate_text",
        failing_generate_text,
    )

    response = client.post(
        "/generate",
        json={
            "prompt": "To be",
            "max_new_tokens": 10,
            "temperature": 0.8,
        },
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Text generation failed"
    }
