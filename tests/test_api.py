from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_empty_question_is_rejected():
    response = client.post(
        "/ask",
        json={"question": "   "}
    )

    assert response.status_code == 400


def test_missing_question_is_rejected():
    response = client.post(
        "/ask",
        json={}
    )

    assert response.status_code == 422


def test_question_response_structure(monkeypatch):
    def fake_generate_answer(question):
        return {
            "answer": "Managers approve or reject leave requests.",
            "sources": ["employee_policy.txt"]
        }

    monkeypatch.setattr(
        "src.api.generate_answer",
        fake_generate_answer
    )

    response = client.post(
        "/ask",
        json={
            "question": "Who approves or rejects leave requests?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == (
        "Who approves or rejects leave requests?"
    )
    assert data["answer"] == (
        "Managers approve or reject leave requests."
    )
    assert data["sources"] == ["employee_policy.txt"]
    assert data["citations"] == []
