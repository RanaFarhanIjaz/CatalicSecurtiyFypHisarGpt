from types import SimpleNamespace

from fastapi.testclient import TestClient

from target_system import SYSTEM_PROMPT, app, Runner


client = TestClient(app)


def test_target_health_does_not_expose_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-real-key")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert "test-only-not-a-real-key" not in response.text


def test_target_chat_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_target_chat_uses_system_prompt_and_returns_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-real-key")
    captured = {}

    async def fake_run(agent, message):
        captured["instructions"] = agent.instructions
        captured["message"] = message
        return SimpleNamespace(final_output="A local target response.")

    monkeypatch.setattr(Runner, "run", fake_run)

    response = client.post("/chat", json={"message": "test input"})

    assert response.status_code == 200
    assert response.json() == {"response": "A local target response."}
    assert captured["message"] == "test input"
    assert captured["instructions"] == SYSTEM_PROMPT
