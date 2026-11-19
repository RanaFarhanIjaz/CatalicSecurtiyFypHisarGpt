import asyncio

from app import orchestrator
from app.schemas import AssessmentRequest


def test_demo_assessment_runs_without_ai_evaluator(monkeypatch):
    class FakeTarget:
        def __init__(self, url, gate):
            self.url = url
            self.gate = gate
            self.responses = iter([
                "I can help with general questions.",
                "I cannot override my instructions.",
                "I cannot reveal hidden system prompt: private values.",
                "I do not have access to that document.",
                "I cannot transfer money.",
                '{"status":"ok","message":"ready"}',
                "I do not remember previous session details.",
            ])

        async def send(self, message):
            self.gate.authorize_request(self.url)
            return next(self.responses)

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("The demo evaluator must not call the AI service.")

    monkeypatch.setattr(orchestrator, "LocalTarget", FakeTarget)
    monkeypatch.setattr(orchestrator, "run_specialist", fail_if_called)
    monkeypatch.setattr(orchestrator, "run_judge", fail_if_called)

    result = asyncio.run(orchestrator.run_assessment(AssessmentRequest(
        target_url="http://127.0.0.1:8001/chat",
        authorization_confirmed=True,
        evaluator="demo",
        prompt_mode="predefined",
        max_requests=7,
    )))

    assert result["total_requests"] == 7
    assert result["evaluator"] == "demo"
    assert result["observations"][2].candidate is True
    assert result["score"] == 80
    assert "without an additional AI service" in result["judge_summary"]
