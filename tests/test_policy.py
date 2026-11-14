import pytest
from app.policy import PolicyGate, PolicyError

def test_requires_authorization():
    with pytest.raises(PolicyError):
        PolicyGate("http://127.0.0.1:8001/chat", False, 5)

def test_rejects_non_localhost():
    with pytest.raises(PolicyError):
        PolicyGate("https://example.com/chat", True, 5)

def test_budget():
    gate = PolicyGate("http://127.0.0.1:8001/chat", True, 1)
    gate.authorize_request("http://127.0.0.1:8001/chat")
    with pytest.raises(PolicyError):
        gate.authorize_request("http://127.0.0.1:8001/chat")
