import httpx
from .policy import PolicyGate

class LocalTarget:
    def __init__(self, url: str, gate: PolicyGate):
        self.url = url
        self.gate = gate

    async def send(self, message: str) -> str:
        self.gate.authorize_request(self.url)
        timeout = httpx.Timeout(120.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            response = await client.post(self.url, json={"message": message})
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict) or "response" not in data:
            return str(data)[:4000]
        return str(data["response"])[:4000]
