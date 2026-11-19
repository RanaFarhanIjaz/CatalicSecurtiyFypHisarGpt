import httpx
from .policy import PolicyGate


class LocalTarget:
    """Adaptive target client supporting message, messages (chat format), prompt, input, and text schemas."""

    def __init__(self, url: str, gate: PolicyGate):
        self.url = url
        self.gate = gate
        self._working_format: str | None = None

    def _extract_response_text(self, data) -> str:
        if isinstance(data, dict):
            if "response" in data:
                return str(data["response"])[:4000]
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                choice = data["choices"][0]
                if isinstance(choice, dict):
                    if "message" in choice and isinstance(choice["message"], dict):
                        return str(choice["message"].get("content", ""))[:4000]
                    if "text" in choice:
                        return str(choice["text"])[:4000]
            if "content" in data:
                return str(data["content"])[:4000]
            if "text" in data:
                return str(data["text"])[:4000]
            if "output" in data:
                return str(data["output"])[:4000]
            if "generated_text" in data:
                return str(data["generated_text"])[:4000]
        return str(data)[:4000]

    async def send(self, message: str) -> str:
        self.gate.authorize_request(self.url)
        clean_msg = (message or "").strip()
        if not clean_msg:
            clean_msg = "Security audit probe"
        clean_msg = clean_msg[:4000]

        timeout = httpx.Timeout(120.0, connect=5.0)

        # Candidate schemas to accommodate various target model APIs
        payload_candidates = [
            ("message", {"message": clean_msg}),
            ("messages", {"messages": [{"role": "user", "content": clean_msg}]}),
            ("prompt", {"prompt": clean_msg}),
            ("input", {"input": clean_msg}),
            ("text", {"text": clean_msg}),
        ]

        # Prioritize the detected working format if known
        if self._working_format:
            payload_candidates.sort(key=lambda x: 0 if x[0] == self._working_format else 1)

        last_error_text = ""
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            for fmt_name, payload in payload_candidates:
                try:
                    response = await client.post(self.url, json=payload)
                except httpx.RequestError as exc:
                    raise RuntimeError(f"Could not connect to target endpoint at {self.url}: {exc}")

                if response.status_code == 200:
                    self._working_format = fmt_name
                    try:
                        data = response.json()
                        return self._extract_response_text(data)
                    except Exception:
                        return response.text[:4000]
                elif response.status_code == 422:
                    # 422 Unprocessable Content indicates payload schema mismatch; try next schema candidate
                    last_error_text = response.text
                    continue
                else:
                    response.raise_for_status()

            raise RuntimeError(
                f"Target endpoint at {self.url} rejected the request with HTTP 422. "
                f"Target validation response: {last_error_text[:250]}. "
                "Ensure target endpoint accepts one of: {'message': ...}, {'prompt': ...}, or {'messages': [...]}."
            )
