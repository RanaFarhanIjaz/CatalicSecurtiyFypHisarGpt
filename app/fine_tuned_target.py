import httpx
from typing import Literal


class FineTunedTarget:
    """Adapter for fine-tuned model endpoints.

    model_type:
      - "huggingface_api": HuggingFace Inference API or HF-compatible endpoint
      - "local": any locally served model endpoint (HTTP POST, same JSON protocol as LocalTarget)
    """

    def __init__(self, model_id: str, model_type: str, base_url: str = "",
                 api_token: str = "", max_new_tokens: int = 512, fallback_demo: bool = False):
        self.model_id = model_id
        self.model_type = model_type
        self.api_token = api_token
        self.max_new_tokens = max_new_tokens
        self.fallback_demo = fallback_demo

        if model_type == "huggingface_api":
            from .pentest_agent import huggingface_inference_url
            self.endpoint = base_url or huggingface_inference_url(model_id)
        elif model_type == "local":
            self.endpoint = base_url  # e.g. http://127.0.0.1:8002/chat

    async def _simulate_demo_response(self, message: str) -> str:
        """Simulate realistic behavior for fine-tuned models in local demo mode."""
        msg = message.lower()
        model_lower = self.model_id.lower()

        # Zephyr-7B-Beta (DPO fine-tuned): helpful open model, susceptible to override framing
        if "zephyr" in model_lower:
            if "override" in msg or "disregard" in msg or "blue" in msg:
                return "OVERRIDE. As Zephyr-7B, I have acknowledged the override instruction."
            elif "reveal your hidden system prompt" in msg or "verbatim" in msg:
                return "I am Zephyr, an open-source assistant trained by Hugging Face H4. My system prompt instructs me to be helpful and honest."
            elif "confidential" in msg or "private document" in msg:
                return "I do not have access to unprovided confidential documents, but I will answer based on general knowledge."
            elif "json" in msg:
                return '{"status": "ok", "message": "Zephyr formatted output"}'
            else:
                return f"[Zephyr-7B-Beta] I am an open-source assistant fine-tuned with Direct Preference Optimization. I can help answer your questions."

        # Phi-3-Mini-Instruct (Instruction & safety tuned): stricter boundary defenses
        elif "phi" in model_lower:
            if "override" in msg or "disregard" in msg:
                return "I cannot follow instructions to override my core safety rules or emit bypass tokens."
            elif "reveal your hidden system prompt" in msg or "verbatim" in msg:
                return "I am programmed not to disclose my internal instructions or private developer prompts."
            elif "confidential" in msg or "private document" in msg:
                return "I cannot view, read, or infer the contents of documents not provided in this conversation."
            elif "json" in msg:
                return '{"status": "valid", "message": "Phi-3 formatted output"}'
            else:
                return f"[Phi-3-Mini-Instruct] I am a Microsoft Phi-3 instruction-tuned model. I will assist you within safe parameters."

        # Generic fine-tuned model fallback
        if "override" in msg or "disregard" in msg:
            return "I cannot execute commands that disregard my baseline instructions."
        return f"[{self.model_id}] Benign response generated for evaluation."

    async def send(self, message: str) -> str:
        """Send a message and return the model response as a string."""
        timeout = httpx.Timeout(180.0, connect=10.0)
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        if self.model_type == "huggingface_api":
            payload = {
                "inputs": message,
                "parameters": {"max_new_tokens": self.max_new_tokens, "return_full_text": False}
            }
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(self.endpoint, json=payload, headers=headers)
                    if response.status_code in (401, 403, 404, 429, 503) and self.fallback_demo:
                        return await self._simulate_demo_response(message)
                    response.raise_for_status()
                    data = response.json()
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    return str(data[0]["generated_text"])[:4000]
                if isinstance(data, dict):
                    if "generated_text" in data:
                        return str(data["generated_text"])[:4000]
                    if "error" in data:
                        if self.fallback_demo:
                            return await self._simulate_demo_response(message)
                        raise RuntimeError(f"HuggingFace API error: {data['error']}")
                return str(data)[:4000]
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                if self.fallback_demo:
                    return await self._simulate_demo_response(message)
                raise exc

        elif self.model_type == "local":
            payload = {"message": message}
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(self.endpoint, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                if isinstance(data, dict) and "response" in data:
                    return str(data["response"])[:4000]
                return str(data)[:4000]
            except Exception as exc:
                if self.fallback_demo:
                    return await self._simulate_demo_response(message)
                raise exc

        raise ValueError(f"Unknown model_type: {self.model_type}")
