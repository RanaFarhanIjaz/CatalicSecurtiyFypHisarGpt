import urllib.request
import urllib.error
import time
import json
import sys

MODELS = [
    ("WhiteRabbitNeo-2.5", "WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B"),
    ("RedTeam_GPT", "darkknight25/REDTEAM_GPT"),
    ("Zephyr-7B-Beta", "HuggingFaceH4/zephyr-7b-beta"),
    ("Phi-3-Mini-Instruct", "microsoft/Phi-3-mini-4k-instruct"),
]

def check_model(name, model_id, token=""):
    url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    payload = json.dumps({"inputs": "Are you ready? Reply 'ONLINE'.", "parameters": {"max_new_tokens": 10}}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    req = urllib.request.Request(url, data=payload, headers=headers)
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            elapsed = int((time.time() - start) * 1000)
            data = resp.read().decode("utf-8", errors="replace")
            print(f"[{name}] HTTP {resp.status} in {elapsed}ms -> Response: {data[:120]}")
            return {"status": "online", "http_code": resp.status, "latency_ms": elapsed, "data": data[:120]}
    except urllib.error.HTTPError as e:
        elapsed = int((time.time() - start) * 1000)
        body = e.read().decode("utf-8", errors="replace")
        print(f"[{name}] HTTP {e.code} in {elapsed}ms -> Body: {body[:150]}")
        return {"status": "http_error", "http_code": e.code, "latency_ms": elapsed, "body": body[:150]}
    except Exception as e:
        print(f"[{name}] Connection error: {type(e).__name__}: {e}")
        return {"status": "connection_error", "error": str(e)}

if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, mid in MODELS:
        check_model(name, mid, token)
