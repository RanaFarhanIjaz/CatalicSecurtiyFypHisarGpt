import asyncio
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

from . import store as store_mod
from .orchestrator import run_assessment
from .policy import PolicyError
from .reporting import create_pdf
from .pentest_agent import huggingface_inference_url
from .prior import PriorInformationError
from .schemas import AssessmentRequest, FindingReview, RegistrySettings, DEFAULT_PENTEST_AGENT_PRESETS

app = FastAPI(title="HisarGPT", version="0.2.0",
              description="Human-supervised, localhost-only LLM assessment MVP. Powered by Catalic Security.")


def _created_at() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

@app.get("/health")
def health():
    return {"project": "HisarGPT", "status": "running", "docs": "/docs"}

@app.post("/assessments/run")
async def start_assessment(req: AssessmentRequest):
    try:
        result = await run_assessment(req)
        result_dict = result if isinstance(result, dict) else result
        # Pydantic models are serialized explicitly for report generation.
        data = result_dict.copy()
        data["observations"] = [
            o.model_dump() if hasattr(o, "model_dump") else o for o in data["observations"]
        ]
        data["created_at"] = _created_at()
        data = store_mod.store.save_assessment(data)
        path = create_pdf(data)
        data["report_path"] = path
        return data
    except PolicyError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except PriorInformationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {type(e).__name__}: {e}")

@app.get("/assessments")
def list_assessments():
    return {"assessments": store_mod.store.list_assessments()}

@app.get("/assessments/{assessment_id}")
def get_assessment(assessment_id: str):
    record = store_mod.store.get_assessment(assessment_id)
    if not record:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return record

@app.get("/findings")
def list_findings():
    findings = store_mod.store.list_findings()
    return {
        "findings": findings,
        "candidate_count": sum(1 for item in findings if item.get("candidate")),
    }

@app.patch("/findings/{finding_id}")
def review_finding(finding_id: str, review: FindingReview):
    try:
        return store_mod.store.review_finding(finding_id, review.status, review.note.strip())
    except KeyError:
        raise HTTPException(status_code=404, detail="Finding not found")

@app.get("/registry")
def get_registry():
    return store_mod.store.get_registry()

@app.put("/registry")
def save_registry(settings: RegistrySettings):
    try:
        return store_mod.store.save_registry(settings)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

class CheckStatusRequest(BaseModel):
    model_id: str = ""
    agent_id: str = ""
    model_type: str = "huggingface_api"
    agent_type: str = "huggingface_api"
    base_url: str = ""
    api_token: str = ""


@app.post("/models/fine-tuned/check-status")
async def check_fine_tuned_status(req: CheckStatusRequest):
    """Pings a Hugging Face or local model endpoint to check if it is online and responding."""
    target_id = req.model_id or req.agent_id
    m_type = req.model_type or req.agent_type or "huggingface_api"
    clean_url = (req.base_url or "").strip()
    token = (req.api_token or "").strip()

    if not target_id and not clean_url:
        return {
            "status": "error",
            "message": "Specify a Model ID or an Endpoint URL to check.",
            "live": False,
        }

    # If it's a local endpoint
    if m_type == "local" or (clean_url and ("127.0.0.1" in clean_url or "localhost" in clean_url)):
        probe_url = clean_url or "http://127.0.0.1:8002/chat"
        try:
            start_t = asyncio.get_event_loop().time()
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.post(probe_url, json={"message": "ping"})
                elapsed = int((asyncio.get_event_loop().time() - start_t) * 1000)
                if res.status_code in (200, 422):
                    return {
                        "status": "online",
                        "live": True,
                        "code": res.status_code,
                        "latency_ms": elapsed,
                        "message": f"Local model server at {probe_url} is active and responding ({elapsed}ms).",
                    }
                return {
                    "status": "http_error",
                    "live": False,
                    "code": res.status_code,
                    "message": f"Local server returned HTTP {res.status_code}.",
                }
        except Exception as e:
            return {
                "status": "offline",
                "live": False,
                "message": f"Cannot connect to local model at {probe_url}: {e}",
            }

    # Hugging Face endpoint probe
    endpoint = clean_url if clean_url else f"https://router.huggingface.co/hf-inference/models/{target_id}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    start_t = asyncio.get_event_loop().time()
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                endpoint,
                json={"inputs": "Status probe. Reply ONLINE.", "parameters": {"max_new_tokens": 10, "return_full_text": False}},
                headers=headers
            )
            elapsed = int((asyncio.get_event_loop().time() - start_t) * 1000)

            if resp.status_code == 200:
                data = resp.json()
                sample = ""
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    sample = str(data[0]["generated_text"]).strip()
                elif isinstance(data, dict) and "generated_text" in data:
                    sample = str(data["generated_text"]).strip()
                return {
                    "status": "online",
                    "live": True,
                    "code": 200,
                    "latency_ms": elapsed,
                    "sample": sample[:120],
                    "message": f"🟢 Model is LIVE on Hugging Face and actively responding ({elapsed}ms).",
                }
            elif resp.status_code == 503:
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                est_time = data.get("estimated_time", 20.0) if isinstance(data, dict) else 20.0
                return {
                    "status": "loading",
                    "live": False,
                    "code": 503,
                    "message": f"🟡 Model is currently loading into Hugging Face GPU memory (~{int(est_time)}s). It will be ready shortly.",
                    "estimated_time": est_time,
                }
            elif resp.status_code in (401, 403):
                return {
                    "status": "auth_required",
                    "live": False,
                    "code": resp.status_code,
                    "message": "🔑 Gated or private model on Hugging Face. Add an optional API token, or leave blank to use the built-in curated pentest engine.",
                    "offline_ready": True,
                }
            elif resp.status_code == 404:
                return {
                    "status": "not_hosted",
                    "live": False,
                    "code": 404,
                    "message": "🟣 Serverless API is not enabled on Hugging Face for this model. HisarGPT's built-in curated pentest engine will run instead.",
                    "offline_ready": True,
                }
            else:
                return {
                    "status": "http_error",
                    "live": False,
                    "code": resp.status_code,
                    "message": f"Hugging Face returned HTTP {resp.status_code}. Built-in curated pentest engine is ready as backup.",
                    "offline_ready": True,
                }
    except Exception as e:
        return {
            "status": "offline_ready",
            "live": False,
            "message": "🟣 No internet or Hugging Face API unreachable. HisarGPT's built-in curated pentest engine is ACTIVE and ready to test your target.",
            "offline_ready": True,
        }


@app.get("/models/fine-tuned/presets")
def list_fine_tuned_presets():
    """Returns pre-configured open-source fine-tuned pentesting models from Hugging Face."""
    return {"presets": DEFAULT_PENTEST_AGENT_PRESETS}


@app.get("/models/fine-tuned/types")
def list_fine_tuned_model_types():
    """Returns the supported pentesting agent model types and their configuration schema."""
    return {
        "supported_types": [
            {
                "type": "huggingface_api",
                "description": "Open-source offensive security & pentesting model from Hugging Face (e.g. WhiteRabbitNeo, RedTeam-GPT)",
                "example_model_ids": [
                    "WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B",
                    "darkknight25/REDTEAM_GPT",
                    "rombodawg/Open-Cybersecurity-LLM-v1"
                ],
                "requires_base_url": False,
                "requires_api_token": "optional (required for private or gated models)",
            },
            {
                "type": "local",
                "description": "Self-trained pentesting LLM hosted locally and queried via HTTP",
                "example_model_ids": ["my-custom-redteam-llm", "finetuned-pentest-agent"],
                "requires_base_url": True,
                "requires_api_token": "optional",
            },
        ]
    }

@app.get("/reports/{assessment_id}")
def get_report(assessment_id: str):
    path = Path("reports") / f"{assessment_id}.pdf"
    if not path.exists():
        record = store_mod.store.get_assessment(assessment_id)
        if not record:
            raise HTTPException(status_code=404, detail="Report not found")
        path = Path(create_pdf(record))
    return FileResponse(path, media_type="application/pdf", filename=f"HisarGPT-{assessment_id}.pdf")
