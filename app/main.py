from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from .schemas import AssessmentRequest
from .orchestrator import run_assessment
from .reporting import create_pdf
from .policy import PolicyError

app = FastAPI(title="HisarGPT", version="0.1.0",
              description="Human-supervised, localhost-only LLM assessment MVP")

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
        path = create_pdf(data)
        data["report_path"] = path
        return data
    except PolicyError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {type(e).__name__}: {e}")

@app.get("/reports/{assessment_id}")
def get_report(assessment_id: str):
    from pathlib import Path
    path = Path("reports") / f"{assessment_id}.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path, media_type="application/pdf", filename=path.name)
