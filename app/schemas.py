from typing import Literal
from pydantic import BaseModel, Field, HttpUrl

class AssessmentRequest(BaseModel):
    target_url: str
    authorization_confirmed: bool = False
    mode: Literal["unauthenticated", "authenticated"] = "unauthenticated"
    evaluator: Literal["demo", "ai"] = "ai"
    max_requests: int = Field(default=14, ge=1, le=30)

class Observation(BaseModel):
    agent: str
    test_name: str
    prompt: str
    response: str
    candidate: bool = False
    rationale: str = ""

class AssessmentResult(BaseModel):
    assessment_id: str
    target_url: str
    mode: str
    evaluator: str = "ai"
    total_requests: int
    score: int
    grade: str
    observations: list[Observation]
    judge_summary: str
    review_status: str = "pending_human_review"
    report_path: str | None = None
