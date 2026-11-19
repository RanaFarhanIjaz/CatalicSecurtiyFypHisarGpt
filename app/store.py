"""Local JSON workspace: assessment history, finding reviews, agent registry."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .agents import AGENT_SPECS
from .schemas import RegistrySettings

SHORT_NAMES = {
    "Reconnaissance Agent": "Reconnaissance",
    "Prompt Injection Agent": "Prompt injection",
    "System Prompt Extraction Agent": "System prompt",
    "RAG Security Agent": "RAG security",
    "Tool Testing Agent": "Tool boundaries",
    "Output Validation Agent": "Output validation",
    "Memory Testing Agent": "Memory",
}

TRIAGE = {
    "Prompt Injection Agent": "High",
    "System Prompt Extraction Agent": "High",
    "RAG Security Agent": "Medium",
    "Tool Testing Agent": "Medium",
    "Memory Testing Agent": "Medium",
    "Output Validation Agent": "Low",
    "Reconnaissance Agent": "Info",
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_registry() -> dict:
    return RegistrySettings(
        enabled_agents=[name for name, _ in AGENT_SPECS],
    ).model_dump()


class Store:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else Path("data")
        self._lock = threading.Lock()

    def _read(self, name: str, fallback):
        path = self.root / name
        if not path.exists():
            return fallback() if callable(fallback) else fallback
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return fallback() if callable(fallback) else fallback

    def _write(self, name: str, payload) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)

    def save_assessment(self, data: dict) -> dict:
        record = dict(data)
        record.setdefault("created_at", _now())
        observations = []
        for index, item in enumerate(record.get("observations") or []):
            observation = dict(item)
            observation["finding_id"] = f"{record['assessment_id']}:{index}"
            observations.append(observation)
        record["observations"] = observations
        with self._lock:
            history = self._read("history.json", lambda: {"assessments": []})
            assessments = [a for a in history.get("assessments", [])
                           if a.get("assessment_id") != record["assessment_id"]]
            assessments.insert(0, record)
            self._write("history.json", {"assessments": assessments[:100]})
        return record

    def list_assessments(self) -> list[dict]:
        with self._lock:
            history = self._read("history.json", lambda: {"assessments": []})
        summaries = []
        for item in history.get("assessments", []):
            observations = item.get("observations") or []
            summaries.append({
                "assessment_id": item.get("assessment_id"),
                "created_at": item.get("created_at"),
                "target_url": item.get("target_url"),
                "mode": item.get("mode"),
                "evaluator": item.get("evaluator"),
                "prompt_mode": item.get("prompt_mode"),
                "total_requests": item.get("total_requests"),
                "score": item.get("score"),
                "grade": item.get("grade"),
                "review_status": item.get("review_status", "pending_human_review"),
                "candidate_count": sum(1 for o in observations if o.get("candidate")),
                "observation_count": len(observations),
                "report_ready": True,
            })
        return summaries

    def get_assessment(self, assessment_id: str) -> dict | None:
        with self._lock:
            history = self._read("history.json", lambda: {"assessments": []})
            reviews = self._read("reviews.json", lambda: {})
        for item in history.get("assessments", []):
            if item.get("assessment_id") == assessment_id:
                record = dict(item)
                record["observations"] = [
                    self._with_review(o, reviews) for o in item.get("observations") or []
                ]
                return record
        return None

    def list_findings(self) -> list[dict]:
        with self._lock:
            history = self._read("history.json", lambda: {"assessments": []})
            reviews = self._read("reviews.json", lambda: {})
        findings = []
        for item in history.get("assessments", []):
            for observation in item.get("observations") or []:
                finding = self._with_review(observation, reviews)
                finding.update({
                    "assessment_id": item.get("assessment_id"),
                    "created_at": item.get("created_at"),
                    "target_url": item.get("target_url"),
                    "triage": TRIAGE.get(observation.get("agent"), "Info"),
                    "agent_short": SHORT_NAMES.get(observation.get("agent"), observation.get("agent")),
                })
                findings.append(finding)
        return findings

    def review_finding(self, finding_id: str, status: str, note: str) -> dict:
        with self._lock:
            history = self._read("history.json", lambda: {"assessments": []})
            match = None
            for item in history.get("assessments", []):
                for observation in item.get("observations") or []:
                    if observation.get("finding_id") == finding_id:
                        match = observation
                        break
                if match:
                    break
            if not match:
                raise KeyError(finding_id)
            reviews = self._read("reviews.json", lambda: {})
            reviews[finding_id] = {"status": status, "note": note, "updated_at": _now()}
            self._write("reviews.json", reviews)
        reviewed = self._with_review(match, reviews)
        reviewed["triage"] = TRIAGE.get(match.get("agent"), "Info")
        return reviewed

    def get_registry(self) -> dict:
        with self._lock:
            saved = self._read("registry.json", default_registry)
        try:
            settings = RegistrySettings.model_validate(saved).model_dump()
        except Exception:
            settings = default_registry()
        enabled = set(settings["enabled_agents"])
        agents = []
        for name, task in AGENT_SPECS:
            agents.append({
                "name": name,
                "short": SHORT_NAMES.get(name, name),
                "task": task,
                "triage": TRIAGE.get(name, "Info"),
                "enabled": name in enabled,
                "predefined_prompts": 3,
            })
        return {"agents": agents, "defaults": settings}

    def save_registry(self, settings: RegistrySettings) -> dict:
        payload = settings.model_dump()
        with self._lock:
            self._write("registry.json", payload)
        return self.get_registry()

    @staticmethod
    def _with_review(observation: dict, reviews: dict) -> dict:
        finding = dict(observation)
        review = reviews.get(observation.get("finding_id"), {})
        finding["review_status"] = review.get("status", "pending_human_review")
        finding["review_note"] = review.get("note", "")
        finding["reviewed_at"] = review.get("updated_at")
        return finding


store = Store()
