# HisarGPT — starter implementation

This is a runnable **local-only MVP scaffold** for the Deliverable-II design:
- FastAPI operator API
- Main orchestrator and seven specialist agents using the OpenAI Agents SDK
- **Hybrid prompting:** every sub-agent can run fixed predefined prompts *and* reason over the main
  agent's brief to generate its own prompts at test time
- Two localhost target endpoints (configure the target URL)
- Shared policy gate: localhost-only, explicit authorization flag, request budget
- Rule-based AI Judge scorecard
- JSON evidence and basic PDF report export

It is a starting implementation, not a validated security scanner. Findings remain candidates
until a human reviewer confirms them. Use only local test applications you own or have written
permission to assess.

## 1. Requirements
- Python 3.11+
- An OpenAI API key for the OpenAI-backed target and optional AI-assisted evaluation.
- Use only a localhost target you own or are authorized to assess.

## 2. Setup (Windows PowerShell)
```powershell
cd HisarGPT
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```
Put your rotated `OPENAI_API_KEY` in `.env`. Never put a real key in `.env.example` or commit
`.env`. The same key can be used by both local services.

## 3. Start the OpenAI-backed target (Terminal 1)
```powershell
uvicorn target_system:app --host 127.0.0.1 --port 8001 --reload
```
This target uses `HISARGPT_TARGET_MODEL` (defaults to `HISARGPT_MODEL`, then `gpt-4.1-mini`)
and a built-in system prompt. Override the target prompt with `HISARGPT_TARGET_SYSTEM_PROMPT`.
Its `/health` endpoint reports whether an API key is configured.

The no-API mock is still available instead:
```powershell
uvicorn demo_target:app --host 127.0.0.1 --port 8001 --reload
```
Run one target service at a time; both use port 8001 by default.

## 4. Start HisarGPT API (Terminal 2)
```powershell
uvicorn app.main:app --port 8000 --reload
```
Open `http://127.0.0.1:8000` for the frontend. Choose **Local heuristic** to run the included
assessment prompts with the local response heuristic. The OpenAI-backed target still uses your
key. Choose **AI-assisted** to also use the specialist agents and AI Judge, which makes
additional OpenAI API calls.

## 5. Run an assessment through the API
Alternatively, open `http://127.0.0.1:8000/docs` and call `POST /assessments/run` with:
```json
{
  "target_url": "http://127.0.0.1:8001/chat",
  "authorization_confirmed": true,
  "max_requests": 14,
  "mode": "unauthenticated",
  "evaluator": "demo"
}
```
The app runs bounded checks against the authorized local target and generates a PDF report.
The evaluator can be `"demo"` (local heuristic) or `"ai"` (requires an API key); API requests
default to `"ai"` for backwards compatibility. Every result is pending human review.

## Prompt strategy (`prompt_mode`)
| Mode | What runs |
|---|---|
| `predefined` | Fixed prompts only (comparable across runs). |
| `adaptive` | Reconnaissance (1 predefined probe) → main agent builds a brief → each sub-agent writes its own prompts. |
| `hybrid` (default) | Predefined prompts first, then adaptive prompts per agent. |

Flow: **Recon (predefined) → main agent builds `TargetBrief` → each sub-agent receives the brief +
findings so far (including its own predefined result) → generates `adaptive_per_agent` new prompts →
prompts are validated and sent through the same policy gate and budget.**

- Budget = per agent `requests_per_agent` (predefined) + `adaptive_per_agent` (adaptive); recon is
  always predefined-only. Default hybrid on all 7 agents = 13 requests.
- Generated prompts must be harmless boundary probes: no URLs, code blocks, or requests for
  credentials/secrets, max 600 chars, no duplicates. Rejected prompts are replaced by templates.
- Each adaptive prompt may carry a random canary token (`HG-XXXXXX`) for a transparent heuristic check.
- With the **Local heuristic** evaluator no extra AI calls are made: adaptive prompts come from
  domain-aware templates (`adaptive-template`). With **AI-assisted**, the main agent and sub-agents
  generate them (`adaptive`), falling back to templates if generation fails.
- Each observation records `prompt_source`, the sub-agent's `reasoning`, and `expected_safe_behavior`.

## Project layout
- `app/agents.py`: seven specialized agents, AI Judge, main-agent brief, adaptive prompt generation
- `app/orchestrator.py`: main orchestration workflow
- `app/policy.py`: fail-closed local-target policy gate and request budget
- `app/target.py`: scoped HTTP adapter
- `app/reporting.py`: PDF report generator
- `app/main.py`: FastAPI endpoints and frontend
- `app/static/index.html`: single-page assessment interface
- `target_system.py`: OpenAI-backed localhost target API
- `demo_target.py`: deliberately simple local target fixture

## Important
The current version intentionally does not include Firebase authentication/history, web search,
credentialed target access, or calibrated scoring. Automated candidates require human review;
the score is a triage aid, not proof of a vulnerability.
