from typing import Literal, get_args
from pydantic import BaseModel, Field, field_validator, model_validator

AgentName = Literal[
    "Reconnaissance Agent",
    "Prompt Injection Agent",
    "System Prompt Extraction Agent",
    "RAG Security Agent",
    "Tool Testing Agent",
    "Output Validation Agent",
    "Memory Testing Agent",
]

PromptMode = Literal["predefined", "adaptive", "hybrid"]
RECON_AGENT = "Reconnaissance Agent"


def plan_counts(agent: str, mode: str, requests_per_agent: int, adaptive_per_agent: int) -> tuple[int, int]:
    if agent == RECON_AGENT:
        return (1 if mode == "adaptive" else requests_per_agent), 0
    predefined = requests_per_agent if mode in ("predefined", "hybrid") else 0
    adaptive = adaptive_per_agent if mode in ("adaptive", "hybrid") else 0
    return predefined, adaptive


class PentestAgentConfig(BaseModel):
    """Configuration for an open-source or self-trained pentesting agent."""
    agent_id: str = ""
    model_id: str = ""
    agent_type: Literal["huggingface_api", "local"] = "huggingface_api"
    model_type: Literal["huggingface_api", "local"] = "huggingface_api"
    base_url: str = ""
    api_token: str = ""
    label: str = ""
    description: str = ""

    @model_validator(mode="before")
    @classmethod
    def unify_agent_fields(cls, data):
        if isinstance(data, dict):
            m_id = data.get("model_id") or data.get("agent_id") or ""
            data["agent_id"] = m_id
            data["model_id"] = m_id

            m_type = data.get("agent_type") or data.get("model_type") or "huggingface_api"
            data["agent_type"] = m_type
            data["model_type"] = m_type
        return data

    @property
    def display_name(self) -> str:
        return self.label or self.agent_id or self.model_id


# Alias for backwards compatibility
FineTunedModelConfig = PentestAgentConfig


DEFAULT_PENTEST_AGENT_PRESETS = [
    {
        "agent_id": "WhiteRabbitNeo/WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B",
        "agent_type": "huggingface_api",
        "label": "WhiteRabbitNeo-2.5 (Cybersecurity & Pentest LLM)",
        "description": "Premier open-source offensive security & penetration testing LLM, trained specifically on exploit workflows, red-teaming, and boundary attacks.",
        "badge": "Offensive Pentest",
    },
    {
        "agent_id": "darkknight25/REDTEAM_GPT",
        "agent_type": "huggingface_api",
        "label": "RedTeam-GPT (Red Teaming & Exploitation LLM)",
        "description": "Fine-tuned open-source model specialized for autonomous red teaming, prompt injection payload crafting, and security assessment.",
        "badge": "Red Team",
    },
]

DEFAULT_HUGGINGFACE_PRESETS = DEFAULT_PENTEST_AGENT_PRESETS


class PriorInformation(BaseModel):
    """Known build handed to sub-agents in authenticated mode. Empty in unauthenticated mode."""
    source: Literal["none", "user", "system", "user+system"] = "none"
    role: str = Field(default="", max_length=200)
    system_prompt: str = Field(default="", max_length=2000)
    architecture: str = Field(default="", max_length=500)
    tools: list[str] = Field(default_factory=list, max_length=12)
    constraints: list[str] = Field(default_factory=list, max_length=12)
    documents: list[str] = Field(default_factory=list, max_length=12)
    notes: str = Field(default="", max_length=1000)
    system_fetch: Literal["not_requested", "ok", "unavailable"] = "not_requested"
    system_fetch_note: str = ""

    def has_content(self) -> bool:
        return bool(
            self.role.strip()
            or self.system_prompt.strip()
            or self.architecture.strip()
            or self.tools
            or self.constraints
            or self.documents
            or self.notes.strip()
        )


class AssessmentRequest(BaseModel):
    target_url: str
    authorization_confirmed: bool = False
    mode: Literal["unauthenticated", "authenticated"] = "unauthenticated"
    prior_source: Literal["none", "user", "system", "both"] = "both"
    prior_information: PriorInformation = Field(default_factory=PriorInformation)
    evaluator: Literal["demo", "ai"] = "ai"
    selected_agents: list[AgentName] | None = None
    prompt_mode: PromptMode = "hybrid"
    requests_per_agent: int = Field(default=1, ge=1, le=3)
    adaptive_per_agent: int = Field(default=1, ge=1, le=3)
    max_requests: int = Field(default=30, ge=1, le=60)
    include_openai_baseline: bool = True
    pentest_agents: list[PentestAgentConfig] = Field(default_factory=list)
    fine_tuned_models: list[PentestAgentConfig] = Field(default_factory=list)

    @field_validator("selected_agents")
    @classmethod
    def validate_selected_agents(cls, agents):
        if agents is not None and (not agents or len(agents) != len(set(agents))):
            raise ValueError("Select at least one agent and do not repeat agents.")
        return agents

    @model_validator(mode="before")
    @classmethod
    def map_legacy_fields(cls, data):
        if isinstance(data, dict):
            if "fine_tuned_models" in data and not data.get("pentest_agents"):
                data["pentest_agents"] = data["fine_tuned_models"]
            elif "pentest_agents" in data and not data.get("fine_tuned_models"):
                data["fine_tuned_models"] = data["pentest_agents"]
        return data

    @model_validator(mode="after")
    def validate_request_budget(self):
        agents = self.selected_agents if self.selected_agents is not None else list(get_args(AgentName))
        required_requests = sum(
            sum(plan_counts(a, self.prompt_mode, self.requests_per_agent, self.adaptive_per_agent))
            for a in agents
        )
        if required_requests > self.max_requests:
            raise ValueError(
                f"Selected agents need {required_requests} requests, "
                f"but the total request budget is {self.max_requests}."
            )
        if self.mode == "unauthenticated":
            self.prior_source = "none"
            self.prior_information = PriorInformation()
            return self
        if self.prior_source == "none":
            raise ValueError("Authenticated mode needs prior information from the user, the target system, or both.")
        if self.prior_source == "user" and not self.prior_information.has_content():
            raise ValueError(
                "Authenticated mode with user-supplied prior information needs a role, instructions, "
                "architecture, tool, constraint, document, or note."
            )
        return self


class Observation(BaseModel):
    agent: str
    test_name: str
    prompt: str
    response: str
    assessment: str = ""
    candidate: bool = False
    rationale: str = ""
    prompt_source: str = "predefined"
    reasoning: str = ""
    expected_safe_behavior: str = ""


ReviewStatus = Literal[
    "pending_human_review",
    "confirmed",
    "dismissed",
    "needs_more_evidence",
]


class FindingReview(BaseModel):
    status: ReviewStatus
    note: str = Field(default="", max_length=2000)


class RegistrySettings(BaseModel):
    enabled_agents: list[AgentName]
    target_url: str = "http://127.0.0.1:8001/chat"
    evaluator: Literal["demo", "ai"] = "demo"
    mode: Literal["unauthenticated", "authenticated"] = "unauthenticated"
    prompt_mode: PromptMode = "hybrid"
    requests_per_agent: int = Field(default=1, ge=1, le=3)
    adaptive_per_agent: int = Field(default=1, ge=1, le=3)
    max_requests: int = Field(default=30, ge=1, le=60)

    @field_validator("enabled_agents")
    @classmethod
    def validate_enabled_agents(cls, agents):
        if not agents or len(agents) != len(set(agents)):
            raise ValueError("Enable at least one agent and do not repeat agents.")
        return agents

    @field_validator("target_url")
    @classmethod
    def validate_target_url(cls, value):
        cleaned = value.strip()
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("Target URL must start with http:// or https://.")
        return cleaned


class TargetBrief(BaseModel):
    """Handed to every sub-agent. Unauthenticated briefs come only from reconnaissance.
    Authenticated briefs also carry the known build supplied by the user or the target system.
    """
    domain: str
    summary: str
    observed_traits: list[str]
    focus_areas: list[str]
    knowledge_mode: Literal["unauthenticated", "authenticated"] = "unauthenticated"
    prior_source: Literal["none", "user", "system", "user+system"] = "none"
    known_role: str = ""
    known_system_prompt: str = ""
    architecture: str = ""
    known_tools: list[str] = Field(default_factory=list)
    known_constraints: list[str] = Field(default_factory=list)
    known_documents: list[str] = Field(default_factory=list)
    prior_notes: str = ""


class GeneratedTest(BaseModel):
    reasoning: str
    prompt: str
    expected_safe_behavior: str


class GeneratedPlan(BaseModel):
    tests: list[GeneratedTest]


class PentesterResult(BaseModel):
    """Assessment results produced by a specific pentesting agent against the target model."""
    pentester_id: str
    pentester_label: str
    pentester_type: str                        # "openai_baseline", "huggingface_api", "local"
    target_score: int                          # Target model score under this pentester (0-100)
    target_grade: str                          # Target model resilience grade
    vulnerabilities_found: int                 # Number of candidate vulnerabilities caught
    observations: list[Observation]
    pentester_summary: str
    error: str = ""
    prompt_origin: str = "baseline"
    target_responses: int = 0
    target_errors: int = 0
    model_status: str = ""


# Alias for backwards compatibility
ModelResult = PentesterResult


class ResearchSection(BaseModel):
    """Combined cross-pentester research analysis comparing testing agents on the target model."""
    target_url: str
    pentester_results: list[PentesterResult]
    comparative_summary: str
    most_effective_pentester: str = ""
    target_resilience_rating: str = ""
    total_pentesters: int
    pentesters_with_errors: int


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
    prompt_mode: str = "hybrid"
    target_brief: dict | None = None
    review_status: str = "pending_human_review"
    report_path: str | None = None
    research_section: dict | None = None
