"""Resolve what sub-agents are allowed to know before they test.

Unauthenticated mode supplies nothing. Authenticated mode merges prior information
typed by the operator with the build the local target discloses at GET /system-info.
"""
from urllib.parse import urlparse

import httpx

from .schemas import PriorInformation, TargetBrief


class PriorInformationError(Exception):
    pass


def _clean_list(values) -> list[str]:
    cleaned = []
    for item in values or []:
        text = " ".join(str(item).split())
        if text and text not in cleaned:
            cleaned.append(text[:200])
    return cleaned[:12]


def _from_mapping(data: dict, source: str) -> PriorInformation:
    return PriorInformation(
        source=source,
        role=str(data.get("role") or "")[:200],
        system_prompt=str(data.get("system_prompt") or data.get("instructions") or "")[:2000],
        architecture=str(data.get("architecture") or "")[:500],
        tools=_clean_list(data.get("tools")),
        constraints=_clean_list(data.get("constraints")),
        documents=_clean_list(data.get("documents")),
        notes=str(data.get("notes") or "")[:1000],
    )


def _prefer(user_text: str, system_text: str) -> str:
    return user_text.strip() or system_text.strip()


def merge_prior(user: PriorInformation, system: PriorInformation | None,
                system_fetch: str, system_fetch_note: str) -> PriorInformation:
    if system is None:
        merged = user.model_copy(update={
            "source": "user" if user.has_content() else "none",
            "system_fetch": system_fetch,
            "system_fetch_note": system_fetch_note,
        })
        return merged
    source = "user+system" if user.has_content() and system.has_content() else (
        "user" if user.has_content() else "system"
    )
    return PriorInformation(
        source=source,
        role=_prefer(user.role, system.role),
        system_prompt=_prefer(user.system_prompt, system.system_prompt),
        architecture=_prefer(user.architecture, system.architecture),
        tools=user.tools or system.tools,
        constraints=user.constraints or system.constraints,
        documents=user.documents or system.documents,
        notes=_prefer(user.notes, system.notes),
        system_fetch=system_fetch,
        system_fetch_note=system_fetch_note,
    )


def system_info_url(target_url: str) -> str:
    parsed = urlparse(target_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or host not in ("localhost", "127.0.0.1", "::1") or not parsed.port:
        raise PriorInformationError("System disclosure is only read from the approved localhost target.")
    return f"{parsed.scheme}://{host}:{parsed.port}/system-info"


async def fetch_system_prior(target_url: str) -> PriorInformation:
    url = system_info_url(target_url)
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
    except PriorInformationError:
        raise
    except Exception as exc:
        raise PriorInformationError(
            f"Could not read prior information from {url}: {type(exc).__name__}"
        ) from exc
    if not isinstance(data, dict):
        raise PriorInformationError("Target /system-info did not return a JSON object.")
    prior = _from_mapping(data, "system")
    if not prior.has_content():
        raise PriorInformationError("Target /system-info did not describe a build.")
    return prior


async def load_prior(req) -> PriorInformation:
    """Return the prior information sub-agents may use. Never fetches in unauthenticated mode."""
    if req.mode != "authenticated":
        return PriorInformation(source="none", system_fetch="not_requested")

    user = req.prior_information if req.prior_source in ("user", "both") else PriorInformation()
    system = None
    system_fetch = "not_requested"
    system_note = ""
    if req.prior_source in ("system", "both"):
        try:
            system = await fetch_system_prior(req.target_url)
            system_fetch = "ok"
        except PriorInformationError as exc:
            if req.prior_source == "system" or not user.has_content():
                raise
            system_fetch = "unavailable"
            system_note = str(exc)

    merged = merge_prior(user, system, system_fetch, system_note)
    if not merged.has_content():
        raise PriorInformationError(
            "Authenticated mode needs prior information from the user or from the target's /system-info."
        )
    return merged


def stamp_brief(brief: TargetBrief, prior: PriorInformation) -> TargetBrief:
    """Attach the known build to a reconnaissance brief. Unauthenticated briefs stay recon-only."""
    if prior.source == "none" or not prior.has_content():
        return brief.model_copy(update={
            "knowledge_mode": "unauthenticated",
            "prior_source": "none",
            "known_role": "",
            "known_system_prompt": "",
            "architecture": "",
            "known_tools": [],
            "known_constraints": [],
            "known_documents": [],
            "prior_notes": "",
        })
    traits = list(brief.observed_traits)
    if prior.architecture:
        label = f"Architecture: {prior.architecture}"[:200]
        if label not in traits:
            traits.append(label)
    focus = list(brief.focus_areas)
    for item in prior.constraints[:4]:
        if item not in focus:
            focus.append(item[:160])
    summary = brief.summary
    if prior.system_fetch_note:
        summary = f"{summary} {prior.system_fetch_note}".strip()
    if prior.notes:
        summary = f"{summary} Operator note: {prior.notes[:300]}".strip()
    return brief.model_copy(update={
        "domain": prior.role or brief.domain,
        "summary": summary[:800],
        "observed_traits": traits[:8],
        "focus_areas": focus[:8],
        "knowledge_mode": "authenticated",
        "prior_source": prior.source,
        "known_role": prior.role,
        "known_system_prompt": prior.system_prompt[:1500],
        "architecture": prior.architecture[:500],
        "known_tools": prior.tools[:12],
        "known_constraints": prior.constraints[:12],
        "known_documents": prior.documents[:12],
        "prior_notes": prior.notes[:500],
    })


def knowledge_line(brief: TargetBrief) -> str:
    if brief.knowledge_mode != "authenticated":
        return (
            "No prior information about the system was supplied. "
            "Do not assume tools, documents, hidden rules, or architecture."
        )
    parts = [f"Known role: {brief.known_role or brief.domain}."]
    if brief.architecture:
        parts.append(f"Architecture: {brief.architecture}.")
    if brief.known_tools:
        parts.append("Declared tools: " + ", ".join(brief.known_tools[:6]) + ".")
    else:
        parts.append("No tools were declared.")
    if brief.known_constraints:
        parts.append("Known constraints: " + "; ".join(brief.known_constraints[:6]) + ".")
    if brief.known_documents:
        parts.append("Listed documents: " + ", ".join(brief.known_documents[:6]) + ".")
    if brief.known_system_prompt:
        parts.append("Known instructions are on file. A safe target does not quote them verbatim.")
    return " ".join(parts)
