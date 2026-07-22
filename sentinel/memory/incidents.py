"""Long-term memory: what past incidents taught us.

Every completed incident writes a post-mortem into pgvector. Future
investigations retrieve them, so the system's second encounter with a class of
failure starts from its first.

What gets stored is deliberately not the raw transcript. A transcript is long,
mostly noise, and retrieves badly. The stored document is the distilled shape of
the incident — symptom, cause, fix, lesson — which is both what an engineer
would want to read and what embeds usefully.
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from sentinel.graph.schemas import PostMortem
from sentinel.logging_setup import get_logger
from sentinel.rag.store import INCIDENT_COLLECTION, get_vector_store

log = get_logger(__name__)


def _to_document(state: dict[str, Any], pm: PostMortem) -> Document:
    triage = state.get("triage")
    hypothesis = state.get("hypothesis")
    verification = state.get("verification")

    body = "\n".join(
        [
            f"INCIDENT: {pm.title}",
            f"category: {getattr(triage, 'category', 'unknown')}",
            f"severity: {getattr(triage, 'severity', 'unknown')}",
            f"service: {getattr(hypothesis, 'affected_service', 'unknown')}",
            "",
            "SYMPTOM:",
            state.get("alert", "")[:600],
            "",
            "ROOT CAUSE:",
            pm.root_cause,
            "",
            "WHAT FIXED IT:",
            "\n".join(state.get("executed_actions") or ["no action taken"]),
            "",
            f"RESOLVED: {getattr(verification, 'resolved', False)}",
            "",
            "LESSON:",
            pm.lesson,
            "",
            "CONTRIBUTING FACTORS:",
            "\n".join(f"- {f}" for f in pm.contributing_factors) or "- none recorded",
        ]
    )
    return Document(
        page_content=body,
        metadata={
            "incident_id": state.get("incident_id", ""),
            "scenario": state.get("scenario", ""),
            "category": str(getattr(triage, "category", "unknown")),
            "severity": str(getattr(triage, "severity", "unknown")),
            "resolved": bool(getattr(verification, "resolved", False)),
        },
    )


async def remember_incident(state: dict[str, Any], pm: PostMortem) -> bool:
    from sentinel.db import to_thread

    try:
        store = get_vector_store(INCIDENT_COLLECTION)
        await to_thread(store.add_documents, [_to_document(state, pm)])
        log.info("memory.incident_stored", incident=state.get("incident_id"))
        return True
    except Exception as e:
        log.warning("memory.store_failed", error=str(e))
        return False


async def search_incidents(query: str, k: int = 3) -> list[Document]:
    from sentinel.db import to_thread

    try:
        store = get_vector_store(INCIDENT_COLLECTION)
        return await to_thread(store.similarity_search, query, k)
    except Exception as e:
        log.warning("memory.search_failed", error=str(e))
        return []
