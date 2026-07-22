"""Alert triage — the swappable classifier.

Two backends behind one function, chosen by TRIAGE_BACKEND in .env:

  baseline  — a general instruction model, prompted with the schema
  finetuned — the Phase 6 QLoRA model, which was trained on this exact task

Keeping them behind one interface is what makes the fine-tuning result
measurable: evals/ runs the same test set through both and compares accuracy,
latency and cost. Without this seam, "the fine-tune helped" would be a claim
rather than a number.
"""

from __future__ import annotations

import time

from sentinel.agents.prompts import TRIAGE_SYSTEM
from sentinel.config import get_settings
from sentinel.graph.schemas import Category, Severity, Triage
from sentinel.logging_setup import get_logger
from sentinel.models.router import ModelTier, get_model
from sentinel.models.structured import generate_structured

log = get_logger(__name__)

_FALLBACK = Triage(
    severity=Severity.P2,
    category=Category.UNKNOWN,
    affected_service=None,
    needs_human=True,
    # When classification fails we escalate rather than guess. An unclassified
    # alert going to a human is a minor annoyance; a P1 silently filed as P4 is
    # an outage nobody is working on.
    reasoning="Classification failed; defaulting to human review.",
)


async def classify(alert_text: str) -> Triage:
    settings = get_settings()
    started = time.perf_counter()

    result = await generate_structured(
        get_model(ModelTier.TRIAGE),
        Triage,
        f"Classify this alert:\n\n{alert_text}",
        system=TRIAGE_SYSTEM,
        max_attempts=2,
        default=_FALLBACK,
    )

    log.info(
        "triage.classified",
        backend=settings.triage_backend,
        severity=str(result.severity),
        category=str(result.category),
        ms=round((time.perf_counter() - started) * 1000),
    )
    return result


async def classify_batch(alerts: list[str]) -> list[Triage]:
    """Classify many alerts concurrently — used by the eval harness."""
    import asyncio

    return list(await asyncio.gather(*[classify(a) for a in alerts]))
