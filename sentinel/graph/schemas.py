"""Structured outputs.

Every model-produced artefact that another node consumes is a Pydantic model,
never free text. Free text between nodes is where agent systems rot: the next
node has to re-parse prose, and a wording change silently breaks a downstream
branch. A schema makes the contract explicit and lets the model be retried
against a validator.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Severity(StrEnum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class Category(StrEnum):
    BAD_DEPLOY = "bad_deploy"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIG_CHANGE = "config_change"
    EXPIRED_CREDENTIAL = "expired_credential"
    DEPENDENCY_FAILURE = "dependency_failure"
    UNKNOWN = "unknown"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Triage(BaseModel):
    """First classification of an incoming alert. Phase 6 fine-tunes this."""

    severity: Severity = Field(..., description="P1 total outage .. P4 cosmetic")
    category: Category = Field(..., description="Best guess at failure class")
    affected_service: str | None = Field(None, description="Primary service, if identifiable")
    needs_human: bool = Field(..., description="True if a human should be paged immediately")
    reasoning: str = Field(..., description="One sentence justification")


class Evidence(BaseModel):
    """A single grounded observation. `source` is what makes it checkable."""

    observation: str = Field(..., description="What was observed, with concrete values")
    source: str = Field(..., description="Tool that produced it, e.g. 'get_metric'")
    supports: str = Field(..., description="What this suggests about the cause")
    strength: Literal["strong", "moderate", "weak"] = "moderate"


class Hypothesis(BaseModel):
    """A causal explanation the critic will attack."""

    root_cause: str = Field(..., description="Causal chain from trigger to symptom")
    category: Category
    affected_service: str
    trigger: str | None = Field(
        None, description="Deploy id, config id, or event that started it, if any"
    )
    confidence: Confidence
    evidence_ids: list[int] = Field(
        default_factory=list, description="Indices into the evidence list"
    )
    unknowns: list[str] = Field(default_factory=list, description="What is still unverified")


class Critique(BaseModel):
    """The critic's verdict. This is what closes or continues the loop."""

    verdict: Literal["accept", "revise"] = Field(
        ..., description="accept when the causal chain is grounded and complete"
    )
    score: int = Field(..., ge=0, le=10, description="0 unsupported .. 10 airtight")
    gaps: list[str] = Field(
        default_factory=list, description="Specific unsupported claims or missing checks"
    )
    next_questions: list[str] = Field(
        default_factory=list, description="Concrete queries that would close the gaps"
    )
    alternative_causes: list[str] = Field(
        default_factory=list, description="Explanations that fit the evidence equally well"
    )
    reasoning: str = Field(..., description="Why this verdict")


class RemediationStep(BaseModel):
    action: Literal["rollback_deploy", "restart_service", "scale_service", "apply_config"]
    target: str = Field(..., description="Deploy id or service name")
    parameters: dict[str, str] = Field(default_factory=dict)
    rationale: str = Field(..., description="Why this step addresses the root cause")
    blast_radius: Literal["low", "medium", "high", "critical"] = "high"
    reversible: bool = True


class RemediationPlan(BaseModel):
    summary: str = Field(..., description="One line: what will be done")
    steps: list[RemediationStep] = Field(..., min_length=1)
    expected_effect: str = Field(..., description="What should change, and how to tell")
    risks: list[str] = Field(default_factory=list)
    rollback_plan: str = Field(..., description="How to undo this if it makes things worse")
    do_nothing_option: str = Field(
        ..., description="What happens if no action is taken — sometimes the right choice"
    )


class Verification(BaseModel):
    resolved: bool
    observations: list[str] = Field(default_factory=list)
    next_action: str = Field(..., description="What to do next if unresolved")


class PostMortem(BaseModel):
    title: str
    timeline: list[str] = Field(..., description="Timestamped events, earliest first")
    impact: str = Field(..., description="User-facing consequence")
    root_cause: str
    contributing_factors: list[str] = Field(default_factory=list)
    what_went_well: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(..., description="Concrete follow-ups")
    lesson: str = Field(..., description="The generalisable lesson, for the memory store")
