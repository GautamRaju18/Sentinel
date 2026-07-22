"""The graph's state object.

Reducers matter here. A plain field is overwritten by whatever a node returns;
an annotated field is merged. Getting this wrong is the most common LangGraph
bug — parallel nodes silently clobber each other's writes because the field had
no reducer.

  messages   : append (add_messages dedupes by id and handles chunks)
  evidence   : append — investigation rounds accumulate, they do not replace
  errors     : append — we want the whole failure history, not the last one
  everything else : last write wins, which is what we want for a hypothesis
                    that gets refined
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

from sentinel.graph.schemas import (
    Critique,
    Evidence,
    Hypothesis,
    PostMortem,
    RemediationPlan,
    Triage,
    Verification,
)

Stage = Literal[
    "triage",
    "investigate",
    "synthesize",
    "critique",
    "plan",
    "await_approval",
    "execute",
    "verify",
    "postmortem",
    "done",
]


def merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Token counters add rather than replace, across parallel branches."""
    out = dict(left)
    for k, v in right.items():
        out[k] = out.get(k, 0) + v
    return out


class IncidentState(TypedDict, total=False):
    # --- identity ---
    incident_id: str
    scenario: str
    alert: str

    # --- conversation ---
    messages: Annotated[list, add_messages]

    # --- pipeline artefacts ---
    triage: Triage | None
    evidence: Annotated[list[Evidence], operator.add]
    hypothesis: Hypothesis | None
    critique: Critique | None
    plan: RemediationPlan | None
    verification: Verification | None
    postmortem: PostMortem | None

    # --- control ---
    stage: Stage
    loop_count: int
    # Questions the critic wants answered — steers the next investigation round.
    open_questions: list[str]
    # Set by the human at the approval interrupt. The agent cannot write this.
    approved: bool | None
    approval_note: str
    executed_actions: Annotated[list[str], operator.add]

    # --- retrieval ---
    runbook_context: str
    similar_incidents: str

    # --- bookkeeping ---
    errors: Annotated[list[str], operator.add]
    token_usage: Annotated[dict[str, int], merge_usage]
    tool_trajectory: Annotated[list[str], operator.add]
    security_flags: Annotated[list[str], operator.add]


def new_state(incident_id: str, scenario: str, alert: str) -> dict[str, Any]:
    return {
        "incident_id": incident_id,
        "scenario": scenario,
        "alert": alert,
        "messages": [],
        "evidence": [],
        "stage": "triage",
        "loop_count": 0,
        "open_questions": [],
        "approved": None,
        "approval_note": "",
        "executed_actions": [],
        "errors": [],
        "token_usage": {"input": 0, "output": 0},
        "tool_trajectory": [],
        "security_flags": [],
    }
