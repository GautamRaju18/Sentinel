"""Graph topology and compilation.

The whole control flow of the system is in one screen of code below, which is
the argument for LangGraph over a hand-rolled loop: the cycle, the branch and
the interrupt are declared, not buried in conditionals.

    triage → retrieve → investigate → synthesize → critique
                            ↑                          │
                            └──────── revise ──────────┤
                                                       ↓ accept
                                                     plan
                                                       ↓
                                            [INTERRUPT: human approves]
                                                       ↓
                                          approval → execute → verify
                                               │                  │
                                               └── rejected ──────┴→ postmortem → END
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from sentinel.graph.nodes import (
    approval_node,
    critique_node,
    execute_node,
    investigate_node,
    plan_node,
    postmortem_node,
    retrieve_node,
    route_after_approval,
    route_after_critique,
    synthesize_node,
    triage_node,
    verify_node,
)
from sentinel.graph.state import IncidentState
from sentinel.logging_setup import get_logger

log = get_logger(__name__)


def build_graph() -> StateGraph:
    g = StateGraph(IncidentState)

    g.add_node("triage", triage_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("investigate", investigate_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("critique", critique_node)
    g.add_node("plan", plan_node)
    g.add_node("approval", approval_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("postmortem", postmortem_node)

    g.add_edge(START, "triage")
    g.add_edge("triage", "retrieve")
    g.add_edge("retrieve", "investigate")
    g.add_edge("investigate", "synthesize")
    g.add_edge("synthesize", "critique")

    # The reflection cycle: the critic can send us back for more evidence.
    g.add_conditional_edges(
        "critique",
        route_after_critique,
        {"investigate": "investigate", "plan": "plan"},
    )

    g.add_edge("plan", "approval")
    g.add_conditional_edges(
        "approval",
        route_after_approval,
        {"execute": "execute", "postmortem": "postmortem"},
    )
    g.add_edge("execute", "verify")
    g.add_edge("verify", "postmortem")
    g.add_edge("postmortem", END)

    return g


# Execution halts BEFORE this node. Everything downstream of it mutates
# production, so the gate sits at the boundary rather than inside execute.
INTERRUPT_BEFORE = ["approval"]


def compile_graph(checkpointer=None, *, with_interrupt: bool = True):
    return build_graph().compile(
        checkpointer=checkpointer or MemorySaver(),
        interrupt_before=INTERRUPT_BEFORE if with_interrupt else [],
    )


@asynccontextmanager
async def postgres_graph(*, with_interrupt: bool = True):
    """Compile with a Postgres checkpointer.

    Durable checkpoints are what make the approval gate usable in reality: the
    graph pauses, the process can exit, and an operator can resume the incident
    tomorrow from a different machine. They also enable time-travel — forking
    from any past checkpoint to try a different branch.
    """
    from sentinel.db import make_checkpointer

    checkpointer, cm = make_checkpointer()
    try:
        yield compile_graph(checkpointer, with_interrupt=with_interrupt)
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)


def render_mermaid() -> str:
    """Mermaid source for the docs and the UI."""
    return build_graph().compile(checkpointer=MemorySaver()).get_graph().draw_mermaid()
