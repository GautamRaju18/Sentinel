"""Driving the graph: start, pause at approval, resume with a decision.

This is the seam the CLI, the API and the UI all sit on. Keeping the
start/resume mechanics here means the approval gate behaves identically however
it is triggered — and it means the UI cannot accidentally invent a path that
skips it.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig

from sentinel.graph.builder import postgres_graph
from sentinel.graph.state import new_state
from sentinel.logging_setup import get_logger
from simulator.world import load_world

log = get_logger(__name__)


@dataclass
class IncidentHandle:
    incident_id: str
    thread_id: str
    scenario: str

    @property
    def config(self) -> RunnableConfig:
        return {"configurable": {"thread_id": self.thread_id}}


def start_incident(scenario: str, incident_id: str | None = None) -> tuple[IncidentHandle, dict]:
    """Load the world and build the initial state. Does not run anything yet."""
    world = load_world(scenario)
    iid = incident_id or f"inc-{uuid.uuid4().hex[:8]}"
    handle = IncidentHandle(incident_id=iid, thread_id=iid, scenario=scenario)
    state = new_state(iid, scenario, world.alert.render())
    return handle, state


async def run_until_pause(
    handle: IncidentHandle, state: dict, *, stream: bool = True
) -> AsyncIterator[tuple[str, Any]]:
    """Run until the approval interrupt (or the end) yielding node updates."""
    async with postgres_graph() as graph:
        if stream:
            async for chunk in graph.astream(state, handle.config, stream_mode="updates"):
                for node, update in chunk.items():
                    yield node, update
        else:
            result = await graph.ainvoke(state, handle.config)
            yield "final", result

        snapshot = await graph.aget_state(handle.config)
        yield (
            "__paused__",
            {
                "next": snapshot.next,
                "values": snapshot.values,
                "awaiting_approval": bool(snapshot.next and "approval" in snapshot.next),
            },
        )


async def resume_with_decision(
    handle: IncidentHandle, *, approved: bool, note: str = ""
) -> AsyncIterator[tuple[str, Any]]:
    """Write the operator's decision into state and continue the graph.

    The decision is written through `aupdate_state`, i.e. from outside the
    agent. No node and no model can produce this value — that is what makes the
    gate meaningful rather than decorative.
    """
    async with postgres_graph() as graph:
        await graph.aupdate_state(handle.config, {"approved": approved, "approval_note": note})
        log.info("runner.resume", incident=handle.incident_id, approved=approved)
        async for chunk in graph.astream(None, handle.config, stream_mode="updates"):
            for node, update in chunk.items():
                yield node, update

        snapshot = await graph.aget_state(handle.config)
        yield "__done__", {"values": snapshot.values, "next": snapshot.next}


async def get_snapshot(handle: IncidentHandle):
    async with postgres_graph() as graph:
        return await graph.aget_state(handle.config)


async def get_history(handle: IncidentHandle, limit: int = 50) -> list[Any]:
    """Checkpoint history — the basis for time-travel replay."""
    async with postgres_graph() as graph:
        out = []
        async for snap in graph.aget_state_history(handle.config, limit=limit):
            out.append(snap)
        return out


async def fork_from(handle: IncidentHandle, checkpoint_id: str, updates: dict) -> RunnableConfig:
    """Rewind to a checkpoint, change something, and branch a new run from it.

    Useful for asking "what if the critic had rejected this?" without re-running
    the whole incident, and for debugging a bad decision at the exact node that
    made it.
    """
    async with postgres_graph() as graph:
        config = {"configurable": {"thread_id": handle.thread_id, "checkpoint_id": checkpoint_id}}
        return await graph.aupdate_state(config, updates)
