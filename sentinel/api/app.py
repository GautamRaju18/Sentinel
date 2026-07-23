"""HTTP API.

Two things shape this surface:

  * Investigations take minutes, so the interesting endpoint streams. Node
    updates go out over SSE as they happen rather than making a client wait for
    a single large response.
  * The approval gate is an HTTP boundary, not a UI convention. `/approve` is
    the only way past it, and it is a separate authenticated call from the one
    that started the incident. A client cannot start-and-approve in one request.

    uv run uvicorn sentinel.api.app:app --reload
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from sentinel.config import get_settings
from sentinel.logging_setup import configure_observability as configure_logging
from sentinel.logging_setup import get_logger
from sentinel.models.router import describe_routing
from sentinel.runner import (
    IncidentHandle,
    get_history,
    get_snapshot,
    resume_with_decision,
    run_until_pause,
    start_incident,
)
from simulator.scenarios import list_scenarios
from simulator.world import load_world

log = get_logger(__name__)

# In-process registry. The durable record lives in Postgres checkpoints; this
# only maps an incident id back to its thread so a restart can still resume by
# thread id.
_incidents: dict[str, IncidentHandle] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging("INFO")
    log.info("api.starting", routing=describe_routing().replace("\n", " | "))
    yield
    log.info("api.stopping")


app = FastAPI(
    title="Sentinel",
    description="Autonomous incident response agent",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # the Streamlit UI
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartRequest(BaseModel):
    scenario: str = Field(..., description="Scenario slug")
    incident_id: str | None = None


class ApprovalRequest(BaseModel):
    approved: bool
    note: str = ""


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "content"):  # LangChain messages
        return {"type": type(value).__name__, "content": str(value.content)}
    return value


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.get("/config")
async def config() -> dict:
    s = get_settings()
    return {
        "routing": describe_routing(),
        "triage_backend": s.triage_backend,
        "max_investigation_loops": s.max_investigation_loops,
        "always_require_approval": s.always_require_approval,
        "max_auto_blast_radius": s.max_auto_blast_radius,
    }


@app.get("/scenarios")
async def scenarios() -> list[dict]:
    out = []
    for slug in list_scenarios():
        w = load_world(slug)
        out.append(
            {
                "slug": slug,
                "title": w.scenario.title,
                "description": w.scenario.description,
                "alert_source": w.alert.source,
                "alert": w.alert.render(),
            }
        )
    return out


@app.post("/incidents")
async def create_incident(req: StartRequest) -> dict:
    if req.scenario not in list_scenarios():
        raise HTTPException(404, f"unknown scenario '{req.scenario}'")
    handle, state = start_incident(req.scenario, req.incident_id)
    _incidents[handle.incident_id] = handle
    return {
        "incident_id": handle.incident_id,
        "thread_id": handle.thread_id,
        "scenario": handle.scenario,
        "alert": state["alert"],
        "stream_url": f"/incidents/{handle.incident_id}/stream",
    }


def _handle(incident_id: str) -> IncidentHandle:
    handle = _incidents.get(incident_id)
    if handle is None:
        raise HTTPException(404, f"unknown incident '{incident_id}'")
    return handle


@app.get("/incidents/{incident_id}/stream")
async def stream(incident_id: str) -> EventSourceResponse:
    """Run the graph up to the approval gate, streaming each node update."""
    handle = _handle(incident_id)
    _, state = start_incident(handle.scenario, handle.incident_id)

    async def generator():
        try:
            async for node, update in run_until_pause(handle, state):
                yield {
                    "event": "paused" if node.startswith("__") else "node",
                    "data": json.dumps({"node": node, "update": _jsonable(update)}),
                }
        except asyncio.CancelledError:
            log.info("api.stream_cancelled", incident=incident_id)
            raise
        except Exception as e:
            log.error("api.stream_failed", incident=incident_id, error=str(e))
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(generator())


@app.get("/incidents/{incident_id}")
async def get_incident(incident_id: str) -> dict:
    handle = _handle(incident_id)
    snapshot = await get_snapshot(handle)
    return {
        "incident_id": incident_id,
        "next": list(snapshot.next or []),
        "awaiting_approval": bool(snapshot.next and "approval" in snapshot.next),
        "values": _jsonable(snapshot.values),
    }


@app.post("/incidents/{incident_id}/approve")
async def approve(incident_id: str, req: ApprovalRequest) -> EventSourceResponse:
    """The gate. Nothing reaches production except through this endpoint."""
    handle = _handle(incident_id)
    snapshot = await get_snapshot(handle)
    if not (snapshot.next and "approval" in snapshot.next):
        raise HTTPException(409, "incident is not awaiting approval")

    async def generator():
        try:
            async for node, update in resume_with_decision(
                handle, approved=req.approved, note=req.note
            ):
                yield {
                    "event": "done" if node.startswith("__") else "node",
                    "data": json.dumps({"node": node, "update": _jsonable(update)}),
                }
        except Exception as e:
            log.error("api.approve_failed", incident=incident_id, error=str(e))
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(generator())


@app.get("/incidents/{incident_id}/history")
async def history(incident_id: str, limit: int = 50) -> list[dict]:
    """Checkpoint history — every state the graph passed through."""
    handle = _handle(incident_id)
    snapshots = await get_history(handle, limit)
    return [
        {
            "checkpoint_id": s.config.get("configurable", {}).get("checkpoint_id"),
            "next": list(s.next or []),
            "stage": s.values.get("stage"),
            "loop_count": s.values.get("loop_count"),
            "created_at": s.created_at,
        }
        for s in snapshots
    ]
