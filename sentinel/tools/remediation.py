"""Write tools — these change the world.

Every one of them routes through guardrails.authorize(). The `approved` flag
is never set by the model; it is injected by the graph after a human clicks
approve. That separation is the whole point: the agent can propose a
rollback, but it cannot grant itself permission to run one.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from sentinel.tools.guardrails import authorize, blast_radius
from simulator.world import get_world

# Set by the graph's remediation node once a human approves. Module-level
# rather than a tool argument precisely so the model cannot forge it.
_APPROVAL_GRANTED: set[str] = set()


def grant_approval(action: str) -> None:
    _APPROVAL_GRANTED.add(action)


def revoke_all_approvals() -> None:
    _APPROVAL_GRANTED.clear()


def _gate(action: str) -> str | None:
    decision, reason = authorize(action, approved=action in _APPROVAL_GRANTED)
    if decision == "allow":
        return None
    return (
        f"BLOCKED: {reason}.\n"
        f"This action has blast radius '{blast_radius(action)}'. Do not retry it. "
        f"Instead, include it in your remediation plan and let the approval step "
        f"present it to a human."
    )


class RollbackInput(BaseModel):
    deploy_id: str = Field(..., description="Deploy to roll back, e.g. 'dpl-8814'.")
    reason: str = Field(..., description="Why this deploy is believed to be the cause.")


@tool("rollback_deploy", args_schema=RollbackInput)
async def rollback_deploy(deploy_id: str, reason: str) -> str:
    """Roll a service back to its previous version. DESTRUCTIVE — blast radius critical.

    Only propose this when a specific deploy is strongly implicated: it landed
    shortly before onset, and the metric change matches what that code change
    would plausibly cause. Rolling back the wrong deploy costs time and can
    itself cause an outage.
    """
    if blocked := _gate("rollback_deploy"):
        return blocked
    result = get_world().rollback_deploy(deploy_id)
    return f"{'OK' if result.ok else 'FAILED'}: {result.message}"


class RestartInput(BaseModel):
    service: str = Field(..., description="Service to restart.")
    reason: str = Field(..., description="Why a restart is expected to help.")


@tool("restart_service", args_schema=RestartInput)
async def restart_service(service: str, reason: str) -> str:
    """Restart all pods for a service. DESTRUCTIVE — blast radius high.

    Restarts clear in-memory state, so they help with leaks and stale
    configuration held in memory. They do nothing for a bad code change, and
    they drop in-flight requests. Treat a restart that "fixes" something as a
    clue about the cause, not as a resolution.
    """
    if blocked := _gate("restart_service"):
        return blocked
    result = get_world().restart_service(service)
    return f"{'OK' if result.ok else 'FAILED'}: {result.message}"


class ScaleInput(BaseModel):
    service: str = Field(..., description="Service to scale.")
    replicas: int = Field(..., description="Desired replica count.", ge=1, le=50)
    reason: str = Field(..., description="Why more or fewer replicas will help.")


@tool("scale_service", args_schema=ScaleInput)
async def scale_service(service: str, replicas: int, reason: str) -> str:
    """Change a service's replica count. Blast radius medium.

    Scaling buys headroom against load. It does not fix a defect, and scaling
    a service that is saturating a shared database will make that worse.
    """
    if blocked := _gate("scale_service"):
        return blocked
    result = get_world().scale_service(service, replicas)
    return f"{'OK' if result.ok else 'FAILED'}: {result.message}"


class ApplyConfigInput(BaseModel):
    service: str = Field(..., description="Service whose config to change.")
    key: str = Field(..., description="Config key, e.g. 'maximumPoolSize'.")
    value: str = Field(..., description="New value.")
    reason: str = Field(..., description="Why this value is correct.")


@tool("apply_config", args_schema=ApplyConfigInput)
async def apply_config(service: str, key: str, value: str, reason: str) -> str:
    """Set a configuration value and trigger a reload. DESTRUCTIVE — blast radius high.

    Use this to revert a bad configuration change. Prefer restoring a known
    previous value over inventing a new one.
    """
    if blocked := _gate("apply_config"):
        return blocked
    result = get_world().apply_config(service, key, value)
    return f"{'OK' if result.ok else 'FAILED'}: {result.message}"


WRITE_TOOLS = [rollback_deploy, restart_service, scale_service, apply_config]
