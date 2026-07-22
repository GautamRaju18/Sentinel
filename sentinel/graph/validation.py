"""Plan validation — the last check before anything touches production.

A model that writes a syntactically valid plan can still write a nonsensical
one. The first end-to-end run produced two examples within a single plan:

  * `apply_config` with an empty key and value, which executed as a silent
    no-op and reported success
  * `scale_service` reducing replicas from 4 to 1 *during an outage*, which
    would have made the incident worse

Neither is a schema violation, so Pydantic passed both. These are semantic
errors, and they need a semantic check. The approval gate is not sufficient
either — an operator skimming a plan should not have to catch that a config
step has no key in it.

Rule of thumb applied here: reject what is incoherent, warn about what is
merely questionable, and never silently repair. A repaired plan is not the plan
the operator approved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sentinel.graph.schemas import RemediationPlan, RemediationStep
from sentinel.logging_setup import get_logger
from simulator.world import SimulatedWorld

log = get_logger(__name__)


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = []
        for e in self.errors:
            lines.append(f"BLOCKED: {e}")
        for w in self.warnings:
            lines.append(f"warning: {w}")
        return "\n".join(lines) or "plan validated"


def validate_step(step: RemediationStep, world: SimulatedWorld) -> ValidationResult:
    r = ValidationResult()
    known_services = set(world.known_services())
    known_deploys = {d.deploy_id for d in world.deploys}

    if not step.target or not step.target.strip():
        r.errors.append(f"{step.action} has an empty target")
        return r

    if step.action == "rollback_deploy":
        if step.target not in known_deploys:
            r.errors.append(
                f"rollback_deploy targets '{step.target}', which is not a known deploy. "
                f"Known: {', '.join(sorted(known_deploys)) or 'none'}"
            )
        else:
            deploy = next(d for d in world.deploys if d.deploy_id == step.target)
            if deploy.rolled_back:
                r.errors.append(f"{step.target} has already been rolled back")

    elif step.action == "apply_config":
        key = step.parameters.get("key", "").strip()
        value = step.parameters.get("value", "").strip()
        if not key:
            r.errors.append(
                "apply_config has no 'key' in parameters — this would apply an empty "
                "configuration change and report success"
            )
        if not value:
            r.errors.append(f"apply_config for key '{key or '?'}' has no 'value'")
        if step.target not in known_services:
            r.errors.append(f"apply_config targets unknown service '{step.target}'")

    elif step.action == "scale_service":
        if step.target not in known_services:
            r.errors.append(f"scale_service targets unknown service '{step.target}'")
            return r
        raw = step.parameters.get("replicas", "")
        try:
            replicas = int(raw)
        except (TypeError, ValueError):
            r.errors.append(f"scale_service needs an integer 'replicas' parameter, got {raw!r}")
            return r
        current = world.services[step.target].replicas_desired
        if not 1 <= replicas <= 50:
            r.errors.append(f"replicas={replicas} is outside the allowed range 1-50")
        elif replicas < current:
            # Scaling down during an active incident removes capacity from a
            # system that is already failing.
            r.errors.append(
                f"scale_service would REDUCE {step.target} from {current} to {replicas} "
                f"replicas during an active incident. If shedding capacity is genuinely "
                f"intended, say so explicitly in the rationale and re-plan."
            )
        elif replicas > current * 3:
            r.warnings.append(
                f"scaling {step.target} {current} -> {replicas} is a large jump; if the "
                f"bottleneck is a shared database this will make it worse"
            )

    elif step.action == "restart_service":
        if step.target not in known_services:
            r.errors.append(f"restart_service targets unknown service '{step.target}'")

    return r


def validate_plan(plan: RemediationPlan, world: SimulatedWorld) -> ValidationResult:
    result = ValidationResult()

    for i, step in enumerate(plan.steps, 1):
        step_result = validate_step(step, world)
        result.errors += [f"step {i}: {e}" for e in step_result.errors]
        result.warnings += [f"step {i}: {w}" for w in step_result.warnings]

    # Cross-step coherence.
    actions = [(s.action, s.target) for s in plan.steps]
    if len(actions) != len(set(actions)):
        result.warnings.append("plan repeats the same action on the same target")

    rollbacks = [s for s in plan.steps if s.action == "rollback_deploy"]
    if len(rollbacks) > 1:
        result.errors.append(
            "plan rolls back more than one deploy at once — do them one at a time so "
            "the effect of each is observable"
        )

    if len(plan.steps) > 4:
        result.warnings.append(
            f"plan has {len(plan.steps)} steps; large plans are hard to attribute when "
            f"only some of them help"
        )

    if not plan.rollback_plan.strip() or plan.rollback_plan.strip().lower() in {"n/a", "none"}:
        result.warnings.append("plan has no rollback strategy")

    if result.errors:
        log.warning("plan.invalid", errors=len(result.errors))
    return result
