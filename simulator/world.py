"""The queryable surface of the simulated environment.

Tools in sentinel/tools/ talk to this and nothing else. The world is
stateful: remediation actions actually mutate it, so the agent's verify
step can observe whether the fix worked.

Ground truth is deliberately NOT reachable through any query method —
only through `scenario.ground_truth`, which the eval harness reads directly.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from simulator.models import (
    Alert,
    Deploy,
    LogEntry,
    LogLevel,
    MetricSeries,
    ServiceHealth,
)
from simulator.scenarios import ANCHOR, WINDOW_MINUTES, Scenario, load_scenario


class ActionResult:
    def __init__(self, ok: bool, message: str, changed: dict | None = None):
        self.ok = ok
        self.message = message
        self.changed = changed or {}

    def __repr__(self) -> str:
        return f"ActionResult(ok={self.ok}, message={self.message!r})"


class SimulatedWorld:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        # Mutable copies — actions change these, queries read them.
        self.services: dict[str, ServiceHealth] = {
            k: v.model_copy(deep=True) for k, v in scenario.services.items()
        }
        self.deploys: list[Deploy] = [d.model_copy(deep=True) for d in scenario.deploys]
        self.logs: list[LogEntry] = list(scenario.logs)
        self.metrics: dict[str, MetricSeries] = dict(scenario.metrics)
        self.action_log: list[str] = []
        self._remediated = False

    # --- identity ---------------------------------------------------------

    @property
    def alert(self) -> Alert:
        return self.scenario.alert

    @property
    def now(self) -> datetime:
        return ANCHOR + timedelta(minutes=WINDOW_MINUTES)

    def known_services(self) -> list[str]:
        return sorted(self.services)

    # --- queries ----------------------------------------------------------

    def query_logs(
        self,
        service: str | None = None,
        level: str | None = None,
        contains: str | None = None,
        since_minutes: int = 90,
        limit: int = 50,
    ) -> list[LogEntry]:
        cutoff = self.now - timedelta(minutes=since_minutes)
        out = [e for e in self.logs if e.timestamp >= cutoff]
        if service:
            out = [e for e in out if e.service == service]
        if level:
            wanted = level.upper()
            order = list(LogLevel)
            try:
                min_idx = order.index(LogLevel(wanted))
                out = [e for e in out if order.index(e.level) >= min_idx]
            except ValueError:
                out = [e for e in out if e.level == wanted]
        if contains:
            pat = re.compile(re.escape(contains), re.IGNORECASE)
            out = [e for e in out if pat.search(e.message)]
        out.sort(key=lambda e: e.timestamp, reverse=True)
        return out[:limit]

    def get_metric(self, service: str, metric: str) -> MetricSeries | None:
        return self.metrics.get(f"{service}.{metric}")

    def list_metrics(self, service: str | None = None) -> list[str]:
        keys = sorted(self.metrics)
        if service:
            keys = [k for k in keys if k.startswith(f"{service}.")]
        return keys

    def get_deploys(self, service: str | None = None, since_hours: int = 48) -> list[Deploy]:
        cutoff = self.now - timedelta(hours=since_hours)
        out = [d for d in self.deploys if d.timestamp >= cutoff]
        if service:
            out = [d for d in out if d.service == service]
        return sorted(out, key=lambda d: d.timestamp, reverse=True)

    def get_health(self, service: str | None = None) -> list[ServiceHealth]:
        if service:
            h = self.services.get(service)
            return [h] if h else []
        return [self.services[k] for k in sorted(self.services)]

    # --- actions (these mutate the world) ---------------------------------

    def rollback_deploy(self, deploy_id: str) -> ActionResult:
        target = next((d for d in self.deploys if d.deploy_id == deploy_id), None)
        if target is None:
            return ActionResult(False, f"no deploy {deploy_id}")
        if target.rolled_back:
            return ActionResult(False, f"{deploy_id} is already rolled back")

        target.rolled_back = True
        self.action_log.append(f"rollback {deploy_id} ({target.service})")

        gt = self.scenario.ground_truth
        correct = deploy_id in gt.correct_remediation
        if correct:
            self._heal(target.service)
            return ActionResult(
                True,
                f"Rolled back {deploy_id}. {target.service} returning to previous version. "
                f"Metrics recovering.",
                {"service": target.service, "healed": True},
            )
        return ActionResult(
            True,
            f"Rolled back {deploy_id}, but {target.service} metrics are unchanged. "
            f"This deploy does not appear to be the cause.",
            {"service": target.service, "healed": False},
        )

    def restart_service(self, service: str) -> ActionResult:
        h = self.services.get(service)
        if h is None:
            return ActionResult(False, f"unknown service {service}")
        self.action_log.append(f"restart {service}")
        h.restarts_last_hour += h.replicas_desired

        gt = self.scenario.ground_truth
        # A restart genuinely fixes an expired cert (new cert gets loaded) and
        # temporarily relieves a leak; it does nothing for a bad deploy.
        if gt.category == "expired_credential" and service == gt.affected_service:
            return ActionResult(
                True,
                f"Restarted {service}. Pods came back but immediately hit the same "
                f"x509 error — the certificate on disk is still expired.",
                {"healed": False},
            )
        if gt.category == "resource_exhaustion" and service == gt.affected_service:
            return ActionResult(
                True,
                f"Restarted {service}. Memory reset to baseline, but with no cache bound "
                f"in place it will climb again. Buys roughly 20 minutes.",
                {"healed": False, "temporary": True},
            )
        return ActionResult(
            True, f"Restarted {service}. No change in error rate.", {"healed": False}
        )

    def scale_service(self, service: str, replicas: int) -> ActionResult:
        h = self.services.get(service)
        if h is None:
            return ActionResult(False, f"unknown service {service}")
        if not 1 <= replicas <= 50:
            return ActionResult(False, "replicas must be between 1 and 50")
        old = h.replicas_desired
        h.replicas_desired = replicas
        h.replicas_ready = replicas
        self.action_log.append(f"scale {service} {old}->{replicas}")
        return ActionResult(
            True,
            f"Scaled {service} from {old} to {replicas} replicas. "
            f"This adds capacity but does not address a root cause on its own.",
            {"healed": False},
        )

    def apply_config(self, service: str, key: str, value: str) -> ActionResult:
        h = self.services.get(service)
        if h is None:
            return ActionResult(False, f"unknown service {service}")
        self.action_log.append(f"config {service} {key}={value}")
        gt = self.scenario.ground_truth
        if gt.category == "config_change" and service == gt.affected_service:
            self._heal(service)
            return ActionResult(
                True,
                f"Applied {key}={value} to {service} and reloaded. "
                f"Pool wait times dropping, error rate recovering.",
                {"healed": True},
            )
        return ActionResult(
            True, f"Applied {key}={value} to {service}. No measurable change.", {"healed": False}
        )

    def _heal(self, service: str) -> None:
        self._remediated = True
        h = self.services.get(service)
        if h:
            h.status = "healthy"
        for name, health in self.services.items():
            if health.status == "degraded" and name != service:
                health.status = "healthy"

    @property
    def remediated(self) -> bool:
        return self._remediated


_world: SimulatedWorld | None = None


def get_world() -> SimulatedWorld:
    """Process-wide world. Tools call this; the CLI sets it via load_world()."""
    global _world
    if _world is None:
        _world = SimulatedWorld(load_scenario("bad_deploy"))
    return _world


def load_world(slug: str) -> SimulatedWorld:
    global _world
    _world = SimulatedWorld(load_scenario(slug))
    return _world


def reset_world() -> None:
    global _world
    _world = None
