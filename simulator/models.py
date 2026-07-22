"""Data shapes for the simulated environment."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Severity(StrEnum):
    P1 = "P1"  # total outage, revenue-impacting
    P2 = "P2"  # major degradation
    P3 = "P3"  # minor degradation, contained
    P4 = "P4"  # cosmetic / informational


class LogEntry(BaseModel):
    timestamp: datetime
    service: str
    level: LogLevel
    message: str
    trace_id: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        trace = f" trace={self.trace_id}" if self.trace_id else ""
        attrs = "".join(f" {k}={v}" for k, v in self.attributes.items())
        return f"{ts} [{self.level:<5}] {self.service}{trace} {self.message}{attrs}"


class MetricPoint(BaseModel):
    timestamp: datetime
    value: float


class MetricSeries(BaseModel):
    service: str
    metric: str
    unit: str
    points: list[MetricPoint]

    def summarize(self) -> str:
        """Compact textual summary — feeding 500 raw points to an LLM is waste."""
        if not self.points:
            return f"{self.service}.{self.metric}: no data"
        vals = [p.value for p in self.points]
        first_q = vals[: max(1, len(vals) // 4)]
        last_q = vals[-max(1, len(vals) // 4) :]
        baseline = sum(first_q) / len(first_q)
        current = sum(last_q) / len(last_q)
        delta = ((current - baseline) / baseline * 100) if baseline else 0.0
        return (
            f"{self.service}.{self.metric} ({self.unit}): "
            f"baseline={baseline:.2f} current={current:.2f} "
            f"min={min(vals):.2f} max={max(vals):.2f} change={delta:+.1f}%"
        )


class Deploy(BaseModel):
    deploy_id: str
    service: str
    version: str
    timestamp: datetime
    author: str
    commit_sha: str
    summary: str
    rolled_back: bool = False

    def render(self) -> str:
        status = " [ROLLED BACK]" if self.rolled_back else ""
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"{self.deploy_id} {ts} {self.service} {self.version} "
            f"by {self.author} ({self.commit_sha[:8]}): {self.summary}{status}"
        )


class ServiceHealth(BaseModel):
    service: str
    status: str  # healthy | degraded | down
    replicas_desired: int
    replicas_ready: int
    restarts_last_hour: int
    version: str

    def render(self) -> str:
        return (
            f"{self.service}: {self.status} "
            f"replicas={self.replicas_ready}/{self.replicas_desired} "
            f"restarts_1h={self.restarts_last_hour} version={self.version}"
        )


class Alert(BaseModel):
    """What arrives at the front door of the system."""

    alert_id: str
    timestamp: datetime
    source: str  # prometheus | sentry | pagerduty | healthcheck
    title: str
    body: str
    service: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)

    def render(self) -> str:
        labels = " ".join(f"{k}={v}" for k, v in self.labels.items())
        return (
            f"[{self.source}] {self.title}\n"
            f"time: {self.timestamp.isoformat()}\n"
            f"labels: {labels}\n\n"
            f"{self.body}"
        )
