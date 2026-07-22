"""Signal shapes for synthetic metrics and logs.

All randomness is seeded by the caller so a scenario replays identically.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from datetime import datetime, timedelta

from simulator.models import LogEntry, LogLevel, MetricPoint, MetricSeries


def series(
    service: str,
    metric: str,
    unit: str,
    start: datetime,
    minutes: int,
    shape: Callable[[float], float],
    noise: float,
    rng: random.Random,
    step_seconds: int = 30,
) -> MetricSeries:
    """Build a series by sampling `shape(t)` where t goes 0.0 -> 1.0."""
    n = max(2, (minutes * 60) // step_seconds)
    points = []
    for i in range(n):
        t = i / (n - 1)
        value = shape(t) * (1.0 + rng.uniform(-noise, noise))
        points.append(
            MetricPoint(
                timestamp=start + timedelta(seconds=i * step_seconds),
                value=max(0.0, round(value, 3)),
            )
        )
    return MetricSeries(service=service, metric=metric, unit=unit, points=points)


# --- shapes: each maps t in [0,1] to a value ------------------------------


def flat(value: float) -> Callable[[float], float]:
    return lambda _t: value


def step(before: float, after: float, at: float = 0.5) -> Callable[[float], float]:
    """Instant level change — the signature of a bad deploy or config flip."""
    return lambda t: before if t < at else after


def ramp(start_v: float, end_v: float, begin: float = 0.0) -> Callable[[float], float]:
    """Linear climb — the signature of a leak or slow exhaustion."""

    def _f(t: float) -> float:
        if t < begin:
            return start_v
        p = (t - begin) / max(1e-6, 1.0 - begin)
        return start_v + (end_v - start_v) * p

    return _f


def sawtooth(low: float, high: float, teeth: int) -> Callable[[float], float]:
    """Climb-and-reset — a leaking process being OOM-killed and restarted."""

    def _f(t: float) -> float:
        p = (t * teeth) % 1.0
        return low + (high - low) * p

    return _f


def spike(base: float, peak: float, at: float = 0.6, width: float = 0.12):
    """A transient excursion that recovers."""

    def _f(t: float) -> float:
        d = abs(t - at)
        if d > width:
            return base
        return base + (peak - base) * math.cos(d / width * math.pi / 2) ** 2

    return _f


# --- log helpers ----------------------------------------------------------

_TRACE_CHARS = "0123456789abcdef"


def trace_id(rng: random.Random) -> str:
    return "".join(rng.choice(_TRACE_CHARS) for _ in range(16))


def emit(
    service: str,
    start: datetime,
    minutes: int,
    count: int,
    level: LogLevel,
    templates: list[str],
    rng: random.Random,
    begin: float = 0.0,
    end: float = 1.0,
    attributes: dict | None = None,
    with_trace: bool = True,
) -> list[LogEntry]:
    """Scatter `count` log lines across the [begin, end] fraction of the window."""
    out: list[LogEntry] = []
    span = minutes * 60
    for _ in range(count):
        t = rng.uniform(begin, end)
        ts = start + timedelta(seconds=t * span)
        tmpl = rng.choice(templates)
        msg = tmpl.format(
            ms=rng.randint(800, 9000),
            n=rng.randint(2, 400),
            pct=rng.randint(80, 99),
            id=rng.randint(10000, 99999),
        )
        out.append(
            LogEntry(
                timestamp=ts,
                service=service,
                level=level,
                message=msg,
                trace_id=trace_id(rng) if with_trace else None,
                attributes=dict(attributes or {}),
            )
        )
    return sorted(out, key=lambda e: e.timestamp)
