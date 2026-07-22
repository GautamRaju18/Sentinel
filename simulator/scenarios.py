"""Failure scenarios.

Each scenario is a self-contained slice of a production incident: the alert
that fires, the logs and metrics an investigator can pull, the deploy history,
and — crucially — the ground truth. The ground truth never reaches the agent;
it exists so evals/ can score whether the agent actually found the cause or
just wrote something plausible.

Adding a scenario: write a builder, register it in SCENARIOS.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from simulator.generators import (
    emit,
    flat,
    ramp,
    sawtooth,
    series,
    spike,
    step,
)
from simulator.models import (
    Alert,
    Deploy,
    LogEntry,
    LogLevel,
    MetricSeries,
    ServiceHealth,
    Severity,
)

# Fixed anchor keeps every replay byte-identical.
ANCHOR = datetime(2026, 3, 14, 9, 0, 0)
WINDOW_MINUTES = 90
# The moment things went wrong, as a fraction of the window.
ONSET = 0.5


@dataclass(frozen=True)
class GroundTruth:
    """Answer key. Never exposed through any tool the agent can call."""

    root_cause: str
    category: str
    severity: Severity
    affected_service: str
    correct_remediation: str
    key_evidence: list[str]
    red_herrings: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    slug: str
    title: str
    description: str
    alert: Alert
    services: dict[str, ServiceHealth]
    deploys: list[Deploy]
    logs: list[LogEntry]
    metrics: dict[str, MetricSeries]
    ground_truth: GroundTruth
    runbook_hint: str = ""


def _onset_at() -> datetime:
    return ANCHOR + timedelta(minutes=WINDOW_MINUTES * ONSET)


# ---------------------------------------------------------------------------
# 1. Bad deploy: an ORM change removes eager loading -> N+1 queries
# ---------------------------------------------------------------------------


def build_bad_deploy() -> Scenario:
    rng = random.Random(1001)
    onset = _onset_at()

    deploys = [
        Deploy(
            deploy_id="dpl-8814",
            service="checkout-api",
            version="v2.31.0",
            timestamp=onset - timedelta(minutes=4),
            author="priya.n",
            commit_sha="a3f91c27de4b5099",
            summary="refactor: simplify order serializer",
        ),
        Deploy(
            deploy_id="dpl-8813",
            service="search-api",
            version="v1.9.2",
            timestamp=onset - timedelta(minutes=52),
            author="marco.l",
            commit_sha="77bd0e13aa920f4c",
            summary="chore: bump elasticsearch client",
        ),
        Deploy(
            deploy_id="dpl-8812",
            service="checkout-api",
            version="v2.30.4",
            timestamp=ANCHOR - timedelta(hours=19),
            author="priya.n",
            commit_sha="15c9e8a0bb37d621",
            summary="fix: correct tax rounding for EU orders",
        ),
    ]

    metrics = {
        "checkout-api.latency_p99_ms": series(
            "checkout-api",
            "latency_p99_ms",
            "ms",
            ANCHOR,
            WINDOW_MINUTES,
            step(240, 4100, ONSET),
            0.08,
            rng,
        ),
        "checkout-api.error_rate": series(
            "checkout-api",
            "error_rate",
            "%",
            ANCHOR,
            WINDOW_MINUTES,
            step(0.2, 6.4, ONSET + 0.03),
            0.15,
            rng,
        ),
        "checkout-api.requests_per_sec": series(
            "checkout-api",
            "requests_per_sec",
            "rps",
            ANCHOR,
            WINDOW_MINUTES,
            flat(310),
            0.06,
            rng,
        ),
        "postgres-main.queries_per_sec": series(
            "postgres-main",
            "queries_per_sec",
            "qps",
            ANCHOR,
            WINDOW_MINUTES,
            step(1250, 38400, ONSET),
            0.05,
            rng,
        ),
        "postgres-main.cpu_percent": series(
            "postgres-main",
            "cpu_percent",
            "%",
            ANCHOR,
            WINDOW_MINUTES,
            step(22, 94, ONSET),
            0.04,
            rng,
        ),
        "checkout-api.memory_mb": series(
            "checkout-api",
            "memory_mb",
            "MB",
            ANCHOR,
            WINDOW_MINUTES,
            flat(512),
            0.05,
            rng,
        ),
    }

    logs = []
    logs += emit(
        "checkout-api",
        ANCHOR,
        WINDOW_MINUTES,
        14,
        LogLevel.INFO,
        ["GET /api/v1/orders completed in {ms}ms", "POST /api/v1/checkout ok in {ms}ms"],
        rng,
        end=ONSET,
    )
    logs += emit(
        "checkout-api",
        ANCHOR,
        WINDOW_MINUTES,
        30,
        LogLevel.WARN,
        [
            "slow query detected: SELECT * FROM order_items WHERE order_id = $1 ({ms}ms)",
            "request exceeded soft latency budget: {ms}ms",
            "connection pool wait {ms}ms, active={n}",
        ],
        rng,
        begin=ONSET,
    )
    logs += emit(
        "checkout-api",
        ANCHOR,
        WINDOW_MINUTES,
        18,
        LogLevel.ERROR,
        [
            "upstream timeout after 5000ms querying postgres-main",
            "TimeoutError: acquiring connection from pool (waited {ms}ms)",
            "500 Internal Server Error on POST /api/v1/checkout",
        ],
        rng,
        begin=ONSET + 0.03,
    )
    logs += emit(
        "postgres-main",
        ANCHOR,
        WINDOW_MINUTES,
        12,
        LogLevel.WARN,
        [
            "duration: {ms}.221 ms  statement: SELECT * FROM order_items WHERE order_id = $1",
            "too many concurrent queries, {n} waiting",
        ],
        rng,
        begin=ONSET,
        with_trace=False,
    )
    logs += emit(
        "checkout-api",
        ANCHOR,
        WINDOW_MINUTES,
        2,
        LogLevel.INFO,
        ["deployment dpl-8814 rolled out: v2.30.4 -> v2.31.0, 6/6 pods ready"],
        rng,
        begin=ONSET - 0.05,
        end=ONSET,
        with_trace=False,
    )

    return Scenario(
        slug="bad_deploy",
        title="Checkout latency spike after deploy",
        description="An ORM refactor dropped eager loading, turning one query into N+1.",
        alert=Alert(
            alert_id="alt-40219",
            timestamp=onset + timedelta(minutes=3),
            source="prometheus",
            title="HighLatencyP99 — checkout-api",
            body=(
                "ALERT HighLatencyP99 firing for checkout-api\n"
                "expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket"
                '{service="checkout-api"}[5m])) > 1.0\n'
                "current value: 4.12s (threshold 1.0s)\n"
                "duration: 3m\n"
                "Customer checkout requests are timing out. Payment conversion dropping."
            ),
            service="checkout-api",
            labels={"severity": "critical", "team": "payments", "env": "production"},
        ),
        services={
            "checkout-api": ServiceHealth(
                service="checkout-api",
                status="degraded",
                replicas_desired=6,
                replicas_ready=6,
                restarts_last_hour=0,
                version="v2.31.0",
            ),
            "postgres-main": ServiceHealth(
                service="postgres-main",
                status="degraded",
                replicas_desired=1,
                replicas_ready=1,
                restarts_last_hour=0,
                version="16.2",
            ),
            "search-api": ServiceHealth(
                service="search-api",
                status="healthy",
                replicas_desired=3,
                replicas_ready=3,
                restarts_last_hour=0,
                version="v1.9.2",
            ),
        },
        deploys=deploys,
        logs=sorted(logs, key=lambda e: e.timestamp),
        metrics=metrics,
        ground_truth=GroundTruth(
            root_cause=(
                "Deploy dpl-8814 (checkout-api v2.31.0) refactored the order serializer "
                "and removed eager loading of order_items, producing an N+1 query pattern. "
                "Query volume to postgres-main rose ~30x, saturating the connection pool "
                "and driving p99 latency from 240ms to 4.1s."
            ),
            category="bad_deploy",
            severity=Severity.P1,
            affected_service="checkout-api",
            correct_remediation="Roll back checkout-api to v2.30.4 (deploy dpl-8812).",
            key_evidence=[
                "dpl-8814 deployed ~4 minutes before onset",
                "postgres-main queries_per_sec jumped 1250 -> 38400 (step change)",
                "repeated slow query on SELECT * FROM order_items WHERE order_id = $1",
                "checkout-api request rate flat, so this is not organic traffic growth",
            ],
            red_herrings=["search-api deployed 52 minutes earlier and is healthy"],
        ),
        runbook_hint="latency-spike",
    )


# ---------------------------------------------------------------------------
# 2. Memory leak: unbounded cache -> OOMKill loop
# ---------------------------------------------------------------------------


def build_memory_leak() -> Scenario:
    rng = random.Random(1002)
    onset = _onset_at()

    metrics = {
        "cart-service.memory_mb": series(
            "cart-service",
            "memory_mb",
            "MB",
            ANCHOR,
            WINDOW_MINUTES,
            sawtooth(410, 2040, 3),
            0.03,
            rng,
        ),
        "cart-service.restarts": series(
            "cart-service",
            "restarts",
            "count",
            ANCHOR,
            WINDOW_MINUTES,
            ramp(0, 7, 0.15),
            0.0,
            rng,
        ),
        "cart-service.error_rate": series(
            "cart-service",
            "error_rate",
            "%",
            ANCHOR,
            WINDOW_MINUTES,
            ramp(0.3, 11.0, 0.2),
            0.2,
            rng,
        ),
        "cart-service.gc_pause_ms": series(
            "cart-service",
            "gc_pause_ms",
            "ms",
            ANCHOR,
            WINDOW_MINUTES,
            ramp(12, 780, 0.25),
            0.15,
            rng,
        ),
        "cart-service.requests_per_sec": series(
            "cart-service",
            "requests_per_sec",
            "rps",
            ANCHOR,
            WINDOW_MINUTES,
            flat(140),
            0.07,
            rng,
        ),
    }

    logs = []
    logs += emit(
        "cart-service",
        ANCHOR,
        WINDOW_MINUTES,
        10,
        LogLevel.WARN,
        [
            "heap usage at {pct}% of limit",
            "gc pause {ms}ms exceeded target",
            "session cache size now {n}000 entries, no eviction policy configured",
        ],
        rng,
        begin=0.15,
    )
    logs += emit(
        "cart-service",
        ANCHOR,
        WINDOW_MINUTES,
        8,
        LogLevel.FATAL,
        [
            "OOMKilled: container exceeded memory limit 2048Mi",
            "java.lang.OutOfMemoryError: Java heap space",
        ],
        rng,
        begin=0.3,
        with_trace=False,
        attributes={"exit_code": 137},
    )
    logs += emit(
        "cart-service",
        ANCHOR,
        WINDOW_MINUTES,
        9,
        LogLevel.INFO,
        ["container restarted, pod cart-service-7d9f-{id}", "starting cart-service v3.4.1"],
        rng,
        begin=0.3,
        with_trace=False,
    )
    logs += emit(
        "cart-service",
        ANCHOR,
        WINDOW_MINUTES,
        12,
        LogLevel.ERROR,
        [
            "503 Service Unavailable — no healthy upstream",
            "request dropped during shutdown, in-flight={n}",
        ],
        rng,
        begin=0.35,
    )

    return Scenario(
        slug="memory_leak",
        title="cart-service crash looping",
        description="An unbounded in-memory session cache leaks until the pod is OOMKilled.",
        alert=Alert(
            alert_id="alt-40255",
            timestamp=onset,
            source="prometheus",
            title="PodRestartLoop — cart-service",
            body=(
                "ALERT PodRestartLoop firing for cart-service\n"
                'expr: increase(kube_pod_container_status_restarts_total{service="cart-service"}[1h]) > 3\n'
                "current value: 7 restarts in the last hour\n"
                "Pods are being terminated with exit code 137."
            ),
            service="cart-service",
            labels={"severity": "critical", "team": "commerce", "env": "production"},
        ),
        services={
            "cart-service": ServiceHealth(
                service="cart-service",
                status="degraded",
                replicas_desired=4,
                replicas_ready=2,
                restarts_last_hour=7,
                version="v3.4.1",
            ),
            "checkout-api": ServiceHealth(
                service="checkout-api",
                status="healthy",
                replicas_desired=6,
                replicas_ready=6,
                restarts_last_hour=0,
                version="v2.30.4",
            ),
        },
        deploys=[
            Deploy(
                deploy_id="dpl-8790",
                service="cart-service",
                version="v3.4.1",
                timestamp=ANCHOR - timedelta(days=3),
                author="dan.k",
                commit_sha="9c1e77af03b2d845",
                summary="feat: cache session state in memory to cut redis calls",
            ),
        ],
        logs=sorted(logs, key=lambda e: e.timestamp),
        metrics=metrics,
        ground_truth=GroundTruth(
            root_cause=(
                "Deploy dpl-8790 (three days ago) added an in-memory session cache with no "
                "eviction policy or size bound. Heap grows until the 2048Mi container limit "
                "is hit and the pod is OOMKilled (exit 137), then the cycle repeats. The "
                "sawtooth memory profile and rising GC pauses are the signature."
            ),
            category="resource_exhaustion",
            severity=Severity.P1,
            affected_service="cart-service",
            correct_remediation=(
                "Short term: raise the memory limit and increase replicas to absorb load. "
                "Real fix: bound the session cache (LRU + TTL) or revert dpl-8790."
            ),
            key_evidence=[
                "memory_mb shows a sawtooth: climbs to ~2040MB then resets",
                "exit_code 137 / OOMKilled in fatal logs",
                "gc_pause_ms rising steadily from 12ms to ~780ms",
                "request rate flat — load is not the cause",
                "log line naming an unbounded session cache with no eviction policy",
            ],
            red_herrings=["The triggering deploy is 3 days old, so recency alone won't find it"],
        ),
        runbook_hint="pod-restart-loop",
    )


# ---------------------------------------------------------------------------
# 3. Connection pool exhaustion from a config change
# ---------------------------------------------------------------------------


def build_pool_exhaustion() -> Scenario:
    rng = random.Random(1003)
    onset = _onset_at()

    metrics = {
        "payment-service.db_pool_active": series(
            "payment-service",
            "db_pool_active",
            "conns",
            ANCHOR,
            WINDOW_MINUTES,
            step(18, 10, ONSET),
            0.02,
            rng,
        ),
        "payment-service.db_pool_wait_ms": series(
            "payment-service",
            "db_pool_wait_ms",
            "ms",
            ANCHOR,
            WINDOW_MINUTES,
            step(3, 4200, ONSET),
            0.12,
            rng,
        ),
        "payment-service.error_rate": series(
            "payment-service",
            "error_rate",
            "%",
            ANCHOR,
            WINDOW_MINUTES,
            step(0.1, 22.0, ONSET),
            0.1,
            rng,
        ),
        "payment-service.requests_per_sec": series(
            "payment-service",
            "requests_per_sec",
            "rps",
            ANCHOR,
            WINDOW_MINUTES,
            flat(95),
            0.05,
            rng,
        ),
        "postgres-main.connections": series(
            "postgres-main",
            "connections",
            "conns",
            ANCHOR,
            WINDOW_MINUTES,
            step(64, 40, ONSET),
            0.03,
            rng,
        ),
    }

    logs = []
    logs += emit(
        "payment-service",
        ANCHOR,
        WINDOW_MINUTES,
        2,
        LogLevel.INFO,
        ["config reloaded from configmap payment-service-config (revision 47)"],
        rng,
        begin=ONSET - 0.02,
        end=ONSET,
        with_trace=False,
    )
    logs += emit(
        "payment-service",
        ANCHOR,
        WINDOW_MINUTES,
        24,
        LogLevel.ERROR,
        [
            "HikariPool-1 - Connection is not available, request timed out after {ms}ms",
            "could not acquire db connection, pool exhausted (active=10 idle=0 waiting={n})",
            "payment authorization failed: db unavailable",
        ],
        rng,
        begin=ONSET,
    )
    logs += emit(
        "payment-service",
        ANCHOR,
        WINDOW_MINUTES,
        10,
        LogLevel.WARN,
        ["HikariPool-1 - Pool stats (total=10, active=10, idle=0, waiting={n})"],
        rng,
        begin=ONSET,
    )

    return Scenario(
        slug="pool_exhaustion",
        title="payment-service database errors",
        description="A configmap change cut the DB pool size from 40 to 10.",
        alert=Alert(
            alert_id="alt-40301",
            timestamp=onset + timedelta(minutes=2),
            source="sentry",
            title="Error rate spike — payment-service",
            body=(
                "Sentry issue PAYMENT-SERVICE-8821\n"
                "Error: HikariPool-1 - Connection is not available, request timed out\n"
                "events: 1,284 in the last 5 minutes (baseline: 3/5min)\n"
                "users affected: 940\n"
                "Payments are failing at the authorization step."
            ),
            service="payment-service",
            labels={"severity": "critical", "team": "payments", "env": "production"},
        ),
        services={
            "payment-service": ServiceHealth(
                service="payment-service",
                status="degraded",
                replicas_desired=4,
                replicas_ready=4,
                restarts_last_hour=0,
                version="v5.2.0",
            ),
            "postgres-main": ServiceHealth(
                service="postgres-main",
                status="healthy",
                replicas_desired=1,
                replicas_ready=1,
                restarts_last_hour=0,
                version="16.2",
            ),
        },
        deploys=[
            Deploy(
                deploy_id="cfg-2247",
                service="payment-service",
                version="config-rev-47",
                timestamp=onset - timedelta(minutes=2),
                author="ops-bot",
                commit_sha="ee4102bb7c9d31a0",
                summary="chore(config): tune connection pools across services",
            ),
            Deploy(
                deploy_id="dpl-8801",
                service="payment-service",
                version="v5.2.0",
                timestamp=ANCHOR - timedelta(days=6),
                author="sara.v",
                commit_sha="3b8f0c19ed77a254",
                summary="feat: add idempotency keys to authorization",
            ),
        ],
        logs=sorted(logs, key=lambda e: e.timestamp),
        metrics=metrics,
        ground_truth=GroundTruth(
            root_cause=(
                "Config change cfg-2247 (configmap revision 47) reduced the HikariCP "
                "maximumPoolSize for payment-service from 40 to 10. At 95 rps the pool "
                "saturates immediately; requests queue then time out. Note the pool ACTIVE "
                "count fell — the ceiling dropped, load did not rise."
            ),
            category="config_change",
            severity=Severity.P1,
            affected_service="payment-service",
            correct_remediation="Revert configmap payment-service-config to revision 46.",
            key_evidence=[
                "config reload log at onset referencing configmap revision 47",
                "db_pool_active DROPPED 18 -> 10 rather than rising",
                "pool stats logs show total=10, a hard ceiling",
                "request rate flat at ~95 rps",
                "postgres-main itself is healthy — the DB is not the bottleneck",
            ],
            red_herrings=[
                "Errors mention the database, which invites blaming postgres-main",
            ],
        ),
        runbook_hint="db-connection-errors",
    )


# ---------------------------------------------------------------------------
# 4. Expired TLS certificate
# ---------------------------------------------------------------------------


def build_cert_expiry() -> Scenario:
    rng = random.Random(1004)
    onset = _onset_at()

    metrics = {
        "auth-service.error_rate": series(
            "auth-service",
            "error_rate",
            "%",
            ANCHOR,
            WINDOW_MINUTES,
            step(0.1, 100.0, ONSET),
            0.0,
            rng,
        ),
        "auth-service.successful_logins_per_min": series(
            "auth-service",
            "successful_logins_per_min",
            "count",
            ANCHOR,
            WINDOW_MINUTES,
            step(420, 0, ONSET),
            0.02,
            rng,
        ),
        "auth-service.cpu_percent": series(
            "auth-service",
            "cpu_percent",
            "%",
            ANCHOR,
            WINDOW_MINUTES,
            step(34, 8, ONSET),
            0.05,
            rng,
        ),
        "auth-service.tls_handshake_failures": series(
            "auth-service",
            "tls_handshake_failures",
            "count",
            ANCHOR,
            WINDOW_MINUTES,
            step(0, 2400, ONSET),
            0.06,
            rng,
        ),
    }

    logs = []
    logs += emit(
        "auth-service",
        ANCHOR,
        WINDOW_MINUTES,
        26,
        LogLevel.ERROR,
        [
            "x509: certificate has expired or is not yet valid: current time is after 2026-03-14T09:45:00Z",
            "TLS handshake error from 10.4.{n}.12: remote error: tls: bad certificate",
            "failed to verify peer certificate for idp.internal",
        ],
        rng,
        begin=ONSET,
        with_trace=False,
    )
    logs += emit(
        "auth-service",
        ANCHOR,
        WINDOW_MINUTES,
        8,
        LogLevel.WARN,
        ["certificate for idp.internal expires in 0 days", "SAML assertion validation aborted"],
        rng,
        begin=ONSET - 0.08,
    )
    logs += emit(
        "checkout-api",
        ANCHOR,
        WINDOW_MINUTES,
        10,
        LogLevel.ERROR,
        ["401 from auth-service /verify — token validation unavailable"],
        rng,
        begin=ONSET + 0.02,
    )

    return Scenario(
        slug="cert_expiry",
        title="All logins failing",
        description="The internal IdP TLS certificate expired at 09:45.",
        alert=Alert(
            alert_id="alt-40388",
            timestamp=onset + timedelta(minutes=1),
            source="healthcheck",
            title="auth-service /healthz failing — 0 successful logins",
            body=(
                "Synthetic login check failed 5 consecutive times.\n"
                "endpoint: https://auth.internal/healthz\n"
                "last error: x509: certificate has expired or is not yet valid\n"
                "successful_logins_per_min dropped 420 -> 0.\n"
                "No user can authenticate. Total outage."
            ),
            service="auth-service",
            labels={"severity": "critical", "team": "platform", "env": "production"},
        ),
        services={
            "auth-service": ServiceHealth(
                service="auth-service",
                status="down",
                replicas_desired=3,
                replicas_ready=3,
                restarts_last_hour=0,
                version="v4.0.7",
            ),
            "checkout-api": ServiceHealth(
                service="checkout-api",
                status="degraded",
                replicas_desired=6,
                replicas_ready=6,
                restarts_last_hour=0,
                version="v2.30.4",
            ),
        },
        deploys=[
            Deploy(
                deploy_id="dpl-8702",
                service="auth-service",
                version="v4.0.7",
                timestamp=ANCHOR - timedelta(days=21),
                author="lin.z",
                commit_sha="c40b7712fa9e0d38",
                summary="fix: retry SAML metadata fetch on cold start",
            ),
        ],
        logs=sorted(logs, key=lambda e: e.timestamp),
        metrics=metrics,
        ground_truth=GroundTruth(
            root_cause=(
                "The TLS certificate for idp.internal expired at 09:45 UTC. auth-service "
                "cannot complete TLS handshakes with the identity provider, so every login "
                "and token verification fails. Nothing was deployed; this is a time bomb, "
                "not a change-induced failure."
            ),
            category="expired_credential",
            severity=Severity.P1,
            affected_service="auth-service",
            correct_remediation=(
                "Renew and roll out the idp.internal certificate, then restart auth-service "
                "pods to pick it up. Follow up by adding expiry alerting at 30/14/7 days."
            ),
            key_evidence=[
                "x509: certificate has expired appears at exactly the onset time",
                "tls_handshake_failures goes 0 -> 2400 as a step",
                "CPU DROPPED — the service is doing less work, not struggling",
                "no deploy anywhere near the onset time",
                "checkout-api 401s are downstream fallout, not the cause",
            ],
            red_herrings=[
                "checkout-api also shows errors, which invites investigating the wrong service",
            ],
        ),
        runbook_hint="auth-outage",
    )


# ---------------------------------------------------------------------------
# 5. Cascading failure from an upstream dependency
# ---------------------------------------------------------------------------


def build_dependency_cascade() -> Scenario:
    rng = random.Random(1005)
    onset = _onset_at()

    metrics = {
        "inventory-service.latency_p99_ms": series(
            "inventory-service",
            "latency_p99_ms",
            "ms",
            ANCHOR,
            WINDOW_MINUTES,
            step(85, 9800, ONSET - 0.05),
            0.1,
            rng,
        ),
        "inventory-service.error_rate": series(
            "inventory-service",
            "error_rate",
            "%",
            ANCHOR,
            WINDOW_MINUTES,
            step(0.1, 47.0, ONSET - 0.05),
            0.1,
            rng,
        ),
        "checkout-api.error_rate": series(
            "checkout-api",
            "error_rate",
            "%",
            ANCHOR,
            WINDOW_MINUTES,
            step(0.2, 31.0, ONSET),
            0.1,
            rng,
        ),
        "checkout-api.latency_p99_ms": series(
            "checkout-api",
            "latency_p99_ms",
            "ms",
            ANCHOR,
            WINDOW_MINUTES,
            step(240, 10200, ONSET),
            0.08,
            rng,
        ),
        "search-api.error_rate": series(
            "search-api",
            "error_rate",
            "%",
            ANCHOR,
            WINDOW_MINUTES,
            step(0.1, 18.0, ONSET + 0.02),
            0.12,
            rng,
        ),
        "inventory-service.threads_blocked": series(
            "inventory-service",
            "threads_blocked",
            "count",
            ANCHOR,
            WINDOW_MINUTES,
            ramp(2, 190, ONSET - 0.05),
            0.08,
            rng,
        ),
        "redis-inventory.latency_p99_ms": series(
            "redis-inventory",
            "latency_p99_ms",
            "ms",
            ANCHOR,
            WINDOW_MINUTES,
            spike(1.2, 9400, ONSET - 0.06, 0.3),
            0.1,
            rng,
        ),
    }

    logs = []
    logs += emit(
        "redis-inventory",
        ANCHOR,
        WINDOW_MINUTES,
        6,
        LogLevel.WARN,
        [
            "MISCONF Redis is configured to save RDB snapshots but is unable to persist to disk",
            "background saving error, fork failed: Cannot allocate memory",
        ],
        rng,
        begin=ONSET - 0.08,
        with_trace=False,
    )
    logs += emit(
        "inventory-service",
        ANCHOR,
        WINDOW_MINUTES,
        20,
        LogLevel.ERROR,
        [
            "redis command timed out after {ms}ms (GET inventory:sku:{id})",
            "circuit breaker OPEN for redis-inventory",
            "thread pool saturated, {n} tasks queued",
        ],
        rng,
        begin=ONSET - 0.05,
    )
    logs += emit(
        "checkout-api",
        ANCHOR,
        WINDOW_MINUTES,
        16,
        LogLevel.ERROR,
        [
            "upstream inventory-service returned 503 after {ms}ms",
            "cannot reserve stock, aborting checkout",
        ],
        rng,
        begin=ONSET,
    )
    logs += emit(
        "search-api",
        ANCHOR,
        WINDOW_MINUTES,
        8,
        LogLevel.WARN,
        ["inventory enrichment degraded, serving stale stock counts"],
        rng,
        begin=ONSET + 0.02,
    )

    return Scenario(
        slug="dependency_cascade",
        title="Multiple services erroring simultaneously",
        description="Redis cannot fork for RDB persistence; inventory blocks, everything downstream fails.",
        alert=Alert(
            alert_id="alt-40410",
            timestamp=onset + timedelta(minutes=2),
            source="pagerduty",
            title="Multiple services degraded — checkout-api, search-api, inventory-service",
            body=(
                "3 services breached error-rate SLO within 4 minutes.\n"
                "  checkout-api      error_rate 31.0%\n"
                "  inventory-service error_rate 47.0%\n"
                "  search-api        error_rate 18.0%\n"
                "No deploys in the last 6 hours.\n"
                "Escalated to primary on-call."
            ),
            service=None,
            labels={"severity": "critical", "team": "platform", "env": "production"},
        ),
        services={
            "inventory-service": ServiceHealth(
                service="inventory-service",
                status="degraded",
                replicas_desired=5,
                replicas_ready=5,
                restarts_last_hour=0,
                version="v7.1.3",
            ),
            "checkout-api": ServiceHealth(
                service="checkout-api",
                status="degraded",
                replicas_desired=6,
                replicas_ready=6,
                restarts_last_hour=0,
                version="v2.30.4",
            ),
            "search-api": ServiceHealth(
                service="search-api",
                status="degraded",
                replicas_desired=3,
                replicas_ready=3,
                restarts_last_hour=0,
                version="v1.9.2",
            ),
            "redis-inventory": ServiceHealth(
                service="redis-inventory",
                status="degraded",
                replicas_desired=1,
                replicas_ready=1,
                restarts_last_hour=0,
                version="7.2.4",
            ),
        },
        deploys=[
            Deploy(
                deploy_id="dpl-8760",
                service="inventory-service",
                version="v7.1.3",
                timestamp=ANCHOR - timedelta(hours=31),
                author="omar.h",
                commit_sha="5518cd90ba7e2f61",
                summary="perf: batch stock lookups",
            ),
        ],
        logs=sorted(logs, key=lambda e: e.timestamp),
        metrics=metrics,
        ground_truth=GroundTruth(
            root_cause=(
                "redis-inventory could not fork for RDB persistence (overcommit_memory "
                "misconfigured / host memory pressure), which stalled command handling. "
                "inventory-service threads blocked on Redis calls until its pool saturated, "
                "and checkout-api plus search-api failed downstream. The origin is Redis, "
                "not any of the three services that alerted."
            ),
            category="dependency_failure",
            severity=Severity.P1,
            affected_service="redis-inventory",
            correct_remediation=(
                "Set vm.overcommit_memory=1 on the Redis host and relieve memory pressure, "
                "or disable RDB snapshotting if AOF is sufficient. Then let the circuit "
                "breakers close. Do not roll back the application services."
            ),
            key_evidence=[
                "redis-inventory latency spikes FIRST, before any application service",
                "MISCONF / fork failed: Cannot allocate memory in redis logs",
                "inventory-service threads_blocked climbing to 190",
                "no deploys in the last 6 hours, so this is not change-induced",
                "the onset order redis -> inventory -> checkout -> search reveals direction",
            ],
            red_herrings=[
                "Three services alert at once, inviting a rollback of the wrong thing",
                "The alert names checkout-api first because it is the most visible",
            ],
        ),
        runbook_hint="multi-service-degradation",
    )


SCENARIOS: dict[str, callable] = {
    "bad_deploy": build_bad_deploy,
    "memory_leak": build_memory_leak,
    "pool_exhaustion": build_pool_exhaustion,
    "cert_expiry": build_cert_expiry,
    "dependency_cascade": build_dependency_cascade,
}


def load_scenario(slug: str) -> Scenario:
    if slug not in SCENARIOS:
        raise KeyError(f"unknown scenario {slug!r}; available: {sorted(SCENARIOS)}")
    return SCENARIOS[slug]()


def list_scenarios() -> list[str]:
    return sorted(SCENARIOS)
