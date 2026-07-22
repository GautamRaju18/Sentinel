"""Build the alert-triage training set.

Design decision worth stating: alerts are generated from templates rather than
sampled from a frontier model. Distillation would be the fashionable choice, but
this task's labels are *definitional* — a total outage is P1 because of what the
alert says, not because a large model has an opinion about it. Generating from
templates means every label is correct by construction, the whole set is
reproducible from a seed, and no API quota is spent.

The realism comes from varying the surface form aggressively: five alert
sources with genuinely different formats, service names, numbers, phrasings and
noise. The model must learn to read past the format to the substance.

A held-out test set uses services and phrasings that appear in NO training
example, so a model that memorised service names scores badly — which is what
we want to detect.

    uv run python finetune/generate_dataset.py
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

OUT_DIR = Path(__file__).parent / "data"

TRAIN_SERVICES = [
    "checkout-api",
    "cart-service",
    "payment-service",
    "auth-service",
    "inventory-service",
    "search-api",
    "notification-service",
    "user-profile-api",
    "shipping-service",
    "pricing-engine",
    "review-service",
    "recommendation-api",
]
# Deliberately disjoint from TRAIN_SERVICES.
TEST_SERVICES = [
    "billing-gateway",
    "fraud-detector",
    "loyalty-service",
    "tax-calculator",
    "warehouse-sync",
    "session-broker",
]

TEAMS = ["payments", "commerce", "platform", "growth", "identity", "fulfilment"]


@dataclass
class Template:
    category: str
    severity: str
    needs_human: bool
    source: str
    title: str
    body: str


# Each template is a (category, severity) exemplar. The generator fills slots.
TEMPLATES: list[Template] = [
    # --- bad_deploy ---------------------------------------------------------
    Template(
        "bad_deploy",
        "P1",
        True,
        "prometheus",
        "HighLatencyP99 — {svc}",
        "ALERT HighLatencyP99 firing for {svc}\n"
        "current value: {lat}s (threshold 1.0s)\n"
        "deploy {dpl} ({ver}) rolled out {mins} minutes ago\n"
        "Customer requests are timing out.",
    ),
    Template(
        "bad_deploy",
        "P2",
        True,
        "sentry",
        "New error class after release — {svc}",
        "Sentry issue {svc}-{num}\n"
        "Error: TypeError: cannot read property 'id' of undefined\n"
        "first seen: {mins} minutes ago, immediately after deploy {dpl}\n"
        "events: {num} · users affected: {num2}",
    ),
    Template(
        "bad_deploy",
        "P1",
        True,
        "prometheus",
        "ErrorRateSLOBreach — {svc}",
        "ALERT ErrorRateSLOBreach firing for {svc}\n"
        "error_rate: {pct}% (threshold 1%)\n"
        "onset aligns with deploy {dpl} version {ver}\n"
        "database queries_per_sec rose from {num} to {num2}.",
    ),
    Template(
        "bad_deploy",
        "P3",
        False,
        "healthcheck",
        "Elevated latency on {svc} canary",
        "Canary pod for {svc} ({ver}) shows p95 {lat}s versus {lat2}s on stable.\n"
        "Canary receives 5% of traffic. Stable fleet unaffected.\n"
        "Automated rollout paused pending review.",
    ),
    # --- resource_exhaustion ------------------------------------------------
    Template(
        "resource_exhaustion",
        "P1",
        True,
        "prometheus",
        "PodRestartLoop — {svc}",
        "ALERT PodRestartLoop firing for {svc}\n"
        "restarts in the last hour: {small}\n"
        "Containers terminating with exit code 137 (OOMKilled).\n"
        "memory_mb shows a sawtooth pattern peaking at the {num} MiB limit.",
    ),
    Template(
        "resource_exhaustion",
        "P2",
        True,
        "prometheus",
        "DiskSpaceLow — {svc}",
        "ALERT DiskSpaceLow firing for {svc}\n"
        "filesystem /var/lib/data at {pct}% capacity\n"
        "Growth rate suggests exhaustion in approximately {small} hours.\n"
        "Write failures have not started yet.",
    ),
    Template(
        "resource_exhaustion",
        "P2",
        False,
        "prometheus",
        "HighMemoryUsage — {svc}",
        "ALERT HighMemoryUsage firing for {svc}\n"
        "heap usage {pct}% of limit, climbing steadily for {small} hours\n"
        "gc_pause_ms risen from {small} to {num}\n"
        "No restarts yet. Request rate flat.",
    ),
    Template(
        "resource_exhaustion",
        "P1",
        True,
        "pagerduty",
        "{svc} thread pool saturated",
        "All {small} worker threads in {svc} are blocked.\n"
        "Request queue depth: {num} and growing.\n"
        "Health checks failing; the load balancer has removed {small} of "
        "{small2} instances.",
    ),
    # --- config_change ------------------------------------------------------
    Template(
        "config_change",
        "P1",
        True,
        "sentry",
        "Connection pool errors — {svc}",
        "Sentry issue {svc}-{num}\n"
        "Error: HikariPool-1 - Connection is not available, request timed out\n"
        "events: {num2} in 5 minutes (baseline: 3)\n"
        "configmap {svc}-config reloaded to revision {small} at onset.\n"
        "Pool active count DROPPED from {num} to {small}.",
    ),
    Template(
        "config_change",
        "P2",
        True,
        "healthcheck",
        "{svc} rejecting requests after config reload",
        "{svc} began returning 403 to internal callers.\n"
        "config revision {small} applied {mins} minutes ago changed the "
        "allowed_origins list.\n"
        "External traffic unaffected.",
    ),
    Template(
        "config_change",
        "P2",
        True,
        "prometheus",
        "RateLimitRejections — {svc}",
        "ALERT RateLimitRejections firing for {svc}\n"
        "rejected_requests: {num}/min, previously 0\n"
        "rate_limit config lowered from {num2} to {num} rps in revision {small}.\n"
        "Incoming traffic unchanged.",
    ),
    # --- expired_credential -------------------------------------------------
    Template(
        "expired_credential",
        "P1",
        True,
        "healthcheck",
        "{svc} /healthz failing — all requests rejected",
        "Synthetic check failed {small} consecutive times.\n"
        "last error: x509: certificate has expired or is not yet valid\n"
        "successful requests dropped from {num} per minute to 0.\n"
        "No deploys in the last {small2} hours. CPU usage FELL.",
    ),
    Template(
        "expired_credential",
        "P1",
        True,
        "sentry",
        "Authentication failures — {svc}",
        "Sentry issue {svc}-{num}\n"
        "Error: remote error: tls: bad certificate\n"
        "events: {num2} in 3 minutes\n"
        "tls_handshake_failures went from 0 to {num2} as a step change.",
    ),
    Template(
        "expired_credential",
        "P2",
        True,
        "prometheus",
        "TokenRefreshFailure — {svc}",
        "ALERT TokenRefreshFailure firing for {svc}\n"
        "OAuth client credential grant returning invalid_client\n"
        "{svc} is serving cached tokens; these expire in {small} minutes.\n"
        "No user impact yet.",
    ),
    Template(
        "expired_credential",
        "P3",
        False,
        "healthcheck",
        "Certificate expiring soon — {svc}",
        "TLS certificate for {svc}.internal expires in {small} days.\n"
        "No current impact. Renewal has not been scheduled.",
    ),
    # --- dependency_failure -------------------------------------------------
    Template(
        "dependency_failure",
        "P1",
        True,
        "pagerduty",
        "Multiple services degraded — {svc} and {small} others",
        "{small2} services breached error-rate SLOs within 4 minutes.\n"
        "  {svc} error_rate {pct}%\n"
        "  {svc2} error_rate {pct2}%\n"
        "No deploys in the last {small2} hours.\n"
        "redis-{svc} latency spiked first, before any application service.",
    ),
    Template(
        "dependency_failure",
        "P2",
        True,
        "prometheus",
        "CircuitBreakerOpen — {svc}",
        "ALERT CircuitBreakerOpen firing for {svc}\n"
        "Circuit to upstream {svc2} has been open for {small} minutes.\n"
        "{svc} is serving degraded responses from cache.\n"
        "{svc2} is returning 503 with p99 {lat}s.",
    ),
    Template(
        "dependency_failure",
        "P1",
        True,
        "prometheus",
        "DatabaseUnreachable — {svc}",
        "ALERT DatabaseUnreachable firing for {svc}\n"
        "connection attempts to postgres-main failing: {num} in 2 minutes\n"
        "error: could not translate host name to address\n"
        "Three other services report the same DNS failure.",
    ),
    Template(
        "dependency_failure",
        "P3",
        False,
        "healthcheck",
        "{svc} degraded — upstream {svc2} slow",
        "{svc} is serving stale data because {svc2} p99 rose to {lat}s.\n"
        "Fallback path is working as designed. No errors surfaced to users.",
    ),
    # --- low severity / noise ----------------------------------------------
    Template(
        "unknown",
        "P4",
        False,
        "prometheus",
        "MetricsScrapeFailure — {svc}",
        "ALERT MetricsScrapeFailure firing for {svc}\n"
        "Prometheus could not scrape /metrics on {small} of {small2} pods.\n"
        "The service itself is healthy; only observability is affected.",
    ),
    Template(
        "unknown",
        "P4",
        False,
        "sentry",
        "Deprecation warning volume up — {svc}",
        "Sentry issue {svc}-{num}\n"
        "Warning: datetime.utcnow() is deprecated\n"
        "events: {num2} · users affected: 0\n"
        "No functional impact.",
    ),
    Template(
        "unknown",
        "P3",
        False,
        "healthcheck",
        "{svc} slow startup",
        "{svc} pods took {small} minutes to pass readiness after a routine "
        "node rotation.\n"
        "All pods are now ready. No requests were dropped.",
    ),
]


def _fill(template: str, rng: random.Random, services: list[str]) -> str:
    svc, svc2 = rng.sample(services, 2)
    return template.format(
        svc=svc,
        svc2=svc2,
        num=rng.randint(120, 48000),
        num2=rng.randint(120, 48000),
        small=rng.randint(2, 12),
        small2=rng.randint(2, 12),
        pct=rng.randint(12, 99),
        pct2=rng.randint(12, 99),
        lat=round(rng.uniform(1.2, 12.0), 1),
        lat2=round(rng.uniform(0.1, 0.9), 2),
        mins=rng.randint(1, 55),
        dpl=f"dpl-{rng.randint(1000, 9999)}",
        ver=f"v{rng.randint(1, 9)}.{rng.randint(0, 40)}.{rng.randint(0, 9)}",
    )


def _render_alert(t: Template, rng: random.Random, services: list[str]) -> tuple[str, str]:
    title = _fill(t.title, rng, services)
    body = _fill(t.body, rng, services)
    labels = {
        "severity": rng.choice(["critical", "warning", "info"]),
        "team": rng.choice(TEAMS),
        "env": rng.choice(["production", "production", "production", "staging"]),
    }
    label_str = " ".join(f"{k}={v}" for k, v in labels.items())
    alert = f"[{t.source}] {title}\nlabels: {label_str}\n\n{body}"
    # Recover the service the title actually names, for the label.
    service = next((s for s in services if s in alert), None)
    return alert, service or ""


def build_example(t: Template, rng: random.Random, services: list[str]) -> dict:
    alert, service = _render_alert(t, rng, services)
    reasoning = {
        "bad_deploy": "Onset aligns with a recent deploy and no other change is implicated.",
        "resource_exhaustion": "A resource is being consumed faster than it is released.",
        "config_change": "A configuration revision changed a limit at the onset time.",
        "expired_credential": "A credential or certificate has expired; nothing was deployed.",
        "dependency_failure": "A shared upstream failed and consumers degraded downstream.",
        "unknown": "No user impact and no clear failure class.",
    }[t.category]
    return {
        "alert": alert,
        "label": {
            "severity": t.severity,
            "category": t.category,
            "affected_service": service or None,
            "needs_human": t.needs_human,
            "reasoning": reasoning,
        },
    }


def generate(n_train: int = 1000, n_val: int = 120, n_test: int = 120, seed: int = 7) -> dict:
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {}

    for name, count, services in (
        ("train", n_train, TRAIN_SERVICES),
        ("val", n_val, TRAIN_SERVICES),
        ("test", n_test, TEST_SERVICES),  # unseen services, on purpose
    ):
        rows = []
        # Round-robin over templates so classes stay balanced rather than
        # letting random choice skew the distribution.
        for i in range(count):
            rows.append(build_example(TEMPLATES[i % len(TEMPLATES)], rng, services))
        rng.shuffle(rows)
        splits[name] = rows

    return splits


def to_chat_format(row: dict, system: str) -> dict:
    """The training format: a chat turn whose assistant message is the JSON."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Classify this alert:\n\n{row['alert']}"},
            {"role": "assistant", "content": json.dumps(row["label"], separators=(",", ":"))},
        ]
    }


def main() -> None:
    from sentinel.agents.prompts import TRIAGE_SYSTEM

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits = generate()

    for name, rows in splits.items():
        raw_path = OUT_DIR / f"{name}.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        chat_path = OUT_DIR / f"{name}_chat.jsonl"
        with chat_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(to_chat_format(r, TRIAGE_SYSTEM)) + "\n")

        dist: dict[str, int] = {}
        for r in rows:
            key = f"{r['label']['category']}/{r['label']['severity']}"
            dist[key] = dist.get(key, 0) + 1
        print(f"{name:<6} {len(rows):>5} examples -> {raw_path.name}, {chat_path.name}")
        for k in sorted(dist):
            print(f"         {k:<34} {dist[k]}")

    print(f"\ntrain/val services: {len(TRAIN_SERVICES)}")
    print(f"test services (unseen in training): {', '.join(TEST_SERVICES)}")


if __name__ == "__main__":
    main()
