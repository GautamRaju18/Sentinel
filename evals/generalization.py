"""Does the triage model generalise, or did it memorise the templates?

The held-out split in finetune/data/test.jsonl varies service names, numbers
and phrasing but reuses the same 21 templates as training. It catches a model
that memorised *entities*. It cannot catch one that memorised *templates* — and
a fine-tune scoring 100% on it is evidence of exactly that ambiguity, not of
understanding.

This suite is the harder question. Its alerts come from two sources that share
no template with the training data:

  1. the five hand-written scenario alerts in simulator/scenarios.py, which
     were written months before the dataset generator existed
  2. adversarial variants — alerts whose surface features point one way and
     whose correct label points another

Group 2 is where a template-matcher and a model that learned the signal come
apart. "Nothing was deployed, CPU FELL, x509 in the logs" is an expired
credential, but every surface cue says outage-therefore-P1-bad-deploy.

A model at 100% on test.jsonl and 50% here memorised templates. That is a
useful, publishable finding — it is not a failure of the experiment.

    uv run python evals/generalization.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from sentinel.config import get_settings
from sentinel.triage import classify
from simulator.scenarios import list_scenarios, load_scenario

c = Console()


def scenario_cases() -> list[dict]:
    """The five hand-written incidents. Ground truth already lives on them."""
    cases = []
    for slug in list_scenarios():
        s = load_scenario(slug)
        cases.append(
            {
                "id": f"scenario:{slug}",
                "alert": s.alert.render(),
                "severity": str(s.ground_truth.severity),
                "category": s.ground_truth.category,
                "source": "hand-written scenario",
            }
        )
    return cases


# Written to defeat surface-feature matching. Each one has a cue pointing at
# the wrong answer and the real signal buried in a detail.
ADVERSARIAL: list[dict] = [
    {
        "id": "adv:cert-looks-like-outage",
        "alert": (
            "[pagerduty] TOTAL OUTAGE — api-gateway returning 500s\n"
            "labels: severity=critical team=platform env=production\n\n"
            "Every request to api-gateway is failing.\n"
            "No deploys in the last 12 hours.\n"
            "CPU utilisation DROPPED from 40% to 6% at onset.\n"
            "logs: x509: certificate has expired or is not yet valid"
        ),
        "severity": "P1",
        "category": "expired_credential",
        "trap": "screams outage; the x509 line and falling CPU are the signal",
    },
    {
        "id": "adv:deploy-is-a-decoy",
        "alert": (
            "[prometheus] HighErrorRate — orders-api\n"
            "labels: severity=critical team=commerce env=production\n\n"
            "orders-api error_rate 34%.\n"
            "A deploy (dpl-9912) landed 3 minutes ago.\n"
            "redis-orders p99 latency rose to 8400ms SIX minutes ago,\n"
            "before the deploy. Two other services also degraded."
        ),
        "severity": "P1",
        "category": "dependency_failure",
        "trap": "a recent deploy invites bad_deploy; the dependency moved first",
    },
    {
        "id": "adv:pool-drop-not-load",
        "alert": (
            "[sentry] Database connection failures — ledger-service\n"
            "labels: severity=critical team=payments env=production\n\n"
            "Error: connection pool exhausted, request timed out.\n"
            "Request rate FLAT at 88 rps.\n"
            "Pool active connections FELL from 35 to 8 at onset.\n"
            "postgres-main is healthy: CPU 18%, 40 connections free."
        ),
        "severity": "P1",
        "category": "config_change",
        "trap": "errors name the database; the ceiling dropped, load did not rise",
    },
    {
        "id": "adv:old-deploy-slow-leak",
        "alert": (
            "[prometheus] PodRestartLoop — media-worker\n"
            "labels: severity=critical team=platform env=production\n\n"
            "9 restarts in the last hour, exit code 137.\n"
            "No deploys in the last 72 hours.\n"
            "memory_mb climbs to the 4096Mi limit then resets, repeatedly.\n"
            "Request rate unchanged for a week."
        ),
        "severity": "P1",
        "category": "resource_exhaustion",
        "trap": "no recent deploy invites 'unknown'; the sawtooth is the signal",
    },
    {
        "id": "adv:scary-words-no-impact",
        "alert": (
            "[sentry] CRITICAL FATAL ERROR — analytics-batch\n"
            "labels: severity=critical team=growth env=production\n\n"
            "Error: FATAL: nightly rollup job exceeded its window.\n"
            "The job completed successfully on retry 20 minutes later.\n"
            "No user-facing surface consumes this data in real time.\n"
            "0 users affected."
        ),
        "severity": "P4",
        "category": "unknown",
        "trap": "CRITICAL/FATAL/severity=critical all lie; impact is zero",
    },
    {
        "id": "adv:config-not-deploy",
        "alert": (
            "[healthcheck] search-api rejecting internal callers\n"
            "labels: severity=warning team=platform env=production\n\n"
            "search-api began returning 403 to three internal services.\n"
            "No code deploy. configmap search-api-config moved to revision 22\n"
            "four minutes before onset, changing allowed_origins.\n"
            "External traffic is unaffected."
        ),
        "severity": "P2",
        "category": "config_change",
        "trap": "labelled warning, and a configmap is easy to miss as 'a deploy'",
    },
]


async def run_case(case: dict) -> dict:
    started = time.perf_counter()
    try:
        pred = await classify(case["alert"])
        pred_sev, pred_cat = str(pred.severity), str(pred.category)
        failed = False
    except Exception as e:
        c.print(f"[red]{case['id']} failed: {e}[/]")
        pred_sev, pred_cat, failed = "?", "?", True
    return {
        **case,
        "pred_severity": pred_sev,
        "pred_category": pred_cat,
        "severity_ok": pred_sev == case["severity"],
        "category_ok": pred_cat == case["category"],
        "ms": (time.perf_counter() - started) * 1000,
        "failed": failed,
    }


async def evaluate() -> dict:
    cases = scenario_cases() + ADVERSARIAL
    backend = get_settings().triage_backend
    c.print(f"[bold]generalization eval[/] backend=[cyan]{backend}[/] n={len(cases)}\n")

    results = []
    for case in cases:  # sequential: local models serialise anyway
        results.append(await run_case(case))
        r = results[-1]
        mark = "[green]OK  [/]" if (r["severity_ok"] and r["category_ok"]) else "[red]MISS[/]"
        c.print(
            f"  {mark} {r['id']:<34} "
            f"pred={r['pred_severity']}/{r['pred_category']:<22} "
            f"true={r['severity']}/{r['category']}"
        )

    scen = [r for r in results if r["id"].startswith("scenario:")]
    adv = [r for r in results if r["id"].startswith("adv:")]

    def acc(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    return {
        "backend": backend,
        "n": len(results),
        "scenario_severity_accuracy": acc(scen, "severity_ok"),
        "scenario_category_accuracy": acc(scen, "category_ok"),
        "adversarial_severity_accuracy": acc(adv, "severity_ok"),
        "adversarial_category_accuracy": acc(adv, "category_ok"),
        "overall_category_accuracy": acc(results, "category_ok"),
        "results": results,
    }


def report(m: dict) -> None:
    t = Table(title=f"generalization — backend={m['backend']}")
    t.add_column("group")
    t.add_column("severity", justify="right")
    t.add_column("category", justify="right")

    def pct(x):
        return f"{x * 100:.0f}%"

    t.add_row(
        "hand-written scenarios (5)",
        pct(m["scenario_severity_accuracy"]),
        pct(m["scenario_category_accuracy"]),
    )
    t.add_row(
        "adversarial (6)",
        pct(m["adversarial_severity_accuracy"]),
        pct(m["adversarial_category_accuracy"]),
    )
    c.print(t)

    misses = [r for r in m["results"] if not (r["severity_ok"] and r["category_ok"])]
    if misses:
        c.print("\n[bold]misses[/]")
        for r in misses:
            c.print(f"  [red]{r['id']}[/]")
            c.print(
                f"    predicted {r['pred_severity']}/{r['pred_category']}, "
                f"expected {r['severity']}/{r['category']}"
            )
            if trap := r.get("trap"):
                c.print(f"    [dim]trap: {trap}[/]")

    c.print(
        "\n[dim]Read this against test.jsonl. Near-perfect there and much weaker "
        "here means the model learned the training templates rather than the "
        "underlying signal — a real finding, and the reason this suite exists.[/]"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    metrics = asyncio.run(evaluate())
    report(metrics)
    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        c.print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
