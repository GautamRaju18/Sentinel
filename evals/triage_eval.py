"""Evaluate the triage classifier.

This is the measurement that makes Phase 6 a result rather than a claim. The
same held-out set runs through whichever backend TRIAGE_BACKEND selects, so
baseline and fine-tuned are compared on identical inputs.

Metrics chosen deliberately:

  * accuracy per field, because severity and category fail differently
  * macro-F1, because the class distribution is not uniform and a model that
    always answers "P1" would score well on plain accuracy
  * **severity-critical error rate** — how often a P1 is called P3/P4. This is
    the number that actually matters operationally. An unattended P1 is an
    outage nobody is working on; the reverse merely wastes attention.
  * p50/p95 latency and token cost, since the entire argument for a small
    fine-tuned model is that it is cheaper and faster at equal quality.

    uv run python evals/triage_eval.py --limit 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from sentinel.config import get_settings
from sentinel.triage import classify

c = Console()
DATA = Path(__file__).parent.parent / "finetune" / "data"


def load_split(name: str, limit: int | None = None) -> list[dict]:
    path = DATA / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: uv run python finetune/generate_dataset.py")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows[:limit] if limit else rows


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    """Unweighted mean of per-class F1 — every class counts the same."""
    classes = {truth for truth, _ in pairs} | {pred for _, pred in pairs}
    scores = []
    for cls in classes:
        tp = sum(1 for t, p in pairs if t == cls and p == cls)
        fp = sum(1 for t, p in pairs if t != cls and p == cls)
        fn = sum(1 for t, p in pairs if t == cls and p != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


_SEV_RANK = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}


async def evaluate(split: str = "test", limit: int | None = None, concurrency: int = 4) -> dict:
    rows = load_split(split, limit)
    backend = get_settings().triage_backend
    c.print(f"[bold]evaluating[/] backend=[cyan]{backend}[/] split={split} n={len(rows)}\n")

    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def one(row: dict, idx: int) -> None:
        async with sem:
            started = time.perf_counter()
            try:
                pred = await classify(row["alert"])
                ok = True
            except Exception as e:
                c.print(f"[red]row {idx} failed: {e}[/]")
                return
            elapsed = (time.perf_counter() - started) * 1000
            results.append(
                {
                    "truth": row["label"],
                    "pred": pred.model_dump(mode="json"),
                    "ms": elapsed,
                    "ok": ok,
                }
            )
            if len(results) % 10 == 0:
                c.print(f"  [dim]{len(results)}/{len(rows)}[/]")

    await asyncio.gather(*[one(r, i) for i, r in enumerate(rows)])

    sev_pairs = [(r["truth"]["severity"], r["pred"]["severity"]) for r in results]
    cat_pairs = [(r["truth"]["category"], r["pred"]["category"]) for r in results]
    latencies = sorted(r["ms"] for r in results)

    n = len(results) or 1
    sev_acc = sum(1 for t, p in sev_pairs if t == p) / n
    cat_acc = sum(1 for t, p in cat_pairs if t == p) / n

    # Severity off by more than one band, in the dangerous direction.
    critical_misses = [(t, p) for t, p in sev_pairs if _SEV_RANK[p] - _SEV_RANK[t] >= 2]
    # Escalating a P4 to P1 wastes attention but breaks nothing.
    over_escalations = [(t, p) for t, p in sev_pairs if _SEV_RANK[t] - _SEV_RANK[p] >= 2]

    needs_human_acc = (
        sum(1 for r in results if r["truth"]["needs_human"] == r["pred"]["needs_human"]) / n
    )

    return {
        "backend": backend,
        "n": len(results),
        "severity_accuracy": sev_acc,
        "severity_macro_f1": macro_f1(sev_pairs),
        "category_accuracy": cat_acc,
        "category_macro_f1": macro_f1(cat_pairs),
        "needs_human_accuracy": needs_human_acc,
        "critical_underestimates": len(critical_misses),
        "critical_underestimate_rate": len(critical_misses) / n,
        "over_escalations": len(over_escalations),
        "latency_p50_ms": statistics.median(latencies) if latencies else 0,
        "latency_p95_ms": latencies[int(len(latencies) * 0.95)] if latencies else 0,
        "confusion_severity": _confusion(sev_pairs),
        "confusion_category": _confusion(cat_pairs),
    }


def _confusion(pairs: list[tuple[str, str]]) -> dict:
    m: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for truth, pred in pairs:
        m[truth][pred] += 1
    return {k: dict(v) for k, v in m.items()}


def report(metrics: dict) -> None:
    t = Table(title=f"triage eval — backend={metrics['backend']} n={metrics['n']}")
    t.add_column("metric")
    t.add_column("value", justify="right")

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    t.add_row("severity accuracy", pct(metrics["severity_accuracy"]))
    t.add_row("severity macro-F1", f"{metrics['severity_macro_f1']:.3f}")
    t.add_row("category accuracy", pct(metrics["category_accuracy"]))
    t.add_row("category macro-F1", f"{metrics['category_macro_f1']:.3f}")
    t.add_row("needs_human accuracy", pct(metrics["needs_human_accuracy"]))
    t.add_row(
        "[red]critical underestimates[/]",
        f"[red]{metrics['critical_underestimates']} "
        f"({pct(metrics['critical_underestimate_rate'])})[/]",
    )
    t.add_row("over-escalations", str(metrics["over_escalations"]))
    t.add_row("latency p50", f"{metrics['latency_p50_ms']:.0f} ms")
    t.add_row("latency p95", f"{metrics['latency_p95_ms']:.0f} ms")
    c.print(t)

    cm = Table(title="severity confusion (rows = truth)")
    cm.add_column("truth")
    for col in ["P1", "P2", "P3", "P4"]:
        cm.add_column(col, justify="right")
    for truth in ["P1", "P2", "P3", "P4"]:
        row = metrics["confusion_severity"].get(truth, {})
        cm.add_row(truth, *[str(row.get(p, 0)) for p in ["P1", "P2", "P3", "P4"]])
    c.print(cm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=None, help="Write metrics JSON here")
    args = ap.parse_args()

    metrics = asyncio.run(evaluate(args.split, args.limit, args.concurrency))
    report(metrics)

    if args.out:
        Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        c.print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
