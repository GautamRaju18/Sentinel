"""Compare baseline against fine-tuned triage on identical inputs.

The headline claim of Phase 6 is "same quality, less cost". This prints the
table that either supports it or does not.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

c = Console()
HERE = Path(__file__).parent


def load(name: str) -> dict | None:
    path = HERE / f"results_{name}.json"
    if not path.exists():
        c.print(f"[yellow]missing {path.name}[/] — run triage_eval.py with --out {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


ROWS = [
    ("severity accuracy", "severity_accuracy", "pct", "up"),
    ("severity macro-F1", "severity_macro_f1", "f3", "up"),
    ("category accuracy", "category_accuracy", "pct", "up"),
    ("category macro-F1", "category_macro_f1", "f3", "up"),
    ("needs_human accuracy", "needs_human_accuracy", "pct", "up"),
    ("critical underestimates", "critical_underestimate_rate", "pct", "down"),
    ("latency p50 (ms)", "latency_p50_ms", "f0", "down"),
    ("latency p95 (ms)", "latency_p95_ms", "f0", "down"),
]


def fmt(value: float, kind: str) -> str:
    return {
        "pct": f"{value * 100:.1f}%",
        "f3": f"{value:.3f}",
        "f0": f"{value:.0f}",
    }[kind]


def main() -> None:
    baseline, finetuned = load("baseline"), load("finetuned")
    if not baseline or not finetuned:
        raise SystemExit(1)

    t = Table(title="triage: baseline vs fine-tuned")
    t.add_column("metric")
    t.add_column(f"baseline\n[dim]{baseline['backend']}[/]", justify="right")
    t.add_column(f"fine-tuned\n[dim]{finetuned['backend']}[/]", justify="right")
    t.add_column("delta", justify="right")

    for label, key, kind, better in ROWS:
        b, f = baseline.get(key, 0), finetuned.get(key, 0)
        delta = f - b
        improved = (delta > 0) if better == "up" else (delta < 0)
        colour = (
            "green" if improved and abs(delta) > 1e-9 else ("red" if abs(delta) > 1e-9 else "dim")
        )
        sign = "+" if delta > 0 else ""
        t.add_row(label, fmt(b, kind), fmt(f, kind), f"[{colour}]{sign}{fmt(delta, kind)}[/]")

    c.print(t)
    c.print(
        "\n[dim]The fine-tune is a win if quality holds within noise while latency "
        "drops. Beating the baseline on accuracy is a bonus, not the claim.[/]"
    )


if __name__ == "__main__":
    main()
