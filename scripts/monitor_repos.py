"""Monitor your GitHub repositories — health table, alerts, and CI investigation.

Three modes in one command:

    uv run python scripts/monitor_repos.py                 # health dashboard
    uv run python scripts/monitor_repos.py --alerts-only   # only what needs attention
    uv run python scripts/monitor_repos.py --investigate   # + diagnose any failing CI

Read-only throughout. The --investigate flag reuses Sentinel's agent loop on a
real failed run; it produces a diagnosis, never a change.
"""

from __future__ import annotations

import argparse
import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sentinel.integrations.github import GitHubMonitor, RepoHealth
from sentinel.logging_setup import configure_logging

c = Console()

STATUS_STYLE = {
    "passing": ("green", "✓ passing"),
    "failing": ("red", "✗ failing"),
    "running": ("yellow", "● running"),
    "no-ci": ("dim", "— no CI"),
    "no-runs": ("dim", "— no runs"),
    "error": ("red", "! error"),
}


def health_table(healths: list[RepoHealth], alerts_only: bool) -> Table:
    t = Table(title="Repository health", show_lines=False)
    t.add_column("repo", style="cyan", no_wrap=True)
    t.add_column("CI")
    t.add_column("PRs", justify="right")
    t.add_column("issues", justify="right")
    t.add_column("★", justify="right", style="dim")
    t.add_column("last push", justify="right", style="dim")
    t.add_column("latest commit", style="dim")

    shown = 0
    for h in healths:
        if alerts_only and not h.needs_attention:
            continue
        shown += 1
        colour, label = STATUS_STYLE.get(h.ci_status, ("white", h.ci_status))
        push = f"{h.days_since_push}d ago" if h.days_since_push >= 0 else "—"
        commit = h.latest_run.commit_message[:38] if h.latest_run else ""
        t.add_row(
            h.name + (" 🔒" if h.private else ""),
            f"[{colour}]{label}[/]",
            str(h.open_prs) if h.open_prs else "",
            str(h.open_issues) if h.open_issues else "",
            str(h.stars) if h.stars else "",
            push,
            commit,
        )
    if shown == 0:
        t.add_row("[green]nothing needs attention[/]", "", "", "", "", "", "")
    return t


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts-only", action="store_true", help="Only rows needing attention")
    ap.add_argument("--investigate", action="store_true", help="Diagnose any failing CI run")
    ap.add_argument("--repos", default="", help="Comma-separated subset (default: watchlist)")
    args = ap.parse_args()
    configure_logging("WARNING")

    monitor = GitHubMonitor()
    rl = monitor.rate_limit()
    subset = [r.strip() for r in args.repos.split(",") if r.strip()] or None

    with c.status("Querying GitHub…"):
        healths = monitor.health_all(subset)

    c.print(health_table(healths, args.alerts_only))

    failing = [h for h in healths if h.ci_status == "failing"]
    attention = [h for h in healths if h.needs_attention]
    c.print(
        f"\n[dim]{len(healths)} repos · {len(failing)} failing CI · "
        f"{len(attention)} need attention · "
        f"API budget {rl['remaining']}/{rl['limit']}[/]"
    )

    if args.investigate and failing:
        from sentinel.integrations.ci_investigator import investigate_run

        for h in failing:
            if not h.latest_run:
                continue
            c.print(f"\n[bold red]Investigating {h.name} — run {h.latest_run.id}…[/]")
            inv = await investigate_run(h.name, h.latest_run, monitor)
            c.print(Panel(inv.diagnosis, title=f"{h.name}: diagnosis", border_style="red"))
            c.print(
                f"[dim]tools: {' → '.join(inv.tool_trajectory)} · "
                f"tokens in={inv.input_tokens} out={inv.output_tokens}[/]"
            )
    elif args.investigate:
        c.print("\n[green]No failing CI runs to investigate.[/]")

    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
