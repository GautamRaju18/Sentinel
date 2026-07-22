"""Command line entry point."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sentinel.agents import run_agent
from sentinel.agents.prompts import INVESTIGATOR_SYSTEM
from sentinel.logging_setup import configure_logging
from sentinel.models.router import ModelTier, describe_routing
from sentinel.tools import get_tools
from simulator.scenarios import list_scenarios
from simulator.world import load_world

app = typer.Typer(help="Sentinel — autonomous incident response agent", no_args_is_help=True)
c = Console()


@app.command()
def routing() -> None:
    """Show which model each tier resolves to."""
    c.print(Panel(describe_routing(), title="model routing", border_style="cyan"))


@app.command()
def scenarios() -> None:
    """List the available incident scenarios."""
    table = Table(title="scenarios", show_lines=False)
    table.add_column("slug", style="cyan")
    table.add_column("title")
    table.add_column("alert source", style="dim")
    for slug in list_scenarios():
        w = load_world(slug)
        table.add_row(slug, w.scenario.title, w.alert.source)
    c.print(table)


@app.command()
def investigate(
    scenario: str = typer.Argument(..., help="Scenario slug, e.g. 'bad_deploy'"),
    tier: str = typer.Option("worker", help="Model tier: worker | reasoner | planner"),
    max_steps: int = typer.Option(12, help="Tool-calling steps before giving up"),
    show_answer: bool = typer.Option(False, help="Print the ground truth afterwards"),
    verbose: bool = typer.Option(False, "-v", help="Debug logging"),
) -> None:
    """Investigate an incident with the Phase 1 single-agent loop."""
    configure_logging("DEBUG" if verbose else "WARNING")
    world = load_world(scenario)

    c.print(Panel(world.alert.render(), title=f"ALERT {world.alert.alert_id}", border_style="red"))

    def on_event(kind: str, payload: dict) -> None:
        if kind == "step":
            c.print(f"\n[dim]── step {payload['step']} ──[/]")
        elif kind == "tool_calls":
            c.print(f"  [yellow]→[/] calling: {', '.join(payload['names'])}")
        elif kind == "tool_result":
            preview = payload["preview"].replace("\n", " ")[:110]
            c.print(f"  [green]←[/] {payload['name']} [dim]({payload['ms']}ms)[/] {preview}")

    run = asyncio.run(
        run_agent(
            task=(
                "Investigate this production alert and determine the root cause.\n\n"
                f"{world.alert.render()}"
            ),
            tools=get_tools(read_only=True),
            system_prompt=INVESTIGATOR_SYSTEM,
            tier=ModelTier(tier),
            max_steps=max_steps,
            on_event=on_event,
        )
    )

    c.print()
    c.print(
        Panel(
            run.final_text or "[no conclusion produced]", title="conclusion", border_style="green"
        )
    )

    stats = Table.grid(padding=(0, 2))
    stats.add_row("steps", str(run.steps))
    stats.add_row("tool calls", str(len(run.invocations)))
    stats.add_row("trajectory", " → ".join(run.tool_sequence) or "none")
    stats.add_row("tokens", f"in={run.input_tokens} out={run.output_tokens}")
    stats.add_row("stopped", run.stopped_because)
    c.print(Panel(stats, title="run stats", border_style="dim"))

    if show_answer:
        gt = world.scenario.ground_truth
        body = (
            f"[bold]root cause[/]\n{gt.root_cause}\n\n"
            f"[bold]category[/] {gt.category}   [bold]severity[/] {gt.severity}   "
            f"[bold]service[/] {gt.affected_service}\n\n"
            f"[bold]correct fix[/]\n{gt.correct_remediation}\n\n"
            f"[bold]key evidence[/]\n" + "\n".join(f"  • {e}" for e in gt.key_evidence)
        )
        if gt.red_herrings:
            body += "\n\n[bold]red herrings[/]\n" + "\n".join(f"  • {r}" for r in gt.red_herrings)
        c.print(Panel(body, title="ground truth", border_style="magenta"))


if __name__ == "__main__":
    app()
