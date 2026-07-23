"""Command line entry point."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sentinel.agents import run_agent
from sentinel.agents.prompts import INVESTIGATOR_SYSTEM
from sentinel.logging_setup import configure_observability as configure_logging
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


@app.command()
def ingest() -> None:
    """Embed the runbook corpus into pgvector."""
    from sentinel.rag.store import ingest_runbooks

    configure_logging("INFO")
    n = asyncio.run(ingest_runbooks())
    c.print(f"[green]✓[/] ingested {n} runbook chunks")


@app.command()
def graph() -> None:
    """Print the graph as Mermaid."""
    from sentinel.graph import render_mermaid

    c.print(render_mermaid())


@app.command()
def tracing() -> None:
    """Show LangSmith tracing status and recent runs."""
    from sentinel.config import get_settings
    from sentinel.observability import configure_tracing, trace_url, verify_tracing

    configure_tracing()
    ok, detail = verify_tracing()
    settings = get_settings()

    body = [
        f"enabled: {'[green]yes[/]' if ok else '[red]no[/]'}",
        f"detail:  {detail}",
        f"project: {settings.langsmith_project}",
    ]
    if ok:
        body.append(f"dashboard: {trace_url()}")
    c.print(Panel("\n".join(body), title="LangSmith tracing", border_style="cyan"))

    if not ok:
        c.print("[dim]To enable: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env[/]")
        return

    try:
        from langsmith import Client

        runs = list(
            Client(api_key=settings.langsmith_api_key).list_runs(
                project_name=settings.langsmith_project, limit=10
            )
        )
    except Exception as e:
        c.print(f"[yellow]could not list runs: {type(e).__name__}: {e}[/]")
        return

    if not runs:
        c.print("[dim]no runs recorded yet — run an incident to generate traces[/]")
        return

    table = Table(title=f"recent traces ({len(runs)})")
    table.add_column("name")
    table.add_column("type", style="dim")
    table.add_column("started", style="dim")
    table.add_column("tokens", justify="right")
    for r in runs:
        total = getattr(r, "total_tokens", None)
        table.add_row(str(r.name)[:34], str(r.run_type), str(r.start_time)[:19], str(total or "—"))
    c.print(table)


def _render_plan(plan) -> Panel:
    body = [f"[bold]{plan.summary}[/]", ""]
    for i, s in enumerate(plan.steps, 1):
        body.append(f"  {i}. [cyan]{s.action}[/]  target=[yellow]{s.target}[/]")
        body.append(f"     blast radius: [red]{s.blast_radius}[/]  reversible: {s.reversible}")
        body.append(f"     {s.rationale}")
    body += ["", f"[bold]expected effect[/]  {plan.expected_effect}"]
    if plan.risks:
        body += ["[bold]risks[/]"] + [f"  • {r}" for r in plan.risks]
    body += [
        f"[bold]rollback[/]  {plan.rollback_plan}",
        f"[bold]do nothing[/]  {plan.do_nothing_option}",
    ]
    return Panel(
        "\n".join(body), title="REMEDIATION PLAN — awaiting approval", border_style="yellow"
    )


@app.command()
def respond(
    scenario: str = typer.Argument(..., help="Scenario slug"),
    auto_approve: bool = typer.Option(False, help="Approve without prompting (demos only)"),
    reject: bool = typer.Option(False, help="Reject the plan instead of approving"),
    show_answer: bool = typer.Option(False, help="Print ground truth at the end"),
    verbose: bool = typer.Option(False, "-v"),
) -> None:
    """Run the full incident response graph, pausing for human approval."""
    from sentinel.runner import resume_with_decision, run_until_pause, start_incident

    configure_logging("INFO" if verbose else "WARNING")
    handle, state = start_incident(scenario)
    world = load_world(scenario)

    c.print(Panel(world.alert.render(), title=f"ALERT {world.alert.alert_id}", border_style="red"))
    c.print(f"[dim]incident {handle.incident_id} · thread {handle.thread_id}[/]\n")

    paused: dict = {}

    async def phase_one() -> None:
        async for node, update in run_until_pause(handle, state):
            if node == "__paused__":
                paused.update(update)
            else:
                _print_node(node, update)

    asyncio.run(phase_one())

    values = paused.get("values", {})
    plan = values.get("plan")

    if not paused.get("awaiting_approval"):
        c.print("[yellow]graph finished without reaching the approval gate[/]")
        return

    c.print()
    c.print(_render_plan(plan))

    approved = (
        not reject
        if (auto_approve or reject)
        else typer.confirm("\nApprove this remediation plan?", default=False)
    )
    note = "approved via CLI" if approved else "rejected via CLI"

    async def phase_two() -> None:
        async for node, update in resume_with_decision(handle, approved=approved, note=note):
            if node != "__done__":
                _print_node(node, update)

    asyncio.run(phase_two())

    _print_summary(handle, values, world, show_answer)


def _print_node(node: str, update: dict) -> None:
    if not isinstance(update, dict):
        return
    icons = {
        "triage": "🏷",
        "retrieve": "📚",
        "investigate": "🔍",
        "synthesize": "🧩",
        "critique": "⚖",
        "plan": "📋",
        "approval": "🔐",
        "execute": "⚡",
        "verify": "✅",
        "postmortem": "📝",
    }
    c.print(f"[bold cyan]{icons.get(node, '•')} {node}[/]")

    if t := update.get("triage"):
        c.print(
            f"   {t.severity} / {t.category} · service={t.affected_service} "
            f"· needs_human={t.needs_human}"
        )
    if (ev := update.get("evidence")) is not None and ev:
        c.print(f"   gathered {len(ev)} observations")
    if h := update.get("hypothesis"):
        c.print(f"   [green]{h.root_cause[:200]}[/]")
        c.print(f"   confidence={h.confidence} trigger={h.trigger}")
    if cr := update.get("critique"):
        colour = "green" if cr.verdict == "accept" else "yellow"
        c.print(f"   [{colour}]{cr.verdict}[/] score={cr.score}/10 — {cr.reasoning[:160]}")
        for q in cr.next_questions[:3]:
            c.print(f"     ? {q}")
    if acts := update.get("executed_actions"):
        for a in acts:
            c.print(f"   [magenta]{a[:220]}[/]")
    if v := update.get("verification"):
        c.print(f"   resolved={v.resolved}")
    if pm := update.get("postmortem"):
        c.print(f"   [bold]{pm.title}[/]")
        c.print(f"   lesson: {pm.lesson[:200]}")
    for flag in update.get("security_flags") or []:
        c.print(f"   [red]⚠ {flag}[/]")
    for err in update.get("errors") or []:
        c.print(f"   [red]error: {err[:200]}[/]")


def _print_summary(handle, values: dict, world, show_answer: bool) -> None:
    usage = values.get("token_usage", {})
    grid = Table.grid(padding=(0, 2))
    grid.add_row("incident", handle.incident_id)
    grid.add_row("loops", str(values.get("loop_count", 0)))
    grid.add_row("evidence", str(len(values.get("evidence") or [])))
    grid.add_row("tokens", f"in={usage.get('input', 0)} out={usage.get('output', 0)}")
    grid.add_row("trajectory", " → ".join((values.get("tool_trajectory") or [])[:12]) or "none")
    c.print(Panel(grid, title="run summary", border_style="dim"))

    if show_answer:
        gt = world.scenario.ground_truth
        c.print(
            Panel(
                f"[bold]root cause[/]\n{gt.root_cause}\n\n"
                f"[bold]correct fix[/]\n{gt.correct_remediation}",
                title="ground truth",
                border_style="magenta",
            )
        )


if __name__ == "__main__":
    app()
