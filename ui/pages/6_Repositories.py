"""Monitor your GitHub repositories — health, alerts, and CI investigation."""

from __future__ import annotations

import asyncio

import pandas as pd
import streamlit as st
from common import hero, page_setup, sidebar_status

from sentinel.config import get_settings
from sentinel.integrations.github import GitHubMonitor

page_setup("Repositories", "📦")
sidebar_status()

hero(
    "📦 Repository Monitor",
    "The same incident-response idea, pointed at your real GitHub repos. Health "
    "at a glance, alerts for what needs attention, and — when a CI run fails — the "
    "agent investigates it with read-only tools and hands you a diagnosis.",
)

settings = get_settings()
if not settings.github_token:
    st.error("No GITHUB_TOKEN configured in .env — repository monitoring needs it.")
    st.stop()

STATUS = {
    "passing": ("#16a34a", "✓ passing"),
    "failing": ("#dc2626", "✗ failing"),
    "running": ("#d97706", "● running"),
    "no-ci": ("#94a3b8", "no CI"),
    "no-runs": ("#94a3b8", "no runs"),
    "error": ("#dc2626", "! error"),
}


@st.cache_data(ttl=120, show_spinner=False)
def load_health(subset: tuple[str, ...] | None):
    m = GitHubMonitor()
    healths = m.health_all(list(subset) if subset else None)
    rl = m.rate_limit()
    # Return plain dicts so Streamlit can cache them.
    return [
        {
            "name": h.name,
            "private": h.private,
            "ci_status": h.ci_status,
            "open_prs": h.open_prs,
            "open_issues": h.open_issues,
            "stars": h.stars,
            "days": h.days_since_push,
            "attention": h.needs_attention,
            "commit": h.latest_run.commit_message if h.latest_run else "",
            "run_id": h.latest_run.id if h.latest_run else None,
            "branch": h.latest_run.branch if h.latest_run else "",
            "run_name": h.latest_run.name if h.latest_run else "",
            "conclusion": h.latest_run.conclusion if h.latest_run else None,
        }
        for h in healths
    ], rl


c1, c2 = st.columns([3, 1])
alerts_only = c1.toggle("Show only repos that need attention", value=False)
if c2.button("↻ Refresh", use_container_width=True):
    load_health.clear()

with st.spinner("Querying GitHub…"):
    rows, rl = load_health(None)

# --- summary strip --------------------------------------------------------

failing = [r for r in rows if r["ci_status"] == "failing"]
attention = [r for r in rows if r["attention"]]
a, b, c, d = st.columns(4)
a.metric("Repositories", len(rows))
b.metric("Failing CI", len(failing))
c.metric("Need attention", len(attention))
d.metric("API budget", f"{rl['remaining']}/{rl['limit']}")

# --- table ----------------------------------------------------------------

shown = [r for r in rows if r["attention"]] if alerts_only else rows
if not shown:
    st.success("Nothing needs attention — every watched repo is healthy.")
else:
    table = []
    for r in shown:
        colour, label = STATUS.get(r["ci_status"], ("#000", r["ci_status"]))
        # Every cell a string — a column mixing int and "" makes Streamlit's
        # Arrow backend raise "Expected bytes, got int".
        table.append(
            {
                "Repo": r["name"] + (" 🔒" if r["private"] else ""),
                "CI": label,
                "PRs": str(r["open_prs"]) if r["open_prs"] else "",
                "Issues": str(r["open_issues"]) if r["open_issues"] else "",
                "★": str(r["stars"]) if r["stars"] else "",
                "Last push": f"{r['days']}d ago" if r["days"] >= 0 else "—",
                "Latest commit": r["commit"][:44],
            }
        )
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

# --- CI investigation -----------------------------------------------------

st.divider()
st.markdown("### Investigate a failed CI run")

if failing:
    st.error(f"{len(failing)} repo(s) have a failing CI run.")
    target = st.selectbox("Which one?", [r["name"] for r in failing])
    chosen = next(r for r in failing if r["name"] == target)
    st.caption(
        f"Latest run: **{chosen['run_name']}** on `{chosen['branch']}` — {chosen['commit'][:60]}"
    )
    if st.button("🔬 Investigate with Sentinel", type="primary"):
        from sentinel.integrations.ci_investigator import investigate_run
        from sentinel.integrations.github import WorkflowRun

        run = WorkflowRun(
            id=chosen["run_id"],
            name=chosen["run_name"],
            status="completed",
            conclusion=chosen["conclusion"],
            branch=chosen["branch"],
            event="push",
            created_at="",
            url="",
            commit_message=chosen["commit"],
        )
        with st.status("Agent investigating the failure…", expanded=True) as s:
            inv = asyncio.run(investigate_run(chosen["name"], run, GitHubMonitor()))
            s.update(label="Diagnosis ready", state="complete")
        st.markdown(inv.diagnosis)
        st.caption(
            f"Tools used: {' → '.join(inv.tool_trajectory)} · "
            f"tokens in={inv.input_tokens} out={inv.output_tokens}"
        )
        st.info(
            "This is a diagnosis, not a change. Sentinel investigated read-only and "
            "recommended a fix — acting on it is your call, exactly as with a "
            "production incident."
        )
else:
    st.success("No failing CI runs right now. When one fails, it appears here to investigate.")
    with st.expander("How this works"):
        st.markdown(
            "A CI failure is structurally an incident: something changed, a signal "
            "went red, you need the cause. Sentinel reuses its Phase-1 agent loop — "
            "bind read-only tools (run summary, failed jobs, log tail, recent "
            "commits), let the model decide what to look at, feed results back — "
            "pointed at your real repo instead of the simulator.\n\n"
            "It deliberately does **not** push a fix. Auto-writing to your repo is "
            "exactly the unattended action this whole project argues against. "
            "Read-only in, advice out."
        )

st.caption(
    "All repository access here is read-only — this page never pushes, merges, or closes anything."
)
