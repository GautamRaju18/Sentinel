"""Sentinel operator console.

Runs the graph in-process rather than through the HTTP API. For a single
operator that removes an entire class of failure (SSE reconnects, partial JSON,
API process lifecycle) for no loss of fidelity — the API exists for programmatic
clients and is exercised by its own tests.

The screen is arranged around the one decision a human actually makes here:
approve or reject. Everything above it exists to justify that decision.

    uv run streamlit run ui/app.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.config import get_settings  # noqa: E402
from sentinel.models.router import describe_routing  # noqa: E402
from sentinel.runner import (  # noqa: E402
    resume_with_decision,
    run_until_pause,
    start_incident,
)
from simulator.scenarios import list_scenarios  # noqa: E402
from simulator.world import load_world  # noqa: E402

st.set_page_config(page_title="Sentinel", page_icon="🛡", layout="wide")

STAGE_ICON = {
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


def init_state() -> None:
    st.session_state.setdefault("handle", None)
    st.session_state.setdefault("initial_state", None)
    st.session_state.setdefault("trace", [])
    st.session_state.setdefault("values", {})
    st.session_state.setdefault("awaiting_approval", False)
    st.session_state.setdefault("finished", False)


init_state()


# --- sidebar --------------------------------------------------------------

with st.sidebar:
    st.title("🛡 Sentinel")
    st.caption("Autonomous incident response")

    scenario = st.selectbox(
        "Incident scenario",
        list_scenarios(),
        format_func=lambda s: load_world(s).scenario.title,
        disabled=st.session_state.handle is not None and not st.session_state.finished,
    )

    col_a, col_b = st.columns(2)
    start = col_a.button("Start", type="primary", use_container_width=True)
    reset = col_b.button("Reset", use_container_width=True)

    st.divider()
    settings = get_settings()
    st.subheader("Configuration")
    st.code(describe_routing(), language=None)
    st.metric("Triage backend", settings.triage_backend)
    st.metric("Max investigation loops", settings.max_investigation_loops)
    st.caption(f"Approval required for every write action: **{settings.always_require_approval}**")

    with st.expander("Ground truth (spoiler)"):
        gt = load_world(scenario).scenario.ground_truth
        st.markdown(f"**Root cause**\n\n{gt.root_cause}")
        st.markdown(f"**Correct fix**\n\n{gt.correct_remediation}")
        st.markdown("**Red herrings**")
        for r in gt.red_herrings or ["none"]:
            st.markdown(f"- {r}")

if reset:
    for key in ("handle", "initial_state", "trace", "values", "awaiting_approval", "finished"):
        st.session_state[key] = (
            None
            if key in {"handle", "initial_state"}
            else ([] if key == "trace" else ({} if key == "values" else False))
        )
    st.rerun()


# --- rendering helpers ----------------------------------------------------


def render_update(node: str, update: dict) -> None:
    if not isinstance(update, dict):
        return
    icon = STAGE_ICON.get(node, "•")

    with st.chat_message("assistant", avatar=icon):
        st.markdown(f"**{node}**")

        if t := update.get("triage"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Severity", str(t.severity))
            c2.metric("Category", str(t.category).replace("_", " "))
            c3.metric("Needs human", "yes" if t.needs_human else "no")
            st.caption(t.reasoning)

        if ev := update.get("evidence"):
            st.caption(f"Gathered {len(ev)} observations")
            with st.expander("evidence"):
                for e in ev:
                    st.markdown(f"**{e.source}** · {e.strength}")
                    st.code(e.observation[:800], language=None)

        if h := update.get("hypothesis"):
            st.success(h.root_cause)
            c1, c2, c3 = st.columns(3)
            c1.metric("Confidence", str(h.confidence))
            c2.metric("Service", h.affected_service)
            c3.metric("Trigger", h.trigger or "none")
            if h.unknowns:
                st.warning("Unverified: " + "; ".join(h.unknowns))

        if cr := update.get("critique"):
            (st.success if cr.verdict == "accept" else st.warning)(
                f"**{cr.verdict}** — score {cr.score}/10\n\n{cr.reasoning}"
            )
            if cr.next_questions:
                st.markdown("**Open questions**")
                for q in cr.next_questions:
                    st.markdown(f"- {q}")
            if cr.alternative_causes:
                st.markdown("**Alternatives considered**")
                for a in cr.alternative_causes:
                    st.markdown(f"- {a}")

        if acts := update.get("executed_actions"):
            for a in acts:
                st.code(a, language=None)

        if v := update.get("verification"):
            (st.success if v.resolved else st.error)(
                f"Resolved: {v.resolved}\n\n" + "\n".join(f"- {o}" for o in v.observations)
            )

        if pm := update.get("postmortem"):
            st.markdown(f"### {pm.title}")
            st.markdown(f"**Impact** — {pm.impact}")
            st.markdown("**Timeline**")
            for item in pm.timeline:
                st.markdown(f"- {item}")
            st.markdown("**Action items**")
            for item in pm.action_items:
                st.markdown(f"- {item}")
            st.info(f"**Lesson stored in memory:** {pm.lesson}")

        for flag in update.get("security_flags") or []:
            st.error(f"⚠ {flag}")
        for err in update.get("errors") or []:
            st.error(err)


def render_plan(plan) -> None:
    st.subheader("Remediation plan")
    st.markdown(f"**{plan.summary}**")

    for i, step in enumerate(plan.steps, 1):
        radius_colour = {
            "low": "green",
            "medium": "orange",
            "high": "red",
            "critical": "red",
        }[step.blast_radius]
        with st.container(border=True):
            st.markdown(
                f"**{i}. `{step.action}` → `{step.target}`**  "
                f":{radius_colour}[blast radius: {step.blast_radius}]"
            )
            if step.parameters:
                st.json(step.parameters)
            st.caption(step.rationale)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Expected effect**")
        st.write(plan.expected_effect)
        st.markdown("**Rollback**")
        st.write(plan.rollback_plan)
    with c2:
        st.markdown("**Risks**")
        for r in plan.risks or ["none stated"]:
            st.write(f"- {r}")
        st.markdown("**If we do nothing**")
        st.write(plan.do_nothing_option)


# --- run ------------------------------------------------------------------


async def collect_until_pause(handle, state, sink) -> dict:
    paused: dict = {}
    async for node, update in run_until_pause(handle, state):
        if node == "__paused__":
            paused = update
        else:
            sink(node, update)
    return paused


async def collect_after_approval(handle, approved: bool, note: str, sink) -> dict:
    done: dict = {}
    async for node, update in resume_with_decision(handle, approved=approved, note=note):
        if node == "__done__":
            done = update
        else:
            sink(node, update)
    return done


st.title("Incident response")

if start:
    handle, state = start_incident(scenario)
    st.session_state.handle = handle
    st.session_state.initial_state = state
    st.session_state.trace = []
    st.session_state.awaiting_approval = False
    st.session_state.finished = False

    st.error(load_world(scenario).alert.render())
    st.caption(f"incident `{handle.incident_id}`")

    with st.status("Investigating…", expanded=True) as status:

        def sink(node, update):
            st.session_state.trace.append((node, update))
            render_update(node, update)

        paused = asyncio.run(collect_until_pause(handle, state, sink))
        status.update(label="Paused for approval", state="complete")

    st.session_state.values = paused.get("values", {})
    st.session_state.awaiting_approval = paused.get("awaiting_approval", False)
    st.rerun()

elif st.session_state.handle is not None:
    handle = st.session_state.handle
    st.error(load_world(handle.scenario).alert.render())
    st.caption(f"incident `{handle.incident_id}`")
    for node, update in st.session_state.trace:
        render_update(node, update)
else:
    st.info(
        "Pick a scenario and press **Start**. Sentinel will triage the alert, "
        "retrieve relevant runbooks, investigate with read-only tools, criticise "
        "its own hypothesis, and stop at a remediation plan for you to approve."
    )
    st.markdown("### Scenarios")
    for slug in list_scenarios():
        w = load_world(slug)
        with st.container(border=True):
            st.markdown(f"**{w.scenario.title}** · `{slug}`")
            st.caption(w.scenario.description)


# --- approval gate --------------------------------------------------------

if st.session_state.awaiting_approval and not st.session_state.finished:
    plan = st.session_state.values.get("plan")
    st.divider()
    with st.container(border=True):
        st.warning(
            "**Execution is paused.** Sentinel cannot proceed without a human "
            "decision. Nothing below has run yet."
        )
        if plan:
            render_plan(plan)

        note = st.text_input("Note (optional)", placeholder="why you approved or rejected")
        c1, c2, _ = st.columns([1, 1, 3])
        approve = c1.button("Approve & execute", type="primary", use_container_width=True)
        rejected = c2.button("Reject", use_container_width=True)

        if approve or rejected:
            with st.status("Executing…", expanded=True):

                def sink(node, update):
                    st.session_state.trace.append((node, update))
                    render_update(node, update)

                asyncio.run(
                    collect_after_approval(
                        st.session_state.handle,
                        approved=bool(approve),
                        note=note or ("approved via UI" if approve else "rejected via UI"),
                        sink=sink,
                    )
                )
            st.session_state.awaiting_approval = False
            st.session_state.finished = True
            st.rerun()


# --- run stats ------------------------------------------------------------

if st.session_state.values:
    values = st.session_state.values
    usage = values.get("token_usage", {}) or {}
    st.divider()
    st.subheader("Run statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Investigation loops", values.get("loop_count", 0))
    c2.metric("Evidence items", len(values.get("evidence") or []))
    c3.metric("Input tokens", f"{usage.get('input', 0):,}")
    c4.metric("Output tokens", f"{usage.get('output', 0):,}")

    if traj := values.get("tool_trajectory"):
        st.caption("Tool trajectory")
        st.code(" → ".join(traj), language=None)
