"""Run an incident end to end, pausing at the human approval gate.

Runs the graph in-process rather than through the HTTP API. For a single
operator that removes an entire class of failure (SSE reconnects, partial JSON,
API process lifecycle) at no loss of fidelity — the API exists for programmatic
clients and has its own tests.

The screen is arranged around the one decision a human actually makes here:
approve or reject. Everything above it exists to justify that decision.
"""

from __future__ import annotations

import asyncio

import streamlit as st
from common import BLAST_COLOUR, SEVERITY_COLOUR, STAGE_ICON, hero, page_setup, pill, sidebar_status

from sentinel.runner import resume_with_decision, run_until_pause, start_incident
from simulator.scenarios import list_scenarios
from simulator.world import load_world

page_setup("Incident Response", "🚨")
sidebar_status()

hero("🚨 Incident Response", "Triage, investigate, criticise, plan — then stop for a human.")

for key, default in [
    ("handle", None),
    ("trace", []),
    ("run_values", {}),
    ("awaiting", False),
    ("finished", False),
]:
    st.session_state.setdefault(key, default)


# --- controls -------------------------------------------------------------

running = st.session_state.handle is not None and not st.session_state.finished

c1, c2, c3 = st.columns([3, 1, 1])
scenario = c1.selectbox(
    "Scenario",
    list_scenarios(),
    format_func=lambda s: f"{load_world(s).scenario.title}  ·  {s}",
    disabled=running,
    label_visibility="collapsed",
)
start = c2.button("▶ Start", type="primary", use_container_width=True, disabled=running)
if c3.button("↺ Reset", use_container_width=True):
    for k, v in [
        ("handle", None),
        ("trace", []),
        ("run_values", {}),
        ("awaiting", False),
        ("finished", False),
    ]:
        st.session_state[k] = v
    st.rerun()

world = load_world(scenario)
with st.expander("Incident brief", expanded=not st.session_state.handle):
    st.caption(world.scenario.description)
    st.code(world.alert.render(), language=None)
with st.expander("🔒 Ground truth (spoiler)"):
    gt = world.scenario.ground_truth
    st.markdown(f"**Root cause** — {gt.root_cause}")
    st.markdown(f"**Correct fix** — {gt.correct_remediation}")
    if gt.red_herrings:
        st.markdown("**Red herrings**")
        for r in gt.red_herrings:
            st.markdown(f"- {r}")


# --- rendering ------------------------------------------------------------


def render(node: str, update: dict) -> None:
    if not isinstance(update, dict):
        return
    with st.chat_message("assistant", avatar=STAGE_ICON.get(node, "•")):
        st.markdown(f"**{node}**")

        if t := update.get("triage"):
            a, b, c = st.columns(3)
            a.markdown(
                f"Severity {pill(str(t.severity), SEVERITY_COLOUR.get(str(t.severity), '#888'))}",
                unsafe_allow_html=True,
            )
            b.markdown(f"Category `{t.category}`")
            c.markdown(f"Needs human: **{'yes' if t.needs_human else 'no'}**")
            st.caption(t.reasoning)

        if ev := update.get("evidence"):
            with st.expander(f"{len(ev)} observations gathered"):
                for e in ev:
                    st.markdown(f"**{e.source}** · _{e.strength}_")
                    st.code(e.observation[:700], language=None)

        if h := update.get("hypothesis"):
            st.success(h.root_cause)
            a, b, c = st.columns(3)
            a.metric("Confidence", str(h.confidence))
            b.metric("Service", h.affected_service)
            c.metric("Trigger", h.trigger or "none")
            if h.unknowns:
                st.warning("Still unverified: " + "; ".join(h.unknowns))

        if cr := update.get("critique"):
            (st.success if cr.verdict == "accept" else st.warning)(
                f"**{cr.verdict}** · {cr.score}/10 — {cr.reasoning}"
            )
            if cr.next_questions:
                st.markdown("**Open questions driving another round**")
                for q in cr.next_questions:
                    st.markdown(f"- {q}")
            if cr.alternative_causes:
                st.caption("Alternatives considered: " + "; ".join(cr.alternative_causes))

        for a in update.get("executed_actions") or []:
            st.code(a, language=None)

        if v := update.get("verification"):
            (st.success if v.resolved else st.error)(f"Resolved: **{v.resolved}**")
            for o in v.observations:
                st.markdown(f"- {o}")

        if pm := update.get("postmortem"):
            st.markdown(f"#### {pm.title}")
            st.markdown(f"**Impact** — {pm.impact}")
            if pm.timeline:
                st.markdown("**Timeline**")
                for item in pm.timeline:
                    st.markdown(f"- {item}")
            st.markdown("**Action items**")
            for item in pm.action_items:
                st.markdown(f"- {item}")
            st.info(f"**Lesson written to long-term memory:** {pm.lesson}")

        for flag in update.get("security_flags") or []:
            st.error(f"⚠️ {flag}")
        for err in update.get("errors") or []:
            st.error(err)


async def _until_pause(handle, state, sink):
    paused = {}
    async for node, update in run_until_pause(handle, state):
        if node == "__paused__":
            paused = update
        else:
            sink(node, update)
    return paused


async def _after_approval(handle, approved, note, sink):
    async for node, update in resume_with_decision(handle, approved=approved, note=note):
        if not node.startswith("__"):
            sink(node, update)


# --- run ------------------------------------------------------------------

if start:
    handle, state = start_incident(scenario)
    st.session_state.handle = handle
    st.session_state.trace = []
    st.session_state.awaiting = False
    st.session_state.finished = False

    def sink(node, update):
        st.session_state.trace.append((node, update))
        render(node, update)

    with st.status("Investigating…", expanded=True) as status:
        paused = asyncio.run(_until_pause(handle, state, sink))
        status.update(label="Paused — awaiting your decision", state="complete")

    # NOTE: the runner's payload key is "values"; the session_state key is
    # "run_values" because `st.session_state.values` resolves to the mapping's
    # own .values() method rather than to stored data.
    st.session_state.run_values = paused.get("values", {})
    st.session_state.awaiting = paused.get("awaiting_approval", False)
    st.rerun()

elif st.session_state.handle:
    st.caption(f"incident `{st.session_state.handle.incident_id}`")
    for node, update in st.session_state.trace:
        render(node, update)


# --- approval gate --------------------------------------------------------

if st.session_state.awaiting and not st.session_state.finished:
    plan = st.session_state.run_values.get("plan")
    st.divider()
    st.markdown('<div class="gate">', unsafe_allow_html=True)
    st.markdown("### 🔐 Execution is paused")
    st.caption(
        "Sentinel cannot proceed without a human decision. Nothing below has run. "
        "The agent has no way to grant itself this approval."
    )

    if plan:
        st.markdown(f"**{plan.summary}**")
        for i, step in enumerate(plan.steps, 1):
            with st.container(border=True):
                st.markdown(
                    f"**{i}. `{step.action}` → `{step.target}`** &nbsp; "
                    + pill(step.blast_radius, BLAST_COLOUR[step.blast_radius]),
                    unsafe_allow_html=True,
                )
                if step.parameters:
                    st.json(step.parameters, expanded=False)
                st.caption(step.rationale)

        a, b = st.columns(2)
        with a:
            st.markdown("**Expected effect**")
            st.write(plan.expected_effect)
            st.markdown("**Rollback**")
            st.write(plan.rollback_plan)
        with b:
            st.markdown("**Risks**")
            for r in plan.risks or ["none stated"]:
                st.write(f"- {r}")
            st.markdown("**If we do nothing**")
            st.write(plan.do_nothing_option)

    note = st.text_input("Note (optional)", placeholder="why you approved or rejected")
    a, b, _ = st.columns([1, 1, 3])
    approve = a.button("✅ Approve & execute", type="primary", use_container_width=True)
    reject = b.button("✕ Reject", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if approve or reject:

        def sink(node, update):
            st.session_state.trace.append((node, update))
            render(node, update)

        with st.status("Executing…", expanded=True):
            asyncio.run(
                _after_approval(
                    st.session_state.handle,
                    bool(approve),
                    note or ("approved via UI" if approve else "rejected via UI"),
                    sink,
                )
            )
        st.session_state.awaiting = False
        st.session_state.finished = True
        st.rerun()


# --- run stats ------------------------------------------------------------

if st.session_state.run_values:
    v = st.session_state.run_values
    usage = v.get("token_usage", {}) or {}
    st.divider()
    a, b, c, d = st.columns(4)
    a.metric("Investigation loops", v.get("loop_count", 0))
    b.metric("Evidence items", len(v.get("evidence") or []))
    c.metric("Input tokens", f"{usage.get('input', 0):,}")
    d.metric("Output tokens", f"{usage.get('output', 0):,}")
    if traj := v.get("tool_trajectory"):
        st.caption("Tool trajectory")
        st.code(" → ".join(traj), language=None)
