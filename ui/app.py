"""Sentinel operator console — overview.

Streamlit multipage app. This is the landing page; the rest live in ui/pages/.

    uv run streamlit run ui/app.py
"""

from __future__ import annotations

import streamlit as st
from common import ROOT, hero, load_results, no_data, page_setup, sidebar_status, stat

page_setup("Overview")
sidebar_status()

hero(
    "🛡️ Sentinel",
    "An autonomous incident response agent — LangGraph orchestration, hybrid RAG, "
    "MCP tooling, a fine-tuned triage model, and a human approval gate that the "
    "agent cannot open by itself.",
)

# --- headline metrics -----------------------------------------------------

baseline = load_results("baseline")
finetuned = load_results("finetuned")
gen_base = load_results("generalization_baseline")
gen_ft = load_results("generalization_finetuned")

st.markdown("### Fine-tuned triage model")

if baseline and finetuned:
    cols = st.columns(4)
    metrics = [
        ("Severity accuracy", "severity_accuracy", True),
        ("Category accuracy", "category_accuracy", True),
        ("Critical misses", "critical_underestimate_rate", False),
        ("Latency p50", "latency_p50_ms", False),
    ]
    for col, (label, key, higher_better) in zip(cols, metrics, strict=True):
        b, f = baseline.get(key, 0), finetuned.get(key, 0)
        if key == "latency_p50_ms":
            value, delta = f"{f:.0f} ms", f"{f - b:+.0f} ms vs baseline"
        else:
            value, delta = f"{f * 100:.1f}%", f"{(f - b) * 100:+.1f} pp vs baseline"
        improved = (f > b) if higher_better else (f < b)
        col.markdown(
            stat(label, value, delta, "#22c55e" if improved else "#ef4444"),
            unsafe_allow_html=True,
        )
    st.caption("Held-out split — 120 alerts on service names that appear nowhere in training.")
else:
    no_data(
        "No triage eval results recorded yet.",
        "uv run python evals/triage_eval.py --out evals/results_finetuned.json",
    )

st.markdown("")
st.markdown("### Does it generalise?")

if gen_base and gen_ft:
    c1, c2 = st.columns(2)
    for col, label, key in [
        (c1, "Realistic unseen alerts", "scenario_category_accuracy"),
        (c2, "Adversarial alerts", "adversarial_category_accuracy"),
    ]:
        b, f = gen_base.get(key, 0), gen_ft.get(key, 0)
        delta = (f - b) * 100
        col.markdown(
            stat(
                label,
                f"{f * 100:.0f}%",
                f"{delta:+.0f} pp vs baseline ({b * 100:.0f}%)",
                "#22c55e" if delta > 0 else "#94a3b8",
            ),
            unsafe_allow_html=True,
        )

    st.warning(
        "**The honest headline.** Large gains on realistic alerts it had never seen, "
        "and *no movement at all* on the adversarial set — cases built so the surface "
        "cues point at the wrong label. The fine-tune learned to map surface features "
        "to labels far better than the baseline; it did not learn the causal reasoning "
        "those cases require. That is what template-generated training data should be "
        "expected to produce, and it is why a 100% score on the held-out split is a "
        "warning rather than a triumph."
    )
else:
    no_data(
        "Generalization eval not run yet — this is the measurement that "
        "distinguishes learning the task from memorising the templates.",
        "uv run python evals/generalization.py --out evals/results_generalization_finetuned.json",
    )

# --- what's inside --------------------------------------------------------

st.divider()
st.markdown("### What this system does")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        "**🚨 Incident Response**\n\n"
        "Triage → retrieve runbooks → investigate with read-only tools → "
        "criticise its own hypothesis → loop if weak → propose a plan → "
        "**stop for a human**."
    )
    st.page_link("pages/1_Incident_Response.py", label="Run an incident", icon="🚨")
with c2:
    st.markdown(
        "**📊 Evaluations**\n\n"
        "Held-out accuracy, out-of-distribution generalization, and an adversarial "
        "red-team suite covering prompt injection and the approval gate."
    )
    st.page_link("pages/2_Evaluations.py", label="View results", icon="📊")
with c3:
    st.markdown(
        "**🔍 Observability**\n\n"
        "LangSmith traces per node, model routing by tier, token accounting, and "
        "live service health."
    )
    st.page_link("pages/3_Observability.py", label="Inspect traces", icon="🔍")

c4, c5, _ = st.columns(3)
with c4:
    st.markdown(
        "**📚 Knowledge**\n\n"
        "Hybrid retrieval over the runbook corpus — BM25 and vector search fused "
        "with reciprocal rank fusion. Try a query against it."
    )
    st.page_link("pages/4_Knowledge.py", label="Search runbooks", icon="📚")
with c5:
    st.markdown(
        "**⚙️ System**\n\n"
        "Graph topology, guardrail configuration, MCP servers, and credential "
        "health — without printing a single secret."
    )
    st.page_link("pages/5_System.py", label="Inspect system", icon="⚙️")

c6, _, _ = st.columns(3)
with c6:
    st.markdown(
        "**📦 Repositories**\n\n"
        "The same idea on your real GitHub repos: health at a glance, alerts for "
        "what needs attention, and CI-failure investigation by the agent."
    )
    st.page_link("pages/6_Repositories.py", label="Monitor repos", icon="📦")

# --- the safety argument, stated plainly ---------------------------------

st.divider()
st.markdown("### Why you can leave this running")

s1, s2 = st.columns(2)
with s1:
    st.markdown(
        "**Guardrails are structural, not advisory.**\n\n"
        "The investigator is handed a read-only tool list, so it *cannot* call a "
        "destructive tool regardless of what it concludes. Write tools check an "
        "approval flag held in module state that no prompt can set. Confidence is "
        "not authorization."
    )
with s2:
    st.markdown(
        "**Tool output is untrusted input.**\n\n"
        "Anyone who can write a log line can write text the agent will read — a "
        "customer-supplied username is enough. Every tool result passes through "
        "injection neutralisation, secret redaction and truncation before it "
        "enters a prompt."
    )

st.caption(
    f"Repository: {ROOT} · "
    "[github.com/GautamRaju18/Sentinel](https://github.com/GautamRaju18/Sentinel)"
)
