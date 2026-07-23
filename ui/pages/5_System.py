"""Graph topology, guardrail configuration, MCP servers, credential health."""

from __future__ import annotations

import hashlib
import subprocess
import sys

import pandas as pd
import streamlit as st
from common import BLAST_COLOUR, ROOT, hero, page_setup, pill, sidebar_status

from sentinel.config import get_settings
from sentinel.graph import render_mermaid
from sentinel.tools import get_tools
from sentinel.tools.guardrails import ACTION_BLAST_RADIUS, authorize

page_setup("System", "⚙️")
sidebar_status()

hero("⚙️ System", "Topology, guardrails, tooling and credential health.")

settings = get_settings()
tab1, tab2, tab3, tab4 = st.tabs(["Graph", "Guardrails", "Tools & MCP", "Credentials"])

# --- graph ----------------------------------------------------------------

with tab1:
    st.caption(
        "The entire control flow in one screen — which is the argument for LangGraph "
        "over a hand-rolled loop. The cycle, the branch and the interrupt are "
        "declared, not buried in conditionals."
    )
    st.markdown(f"```mermaid\n{render_mermaid()}\n```")

    with st.expander("Mermaid source"):
        st.code(render_mermaid(), language="text")

    st.markdown("#### The two structural features")
    a, b = st.columns(2)
    a.info(
        "**The cycle.** `critique` can route back to `investigate` when the "
        "hypothesis is weak, carrying specific questions to answer. Bounded by "
        f"`MAX_INVESTIGATION_LOOPS={settings.max_investigation_loops}` — an agent "
        "that is confused does not become less confused by iterating."
    )
    b.warning(
        "**The interrupt.** Execution halts *before* the `approval` node. The "
        "decision is written from outside the graph, so no node and no model can "
        "produce it. That is what makes the gate meaningful rather than decorative."
    )

# --- guardrails -----------------------------------------------------------

with tab2:
    st.markdown("#### Blast radius by action")
    st.caption(
        "Anything not classified is treated as critical — fail closed. An action "
        "nobody thought about is the most dangerous kind."
    )

    rows = []
    for action, radius in sorted(ACTION_BLAST_RADIUS.items(), key=lambda kv: kv[0]):
        decision, reason = authorize(action)
        rows.append(
            {
                "Action": action,
                "Blast radius": radius,
                "Unapproved": {"allow": "✅ runs", "require_approval": "🔐 blocked"}.get(
                    decision, decision
                ),
                "Why": reason,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Active configuration")
    a, b, c = st.columns(3)
    a.metric("Always require approval", str(settings.always_require_approval))
    b.markdown(
        f"Max auto blast radius<br>"
        f"{pill(settings.max_auto_blast_radius, BLAST_COLOUR[settings.max_auto_blast_radius])}",
        unsafe_allow_html=True,
    )
    c.metric("Tool timeout", f"{settings.tool_timeout_seconds:.0f}s")

    st.markdown("#### Try the gate")
    st.caption("Invoke a destructive tool directly. It should refuse.")
    if st.button("🔓 Attempt rollback without approval"):
        import asyncio

        from sentinel.tools import get_tool

        result = asyncio.run(
            get_tool("rollback_deploy").ainvoke(
                {"deploy_id": "dpl-8814", "reason": "testing the gate from the UI"}
            )
        )
        (st.success if str(result).startswith("BLOCKED") else st.error)(str(result))

# --- tools & mcp ----------------------------------------------------------

with tab3:
    st.markdown("#### Tool inventory")
    st.caption(
        "The investigator is handed the read-only list. It *cannot* call a "
        "destructive tool regardless of what it decides — a stronger guarantee than "
        "asking it nicely."
    )

    read_only = {t.name for t in get_tools(read_only=True)}
    rows = [
        {
            "Tool": t.name,
            "Investigator can call": "✅" if t.name in read_only else "—",
            "Blast radius": ACTION_BLAST_RADIUS.get(t.name, "critical"),
            "Description": (t.description or "").split("\n")[0][:80],
        }
        for t in get_tools()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### MCP servers")
    st.caption(
        "Sentinel publishes its own tools over MCP, and consumes tools published by "
        "others. Capabilities become a deployment concern rather than a code one — "
        "adding filesystem access is an edit to `mcp.config.json`, not a new module."
    )
    try:
        from sentinel.mcp_client import describe_servers

        st.code(describe_servers(), language=None)
    except Exception as e:
        st.warning(f"Could not read MCP config: {type(e).__name__}: {e}")

    with st.expander("What Sentinel publishes over MCP"):
        st.markdown(
            "**Tools** — query_logs, get_metric, list_metrics, get_deploys, "
            "get_service_health, propose_remediation\n\n"
            "**Resources** — `incident://current/alert`, `incident://current/summary`, "
            "`incident://scenarios`\n\n"
            "**Prompts** — investigate, postmortem\n\n"
            "Remediation is exposed only as `propose_remediation`, which records a "
            "proposal and executes nothing. An external MCP client has no way to "
            "prove a human approved anything, so execution stays inside the graph."
        )

# --- credentials ----------------------------------------------------------

with tab4:
    st.caption(
        "Values are identified by a sha256 prefix, never a prefix of the key itself. "
        "A key prefix leaks real material into screenshots and logs; a hash still "
        "lets you confirm a value actually changed after a rotation."
    )

    def fp(secret: str) -> str:
        return hashlib.sha256(secret.encode()).hexdigest()[:8] if secret else "—"

    rows = [
        ("OpenRouter", settings.openrouter_api_key, "planner / reasoner tiers"),
        ("GitHub", settings.github_token, "MCP GitHub server, pushes"),
        ("HuggingFace", settings.huggingface_token, "only needed to retrain"),
        ("LangSmith", settings.langsmith_api_key, "tracing"),
    ]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Service": name,
                    "Configured": "✅" if secret else "—",
                    "Fingerprint": fp(secret),
                    "Used for": use,
                }
                for name, secret, use in rows
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    if st.button("▶ Run live credential check"):
        with st.spinner("Calling each service…"):
            proc = subprocess.run(
                [sys.executable, "scripts/check_credentials.py"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        (st.success if proc.returncode == 0 else st.error)(
            "All configured credentials valid"
            if proc.returncode == 0
            else "Some credentials failed"
        )
        st.code(proc.stdout or proc.stderr, language=None)

    st.markdown("#### Model configuration")
    st.json(
        {
            "triage_backend": settings.triage_backend,
            "model_triage": settings.model_triage,
            "model_planner": settings.model_planner,
            "model_reasoner": settings.model_reasoner,
            "model_worker": settings.model_worker,
            "model_embed": settings.model_embed,
        }
    )
