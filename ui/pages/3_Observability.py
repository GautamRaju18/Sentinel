"""LangSmith tracing, model routing, and live service health."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from common import hero, page_setup, sidebar_status

from sentinel.config import get_settings
from sentinel.models.router import describe_routing
from sentinel.observability import configure_tracing, trace_url, verify_tracing

page_setup("Observability", "🔍")
sidebar_status()

hero(
    "🔍 Observability",
    "The printed run summary is a summary; the trace is the evidence. When the "
    "critic rejects a hypothesis three times and the run burns 130k tokens, the "
    "question is which node, on which loop, with what prompt — and that is a "
    "question about a tree, not a log.",
)

settings = get_settings()
configure_tracing()
ok, detail = verify_tracing()

# --- tracing status -------------------------------------------------------

c1, c2 = st.columns([2, 1])
with c1:
    if ok:
        st.success(f"**Tracing live** — {detail}")
        st.link_button("Open LangSmith dashboard ↗", trace_url(), type="primary")
    else:
        st.warning(f"**Tracing off** — {detail}")
        st.caption("Enable it in `.env`:")
        st.code("LANGSMITH_TRACING=true\nLANGSMITH_API_KEY=...", language="bash")
        st.caption(
            "Note: setting the flag alone is not enough. pydantic-settings loads "
            "`.env` into a settings object, but LangChain's tracer reads `os.environ` "
            "directly — `sentinel/observability.py` bridges the two, which is why "
            "this page calls `configure_tracing()` before checking."
        )
with c2:
    st.metric("Project", settings.langsmith_project)

# --- recent traces --------------------------------------------------------

if ok:
    st.markdown("### Recent traces")
    limit = st.slider("How many", 5, 50, 15, step=5)
    try:
        from langsmith import Client

        runs = list(
            Client(api_key=settings.langsmith_api_key).list_runs(
                project_name=settings.langsmith_project, limit=limit
            )
        )
    except Exception as e:
        runs = []
        st.error(f"Could not list runs: {type(e).__name__}: {e}")

    if not runs:
        st.info("No traces recorded yet. Run an incident to generate some.")
        st.page_link("pages/1_Incident_Response.py", label="Run an incident", icon="🚨")
    else:
        rows = []
        total_tokens = 0
        for r in runs:
            tokens = getattr(r, "total_tokens", None) or 0
            total_tokens += tokens
            latency = ""
            if r.end_time and r.start_time:
                latency = f"{(r.end_time - r.start_time).total_seconds():.1f}s"
            rows.append(
                {
                    "Name": str(r.name)[:36],
                    "Type": str(r.run_type),
                    "Started": str(r.start_time)[:19],
                    "Latency": latency,
                    "Tokens": tokens or "—",
                    "Error": "❌" if r.error else "",
                }
            )

        a, b, c = st.columns(3)
        a.metric("Traces shown", len(runs))
        b.metric("Total tokens", f"{total_tokens:,}")
        c.metric("With errors", sum(1 for r in runs if r.error))

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- routing --------------------------------------------------------------

st.divider()
st.markdown("### Model routing")
st.caption(
    "Nodes ask for a *tier*, not a model name. `.env` decides what that resolves to. "
    "That seam is what makes the local-vs-hosted comparison and the fine-tuning "
    "result measurable rather than anecdotal."
)
st.code(describe_routing(), language=None)

with st.expander("Why tiers"):
    st.markdown(
        "- **planner** — hardest reasoning: root-cause synthesis, remediation plans\n"
        "- **reasoner** — evidence synthesis, critique, summarisation\n"
        "- **worker** — cheap and local: tool-heavy loops, extraction\n"
        "- **triage** — classification only; this is the tier the fine-tuned model replaced\n\n"
        "Hosted calls carry a fallback chain, because free-tier providers return "
        "429/503 constantly and a single saturated model should not fail a run."
    )

# --- service health -------------------------------------------------------

st.divider()
st.markdown("### Service health")

if st.button("Check services"):
    import httpx

    checks = []

    try:
        r = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=10)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        checks.append(("Ollama", "✅", f"{len(names)} models: {', '.join(names[:3])}"))
    except Exception as e:
        checks.append(("Ollama", "❌", f"{type(e).__name__} at {settings.ollama_base_url}"))

    try:
        import psycopg

        with psycopg.connect(settings.database_url, connect_timeout=8) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT extname FROM pg_extension WHERE extname='vector'")
                vec = cur.fetchone() is not None
        checks.append(("Postgres", "✅", f"connected · pgvector={'yes' if vec else 'NO'}"))
    except Exception as e:
        checks.append(("Postgres", "❌", f"{type(e).__name__}: {str(e)[:70]}"))

    if settings.openrouter_api_key:
        try:
            r = httpx.get(
                "https://openrouter.ai/api/v1/key",
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
                timeout=20,
            )
            r.raise_for_status()
            d = r.json().get("data", {})
            checks.append(("OpenRouter", "✅", f"valid · usage={d.get('usage')}"))
        except Exception as e:
            checks.append(("OpenRouter", "❌", f"{type(e).__name__}"))
    else:
        checks.append(("OpenRouter", "—", "no key set"))

    st.dataframe(
        pd.DataFrame(checks, columns=["Service", "Status", "Detail"]),
        use_container_width=True,
        hide_index=True,
    )
