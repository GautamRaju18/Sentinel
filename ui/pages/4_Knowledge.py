"""The runbook corpus, the hybrid retriever, and incident memory."""

from __future__ import annotations

import asyncio

import pandas as pd
import streamlit as st
from common import ROOT, hero, page_setup, sidebar_status

from sentinel.rag.retriever import bm25_search, hybrid_search, vector_search
from sentinel.rag.store import RUNBOOK_DIR, load_runbook_documents

page_setup("Knowledge", "📚")
sidebar_status()

hero(
    "📚 Knowledge base",
    "Hybrid retrieval over the runbook corpus. Neither half is sufficient alone: "
    "vector search finds 'memory exhaustion' from 'exit code 137', while BM25 finds "
    "the rare exact tokens — x509, HikariPool-1, MISCONF — that embeddings blur.",
)

tab1, tab2, tab3 = st.tabs(["Retrieval playground", "Corpus", "Incident memory"])

# --- playground -----------------------------------------------------------

with tab1:
    st.caption(
        "Compare the two halves against the fusion. Reciprocal rank fusion is used "
        "rather than score normalisation because BM25 scores and cosine similarities "
        "are not on comparable scales, and normalising them needs per-corpus tuning "
        "that will silently rot."
    )

    examples = [
        "pods terminating with exit code 137 OOMKilled",
        "x509 certificate has expired, all logins failing",
        "HikariPool connection not available, pool exhausted",
        "three services degraded at once, no deploys in 6 hours",
        "latency spiked right after a deploy, request rate flat",
    ]
    pick = st.selectbox("Example queries", ["— custom —"] + examples)
    query = st.text_input(
        "Query",
        value="" if pick == "— custom —" else pick,
        placeholder="describe a symptom the way an alert would",
    )

    top_n = st.slider("Results", 3, 8, 4)

    if query:
        with st.spinner("Retrieving…"):
            lexical = bm25_search(query, k=top_n)
            semantic = asyncio.run(vector_search(query, k=top_n))
            fused = asyncio.run(hybrid_search(query, top_n=top_n))

        a, b, c = st.columns(3)
        for col, title, docs, note in [
            (a, "BM25 (lexical)", lexical, "exact rare tokens"),
            (b, "Vector (semantic)", semantic, "paraphrase"),
            (c, "**Fused (RRF)**", fused, "what the agent sees"),
        ]:
            with col:
                st.markdown(f"**{title}**")
                st.caption(note)
                if not docs:
                    st.caption("_no results_")
                for d in docs:
                    st.markdown(
                        f"`{d.metadata.get('source', '?')}`  \n"
                        f"<small>§ {d.metadata.get('section', '')}</small>",
                        unsafe_allow_html=True,
                    )

        with st.expander("Full text of the fused results"):
            for d in fused:
                st.markdown(f"**{d.metadata.get('source')}** § {d.metadata.get('section', '')}")
                st.code(d.page_content[:1200], language=None)

# --- corpus ---------------------------------------------------------------

with tab2:
    docs = load_runbook_documents()
    files = sorted(RUNBOOK_DIR.glob("*.md"))

    a, b = st.columns(2)
    a.metric("Runbooks", len(files))
    b.metric("Chunks", len(docs))

    st.caption(
        "Chunked on markdown headings, not a fixed character count — a runbook "
        "section is already a coherent unit of advice, and splitting '## Fix' away "
        "from its diagnostic context produces chunks that retrieve well and read "
        "uselessly. Each chunk carries its heading path, which makes the embedding "
        "far more specific."
    )

    rows = [
        {
            "Source": d.metadata.get("source"),
            "Section": d.metadata.get("section"),
            "Chars": len(d.page_content),
        }
        for d in docs
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=320)

    choice = st.selectbox("Read a runbook", [f.stem for f in files])
    if choice:
        st.markdown((RUNBOOK_DIR / f"{choice}.md").read_text(encoding="utf-8"))

# --- incident memory ------------------------------------------------------

with tab3:
    st.caption(
        "Every completed incident writes a distilled post-mortem into pgvector, and "
        "future investigations retrieve it — so the system's second encounter with a "
        "class of failure starts from its first. What gets stored is the *shape* of "
        "the incident, not the raw transcript: transcripts are long, mostly noise, "
        "and retrieve badly."
    )

    q = st.text_input("Search past incidents", placeholder="e.g. connection pool errors")
    if q:
        from sentinel.memory.incidents import search_incidents

        with st.spinner("Searching memory…"):
            hits = asyncio.run(search_incidents(q, k=4))
        if not hits:
            st.info(
                "No incidents in memory yet — memory fills as you complete runs "
                "through to the post-mortem step."
            )
            st.page_link("pages/1_Incident_Response.py", label="Run an incident", icon="🚨")
        for h in hits:
            with st.container(border=True):
                meta = h.metadata
                st.markdown(
                    f"**{meta.get('scenario', '?')}** · {meta.get('category', '?')} · "
                    f"{meta.get('severity', '?')} · resolved={meta.get('resolved')}"
                )
                st.code(h.page_content[:900], language=None)

st.caption(f"Corpus: {RUNBOOK_DIR.relative_to(ROOT)}")
