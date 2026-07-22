"""Hybrid retrieval: BM25 + vector, fused with Reciprocal Rank Fusion.

Neither half is sufficient alone in this domain:

  * Vector search handles paraphrase — an alert saying "pods terminating with
    exit 137" should find a runbook section headed "Memory exhaustion", which
    shares no words with the query.
  * BM25 handles the exact rare tokens that matter most here — `x509`,
    `HikariPool-1`, `MISCONF`, `dpl-8814`. Embeddings blur precisely these,
    because they are near-unique strings with little semantic content.

RRF is used rather than score normalisation because BM25 scores and cosine
similarities are not on comparable scales, and normalising them requires
per-corpus tuning that will silently rot. Rank fusion needs no tuning.
"""

from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.documents import Document

from sentinel.graph.schemas import Triage
from sentinel.logging_setup import get_logger
from sentinel.rag.store import (
    INCIDENT_COLLECTION,
    RUNBOOK_COLLECTION,
    get_vector_store,
    runbook_texts,
)

log = get_logger(__name__)

_TOKEN = re.compile(r"[a-zA-Z0-9_\-]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


@lru_cache(maxsize=1)
def _bm25_index():
    from rank_bm25 import BM25Okapi

    corpus = runbook_texts()
    tokenized = [_tokenize(text) for text, _ in corpus]
    return BM25Okapi(tokenized), corpus


def bm25_search(query: str, k: int = 6) -> list[Document]:
    index, corpus = _bm25_index()
    scores = index.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        Document(page_content=corpus[i][0], metadata={**corpus[i][1], "bm25_score": scores[i]})
        for i in ranked
        if scores[i] > 0
    ]


async def vector_search(query: str, k: int = 6, collection: str = RUNBOOK_COLLECTION):
    from sentinel.db import to_thread

    try:
        store = get_vector_store(collection)
        return await to_thread(store.similarity_search, query, k)
    except Exception as e:
        # Retrieval must never break an investigation. BM25 alone still works.
        log.warning("rag.vector_search_failed", error=str(e), collection=collection)
        return []


def reciprocal_rank_fusion(
    rankings: list[list[Document]], *, k: int = 60, top_n: int = 6
) -> list[Document]:
    """Fuse ranked lists. Score = sum over lists of 1/(k + rank)."""
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = doc.page_content[:200]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            docs.setdefault(key, doc)
    best = sorted(scores, key=lambda key: scores[key], reverse=True)[:top_n]
    return [docs[key] for key in best]


async def hybrid_search(query: str, *, top_n: int = 5) -> list[Document]:
    lexical = bm25_search(query, k=8)
    semantic = await vector_search(query, k=8)
    fused = reciprocal_rank_fusion([lexical, semantic], top_n=top_n)
    log.info("rag.hybrid", lexical=len(lexical), semantic=len(semantic), fused=len(fused))
    return fused


def _render(docs: list[Document], limit: int = 4000) -> str:
    out, total = [], 0
    for d in docs:
        section = d.metadata.get("section", "")
        header = f"— {d.metadata.get('source', '?')}{f' § {section}' if section else ''}"
        block = f"{header}\n{d.page_content}"
        if total + len(block) > limit:
            break
        out.append(block)
        total += len(block)
    return "\n\n".join(out)


def expand_query(alert: str, triage: Triage | None) -> str:
    """Add classification terms so retrieval keys on the failure class too.

    The alert text alone is often symptom-only ("error rate spike"), which
    retrieves poorly. The triage category names the failure class directly.
    """
    parts = [alert[:1200]]
    if triage:
        parts.append(f"failure category: {triage.category}")
        if triage.affected_service:
            parts.append(f"service: {triage.affected_service}")
    return "\n".join(parts)


async def retrieve_context(alert: str, triage: Triage | None = None) -> tuple[str, str]:
    """Return (runbook guidance, similar past incidents) for the investigator."""
    query = expand_query(alert, triage)
    runbook_docs = await hybrid_search(query, top_n=4)
    incident_docs = await vector_search(query, k=3, collection=INCIDENT_COLLECTION)
    return _render(runbook_docs), _render(incident_docs, limit=2000)
