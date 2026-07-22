"""Vector store and ingestion.

Chunking choice: runbooks are split on markdown headings rather than by a fixed
character count. A runbook section is already a coherent unit of advice, and
splitting "## Fix" away from its diagnostic context produces chunks that
retrieve well but read uselessly.

Each chunk carries its document title and heading path in the text itself —
"contextual retrieval". Retrieval quality improves measurably because the
embedding of a chunk that says "Runbook: Database connection errors > Fix"
is far more specific than one that opens mid-sentence.
"""

from __future__ import annotations

import re
from functools import lru_cache

from langchain_core.documents import Document

from sentinel.config import PROJECT_ROOT, get_settings
from sentinel.logging_setup import get_logger
from sentinel.models.router import get_embeddings

log = get_logger(__name__)

RUNBOOK_DIR = PROJECT_ROOT / "data" / "runbooks"
RUNBOOK_COLLECTION = "runbooks"
INCIDENT_COLLECTION = "incidents"

_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


def chunk_markdown(text: str, *, source: str, min_chars: int = 120) -> list[Document]:
    """Split on headings, keeping the heading path attached to each chunk."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [Document(page_content=text, metadata={"source": source, "section": ""})]

    title = matches[0].group(2) if matches[0].group(1) == "#" else source
    chunks: list[Document] = []

    for i, m in enumerate(matches):
        level, heading = len(m.group(1)), m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) < min_chars:
            continue
        # Prefixing the breadcrumb is what makes the embedding specific.
        content = f"{title} > {heading}\n\n{body}" if level > 1 else f"{title}\n\n{body}"
        chunks.append(
            Document(
                page_content=content,
                metadata={"source": source, "title": title, "section": heading, "level": level},
            )
        )
    return chunks


def load_runbook_documents() -> list[Document]:
    docs: list[Document] = []
    for path in sorted(RUNBOOK_DIR.glob("*.md")):
        docs.extend(chunk_markdown(path.read_text(encoding="utf-8"), source=path.stem))
    log.info("rag.loaded_runbooks", files=len(list(RUNBOOK_DIR.glob("*.md"))), chunks=len(docs))
    return docs


@lru_cache(maxsize=4)
def get_vector_store(collection: str = RUNBOOK_COLLECTION):
    """Synchronous PGVector. Async callers go through sentinel.db.to_thread.

    Sync on purpose — see the module docstring in sentinel/db.py. psycopg's
    async mode cannot coexist with MCP's subprocess transport on Windows.
    """
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection,
        connection=get_settings().async_database_url,
        use_jsonb=True,
        create_extension=False,  # docker/init-db.sql already created it
    )


async def ingest_runbooks(*, reset: bool = True) -> int:
    """Embed the runbook corpus into pgvector. Idempotent when reset=True."""
    from sentinel.db import to_thread

    docs = load_runbook_documents()
    store = get_vector_store(RUNBOOK_COLLECTION)
    if reset:
        try:
            await to_thread(store.delete_collection)
            await to_thread(store.create_collection)
        except Exception as e:
            log.warning("rag.reset_failed", error=str(e))
    await to_thread(store.add_documents, docs)
    log.info("rag.ingested", chunks=len(docs), collection=RUNBOOK_COLLECTION)
    return len(docs)


def runbook_texts() -> list[tuple[str, dict]]:
    """Raw corpus for the BM25 half of hybrid retrieval."""
    return [(d.page_content, d.metadata) for d in load_runbook_documents()]
