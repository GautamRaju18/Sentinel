"""Database bridging.

A platform conflict forced this design, and it turned out to be the portable
answer anyway.

psycopg's async mode refuses to run on Windows' default ProactorEventLoop and
demands a SelectorEventLoop. But MCP's stdio transport spawns subprocesses,
which asyncio only supports on the Proactor loop. One process cannot satisfy
both, and switching the global loop policy breaks whichever subsystem loses.

So: no async database drivers at all. Everything Postgres runs through the
ordinary synchronous driver, and `asyncio.to_thread` bridges it into async code.
Database calls are I/O-bound and release the GIL, so a thread is exactly the
right tool — this costs a thread hop and buys platform independence.

`ThreadedCheckpointer` is the adapter: LangGraph's async graph calls `aput`,
which lands on the sync `PostgresSaver` in a worker thread.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from sentinel.config import get_settings
from sentinel.logging_setup import get_logger

log = get_logger(__name__)


class ThreadedCheckpointer(BaseCheckpointSaver):
    """Async checkpointer backed by a synchronous PostgresSaver.

    Every async method delegates to its sync counterpart in a worker thread.
    """

    def __init__(self, sync_saver: BaseCheckpointSaver):
        super().__init__()
        self._saver = sync_saver

    # --- sync passthrough -------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self._saver.get_tuple(config)

    def list(self, config: RunnableConfig | None, **kwargs: Any) -> Iterator[CheckpointTuple]:
        return self._saver.list(config, **kwargs)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self._saver.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, *a: Any
    ) -> None:
        return self._saver.put_writes(config, writes, task_id, *a)

    # --- async adapters ---------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self._saver.get_tuple, config)

    async def alist(
        self, config: RunnableConfig | None, **kwargs: Any
    ) -> AsyncIterator[CheckpointTuple]:
        # The sync iterator is drained inside the thread; streaming it lazily
        # across the boundary would hold a database cursor open for the whole
        # consumption, which is worse than materialising a bounded page.
        items = await asyncio.to_thread(lambda: list(self._saver.list(config, **kwargs)))
        for item in items:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self._saver.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, *a: Any
    ) -> None:
        return await asyncio.to_thread(self._saver.put_writes, config, writes, task_id, *a)


def make_checkpointer():
    """Postgres-backed checkpointer, or an in-memory one if Postgres is down.

    Returns (checkpointer, context_manager_or_None). The caller keeps the
    context manager alive for the duration of the run.
    """
    from langgraph.checkpoint.postgres import PostgresSaver

    url = get_settings().database_url
    try:
        cm = PostgresSaver.from_conn_string(url)
        saver = cm.__enter__()
        saver.setup()
        log.info("checkpointer.postgres", url=url.split("@")[-1])
        return ThreadedCheckpointer(saver), cm
    except Exception as e:
        from langgraph.checkpoint.memory import MemorySaver

        # Losing durability is bad; refusing to run an incident response is
        # worse. Degrade loudly.
        log.error("checkpointer.postgres_failed", error=str(e))
        log.warning("checkpointer.fallback_memory", impact="checkpoints will not survive restart")
        return MemorySaver(), None


async def to_thread(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)
