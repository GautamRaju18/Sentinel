"""Semantic cache for alerts.

Production alerting is repetitive: the same flapping check fires fifty times an
hour with a different timestamp each time. Exact-match caching never hits;
embedding similarity does.

The tradeoff is real and worth stating. A cache hit skips an entire
investigation, which is a large saving and a real risk: two alerts can look
similar and have different causes. Mitigations here are (a) a deliberately high
similarity threshold, (b) never caching across different services, and (c)
storing the triage classification rather than the remediation — reusing a
*classification* is far safer than reusing a *plan to change production*.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from sentinel.graph.schemas import Triage
from sentinel.logging_setup import get_logger
from sentinel.models.router import get_embeddings

log = get_logger(__name__)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class CacheEntry:
    text: str
    embedding: list[float]
    triage: Triage
    stored_at: float = field(default_factory=time.time)
    hits: int = 0


class SemanticCache:
    def __init__(self, *, threshold: float = 0.94, ttl_seconds: float = 1800, max_size: int = 500):
        # 0.94 is high on purpose. At 0.85 this cache confidently returns the
        # wrong classification for genuinely different alerts.
        self.threshold = threshold
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._entries: list[CacheEntry] = []
        self.lookups = 0
        self.hit_count = 0

    def _evict(self) -> None:
        now = time.time()
        self._entries = [e for e in self._entries if now - e.stored_at < self.ttl]
        if len(self._entries) > self.max_size:
            self._entries.sort(key=lambda e: (e.hits, e.stored_at), reverse=True)
            self._entries = self._entries[: self.max_size]

    async def get(self, alert: str, *, service: str | None = None) -> Triage | None:
        self.lookups += 1
        self._evict()
        if not self._entries:
            return None
        try:
            query = await get_embeddings().aembed_query(alert)
        except Exception as e:
            log.warning("cache.embed_failed", error=str(e))
            return None

        best, best_score = None, 0.0
        for entry in self._entries:
            # Never match across services — same symptom, different system.
            if (
                service
                and entry.triage.affected_service
                and entry.triage.affected_service != service
            ):
                continue
            score = cosine(query, entry.embedding)
            if score > best_score:
                best, best_score = entry, score

        if best and best_score >= self.threshold:
            best.hits += 1
            self.hit_count += 1
            log.info("cache.hit", score=round(best_score, 4), hits=best.hits)
            return best.triage
        log.debug("cache.miss", best_score=round(best_score, 4))
        return None

    async def put(self, alert: str, triage: Triage) -> None:
        try:
            embedding = await get_embeddings().aembed_query(alert)
        except Exception as e:
            log.warning("cache.embed_failed", error=str(e))
            return
        self._entries.append(CacheEntry(text=alert, embedding=embedding, triage=triage))
        self._evict()

    @property
    def hit_rate(self) -> float:
        return self.hit_count / self.lookups if self.lookups else 0.0

    def stats(self) -> dict[str, float | int]:
        return {
            "entries": len(self._entries),
            "lookups": self.lookups,
            "hits": self.hit_count,
            "hit_rate": round(self.hit_rate, 3),
        }


_cache: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
