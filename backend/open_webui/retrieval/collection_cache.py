"""
In-memory cache of whole vector-DB collections, shared across conversations.

Hybrid search (BM25 + vector) and full-context retrieval read every chunk of a
collection on each chat turn and, for hybrid search, rebuild a BM25 index per
query. For large knowledge bases that dominates latency. This module keeps the
loaded collection (and its derived BM25 indexes) in process memory so that later
turns of a conversation reuse it.

Design
------
* One entry per collection name (shared store). Conversations only register a
  *hold* (``chat_id -> last used``) on the entries they use; an entry is dropped
  once no conversation has used it within the TTL, when its last holder chat is
  released, or when it is evicted to stay inside the memory budget (LRU).
* Staleness is detected with a per-collection *version*. The vector client is
  instrumented (see ``retrieval/vector/factory.py``) to bump the version on every
  write. The version is captured *before* loading, so a write racing with a load
  makes the loaded entry look stale on its next use instead of being served.
* The version store sits behind ``VersionStore`` so a shared implementation
  (e.g. Redis, for multi-worker deployments) can be added later. Only the
  in-process implementation exists today: with several workers each worker's
  cache would not see writes handled by another worker.
* Sizes are *estimates* (``sys.getsizeof`` based), not exact process memory.
* All cache state is touched from the event loop thread only. Versions may be
  bumped from worker threads and are protected by a lock.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Hashable, Optional, Protocol

log = logging.getLogger(__name__)

_MB = 1024 * 1024


class VersionStore(Protocol):
    def get(self, collection_name: str) -> Hashable: ...

    def bump(self, collection_name: str) -> None: ...

    def bump_all(self) -> None: ...


class InMemoryVersionStore:
    """Per-process collection versions. Thread-safe: writes happen in worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._epoch = 0
        self._versions: dict[str, int] = {}

    def get(self, collection_name: str) -> Hashable:
        with self._lock:
            return (self._epoch, self._versions.get(collection_name, 0))

    def bump(self, collection_name: str) -> None:
        with self._lock:
            self._versions[collection_name] = self._versions.get(collection_name, 0) + 1

    def bump_all(self) -> None:
        with self._lock:
            self._epoch += 1


collection_versions: VersionStore = InMemoryVersionStore()


def estimate_object_bytes(obj: Any, _depth: int = 0) -> int:
    """Cheap, approximate deep size for str/number/list/dict trees (e.g. chunk metadata)."""
    size = sys.getsizeof(obj)
    if _depth >= 4:
        return size
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += estimate_object_bytes(k, _depth + 1) + estimate_object_bytes(v, _depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            size += estimate_object_bytes(v, _depth + 1)
    return size


def estimate_result_bytes(result: Any) -> int:
    """Approximate memory held by a GetResult (ids, documents, metadatas)."""
    total = 0
    for field in ('ids', 'documents', 'metadatas'):
        outer = getattr(result, field, None) or []
        for inner in outer:
            total += sys.getsizeof(inner)
            for item in inner:
                total += estimate_object_bytes(item)
    return total


def result_chunk_count(result: Any) -> int:
    docs = getattr(result, 'documents', None)
    return len(docs[0]) if docs and docs[0] else 0


class CacheEntry:
    """A loaded collection plus lazily-built derived data (e.g. BM25 indexes)."""

    def __init__(self, name: str, version: Hashable, result: Any, cache: Optional['CollectionCache'] = None):
        self.name = name
        self.version = version
        self.result = result
        self.chunks = result_chunk_count(result)
        self.size_bytes = estimate_result_bytes(result)
        self.holds: dict[str, float] = {}
        self.last_used = time.monotonic()
        self._cache = cache  # None for transient (uncached) entries
        self._derived: dict[str, Any] = {}
        self._derive_locks: dict[str, asyncio.Lock] = {}

    async def derive(self, key: str, builder: Callable[[], tuple[Any, int]]) -> Any:
        """Build (once) and return derived data. ``builder`` runs in a worker thread
        and returns ``(value, estimated_bytes)``."""
        if key in self._derived:
            return self._derived[key]
        lock = self._derive_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key not in self._derived:
                started = time.monotonic()
                value, size = await asyncio.to_thread(builder)
                self._derived[key] = value
                self.size_bytes += size
                log.info(
                    'rag_collection_cache: built %s collection=%s est_mb=%.1f build_ms=%d',
                    key,
                    self.name,
                    size / _MB,
                    (time.monotonic() - started) * 1000,
                )
                if self._cache is not None:
                    self._cache._on_entry_grown(self)
            return self._derived[key]


class CollectionCache:
    def __init__(
        self,
        *,
        enabled: bool,
        ttl_seconds: float,
        max_bytes: int,
        versions: VersionStore = collection_versions,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self.max_bytes = max_bytes
        self.versions = versions
        self._clock = clock
        self._entries: 'OrderedDict[str, CacheEntry]' = OrderedDict()
        self._loading: dict[str, tuple[Hashable, asyncio.Future]] = {}
        self._total_bytes = 0

    # -- introspection (used by tests) ------------------------------------
    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    # -- public API --------------------------------------------------------
    async def get(
        self,
        name: str,
        chat_id: Optional[str],
        loader: Callable[[], Awaitable[Any]],
    ) -> Optional[CacheEntry]:
        """Return the entry for ``name``, loading it with ``loader`` when needed.

        Without a ``chat_id`` (or when disabled) the entry is loaded but not stored.
        Returns ``None`` when the loader returns ``None``. Loader exceptions propagate.
        """
        if not self.enabled or not chat_id:
            return await self._load_transient(name, loader)

        self._sweep()
        version = self.versions.get(name)
        entry = self._entries.get(name)
        status = 'miss'
        if entry is not None:
            if entry.version == version:
                self._touch(entry, chat_id)
                self._log('hit', entry, chat_id, 0.0)
                return entry
            status = 'stale'
            self._drop(name)

        started = self._clock()
        entry = await self._load_shared(name, version, loader)
        if entry is None:
            return None
        load_ms = (self._clock() - started) * 1000

        if entry.size_bytes > self.max_bytes:
            self._log('too_large', entry, chat_id, load_ms)
            return entry  # used for this request only

        # A concurrent waiter may have stored it already; keep a single stored entry.
        existing = self._entries.get(name)
        if existing is not None and existing.version == entry.version:
            entry = existing
        else:
            if existing is not None:
                self._drop(name)
            entry._cache = self
            self._entries[name] = entry
            self._total_bytes += entry.size_bytes
        self._touch(entry, chat_id)
        self._log(status, entry, chat_id, load_ms)
        self._evict_over_budget(keep=name)
        return entry

    def release_chat(self, chat_id: str) -> None:
        """Drop a conversation's holds (e.g. the chat was deleted)."""
        for name in list(self._entries):
            entry = self._entries[name]
            entry.holds.pop(chat_id, None)
            if not entry.holds:
                self._drop(name)

    def clear(self) -> None:
        for entry in self._entries.values():
            entry._cache = None
        self._entries.clear()
        self._total_bytes = 0

    # -- internals ---------------------------------------------------------
    async def _load_transient(self, name: str, loader) -> Optional[CacheEntry]:
        version = self.versions.get(name)
        result = await loader()
        if result is None:
            return None
        return CacheEntry(name, version, result, cache=None)

    async def _load_shared(self, name: str, version: Hashable, loader) -> Optional[CacheEntry]:
        """Single-flight load: concurrent callers for the same (name, version) share one load."""
        inflight = self._loading.get(name)
        if inflight is not None and inflight[0] == version:
            return await asyncio.shield(inflight[1])

        async def _run() -> Optional[CacheEntry]:
            result = await loader()
            if result is None:
                return None
            return CacheEntry(name, version, result, cache=None)

        future = asyncio.ensure_future(_run())
        self._loading[name] = (version, future)
        try:
            return await asyncio.shield(future)
        finally:
            current = self._loading.get(name)
            if current is not None and current[1] is future:
                del self._loading[name]

    def _touch(self, entry: CacheEntry, chat_id: str) -> None:
        now = self._clock()
        entry.last_used = now
        entry.holds[chat_id] = now
        self._entries.move_to_end(entry.name)

    def _drop(self, name: str) -> None:
        entry = self._entries.pop(name, None)
        if entry is not None:
            self._total_bytes -= entry.size_bytes
            entry._cache = None

    def _sweep(self) -> None:
        now = self._clock()
        for name in list(self._entries):
            entry = self._entries[name]
            entry.holds = {c: t for c, t in entry.holds.items() if now - t <= self.ttl_seconds}
            if not entry.holds:
                self._drop(name)

    def _evict_over_budget(self, keep: Optional[str] = None) -> None:
        while self._total_bytes > self.max_bytes:
            victim = next((n for n in self._entries if n != keep), None)
            if victim is None:
                break
            log.info('rag_collection_cache: evict collection=%s', victim)
            self._drop(victim)

    def _on_entry_grown(self, entry: CacheEntry) -> None:
        """A derived structure was added to a stored entry: recompute totals and enforce the budget."""
        if self._entries.get(entry.name) is not entry:
            return
        self._total_bytes = sum(e.size_bytes for e in self._entries.values())
        if entry.size_bytes > self.max_bytes:
            log.info('rag_collection_cache: too_large collection=%s est_mb=%.1f', entry.name, entry.size_bytes / _MB)
            self._drop(entry.name)
            return
        self._evict_over_budget(keep=entry.name)

    def _log(self, status: str, entry: CacheEntry, chat_id: Optional[str], load_ms: float) -> None:
        log.info(
            'rag_collection_cache: %s collection=%s chunks=%d est_mb=%.1f chat=%s load_ms=%d',
            status,
            entry.name,
            entry.chunks,
            entry.size_bytes / _MB,
            chat_id,
            load_ms,
        )


_cache: Optional[CollectionCache] = None


def get_collection_cache() -> CollectionCache:
    global _cache
    if _cache is None:
        from open_webui.env import (
            ENABLE_RAG_COLLECTION_CACHE,
            RAG_COLLECTION_CACHE_MAX_MB,
            RAG_COLLECTION_CACHE_TTL,
        )

        _cache = CollectionCache(
            enabled=ENABLE_RAG_COLLECTION_CACHE,
            ttl_seconds=RAG_COLLECTION_CACHE_TTL,
            max_bytes=int(RAG_COLLECTION_CACHE_MAX_MB * _MB),
        )
    return _cache
