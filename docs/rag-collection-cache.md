# RAG collection cache (fork feature)

Keeps whole vector-DB collections (and their BM25 indexes) in process memory so later turns of a
conversation don't reload and re-index a large knowledge base. Covers hybrid search (when the vector DB
has no native hybrid search, or enriched texts are on) and full-context retrieval.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENABLE_RAG_COLLECTION_CACHE` | `False` | Turn the cache on. Off = upstream behaviour. |
| `RAG_COLLECTION_CACHE_TTL` | `1800` | Seconds a collection is kept without any conversation using it. |
| `RAG_COLLECTION_CACHE_MAX_MB` | `4096` | Approximate memory budget (estimate, not exact RSS); least recently used evicted first. A single collection larger than this is used but not kept. |

- One entry per collection, shared by all conversations; each conversation just holds it.
- Any write to a collection (insert/upsert/delete/delete_collection/reset) invalidates the cached copy.
- State is per worker process and versions are in-memory: with `UVICORN_WORKERS > 1` other workers
  won't see writes made by one worker. Intended for single-worker deployments.
- Requests without a `chat_id` bypass the cache.

## Verifying

```
docker logs <container> 2>&1 | grep rag_collection_cache
rag_collection_cache: miss collection=<id> chunks=N est_mb=X chat=<chat_id> load_ms=T
rag_collection_cache: built bm25:enriched|plain collection=<id> est_mb=X build_ms=T
rag_collection_cache: hit  collection=<id> chunks=N est_mb=X chat=<chat_id> load_ms=0
```

`stale` appears on the first turn after the collection was modified; `too_large` and `evict` indicate the
budget is too small. Log level must be INFO for the `open_webui` loggers (default).
