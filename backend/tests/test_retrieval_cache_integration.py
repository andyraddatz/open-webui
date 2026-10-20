"""Integration test: hybrid-search and full-context retrieval reuse the collection cache.

Needs the backend's runtime dependencies; skipped when they are not installed.
The vector DB factory is stubbed so no real vector database is required.
"""

import os
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('DATA_DIR', tempfile.mkdtemp(prefix='owui-test-'))
os.environ.setdefault('WEBUI_SECRET_KEY', 'test')

pytest.importorskip('langchain_classic')
pytest.importorskip('rank_bm25')
pytest.importorskip('chromadb')

# Stub the factory *before* importing retrieval code: the real one connects to a vector DB at import.
_factory = types.ModuleType('open_webui.retrieval.vector.factory')
_factory.VECTOR_DB_CLIENT = None
_factory.get_vector_db_client = lambda: None
sys.modules.setdefault('open_webui.retrieval.vector.factory', _factory)

from open_webui.retrieval import utils  # noqa: E402
from open_webui.retrieval.collection_cache import CollectionCache, InMemoryVersionStore  # noqa: E402

DOCS = [
    'the quick brown fox jumps over the lazy dog',
    'pack my box with five dozen liquor jugs',
    'sphinx of black quartz judge my vow',
]


class FakeVectorClient:
    def __init__(self):
        self.get_calls = 0

    async def get(self, collection_name):
        self.get_calls += 1
        return SimpleNamespace(
            ids=[[f'{collection_name}-{i}' for i in range(len(DOCS))]],
            documents=[list(DOCS)],
            metadatas=[[{'name': f'doc{i}.txt', 'file_id': f'f{i}'} for i in range(len(DOCS))]],
        )

    async def search(self, collection_name, vectors, filter=None, limit=10):
        return None  # vector side contributes nothing; BM25 carries the test

    supports_hybrid_search = False


async def embedding_function(query, prefix=None, user=None):
    if isinstance(query, str):
        return [1.0, 0.0, 0.0]
    return [[1.0, 0.0, 0.0] for _ in query]


@pytest.fixture
def env(monkeypatch):
    client = FakeVectorClient()
    versions = InMemoryVersionStore()
    cache = CollectionCache(enabled=True, ttl_seconds=100, max_bytes=500 * 1024 * 1024, versions=versions)
    monkeypatch.setattr(utils, 'ASYNC_VECTOR_DB_CLIENT', client)
    monkeypatch.setattr(utils, 'get_collection_cache', lambda: cache)
    return SimpleNamespace(client=client, cache=cache, versions=versions)


async def hybrid(chat_id, queries=('quick fox',), collections=('kb',)):
    return await utils.query_collection_with_hybrid_search(
        collection_names=list(collections),
        queries=list(queries),
        embedding_function=embedding_function,
        k=2,
        reranking_function=None,
        k_reranker=2,
        r=0.0,
        hybrid_bm25_weight=1.0,
        enable_enriched_texts=True,  # forces the non-native path that loads the whole collection
        chat_id=chat_id,
    )


async def test_hybrid_search_loads_collection_once_per_conversation(env):
    first = await hybrid('chat1', queries=('quick fox', 'liquor jugs'))
    assert env.client.get_calls == 1
    second = await hybrid('chat1')
    third = await hybrid('chat2')
    assert env.client.get_calls == 1
    # enriched texts append metadata (filename) to the chunk text
    assert any(d.startswith(DOCS[0]) for d in first['documents'][0])
    assert second['documents'][0][0].startswith(DOCS[0])
    assert third['documents'][0][0].startswith(DOCS[0])


async def test_hybrid_search_without_chat_id_does_not_cache(env):
    await hybrid(None)
    await hybrid(None)
    assert env.client.get_calls == 2


async def test_write_to_collection_triggers_reload(env):
    await hybrid('chat1')
    env.versions.bump('kb')
    await hybrid('chat1')
    assert env.client.get_calls == 2


async def test_cached_results_are_not_corrupted_by_downstream_mutation(env):
    first = await hybrid('chat1')
    for metadata in first['metadatas'][0]:
        metadata['name'] = 'mutated'
        metadata['source'] = 'mutated'
    second = await hybrid('chat1')
    assert all(m['name'] != 'mutated' for m in second['metadatas'][0])


async def test_full_context_path_reuses_cache(env):
    first = await utils.get_all_items_from_collections_cached(['kb'], 'chat1')
    second = await utils.get_all_items_from_collections_cached(['kb'], 'chat1')
    assert env.client.get_calls == 1
    assert first['documents'][0] == DOCS
    assert second['ids'][0] == ['kb-0', 'kb-1', 'kb-2']

    # callers rewrite metadata in place; the cache must not be affected
    second['metadatas'][0][0]['name'] = 'mutated'
    third = await utils.get_all_items_from_collections_cached(['kb'], 'chat1')
    assert third['metadatas'][0][0]['name'] == 'doc0.txt'


async def test_hybrid_and_full_context_share_the_same_entry(env):
    await hybrid('chat1')
    await utils.get_all_items_from_collections_cached(['kb'], 'chat1')
    assert env.client.get_calls == 1
