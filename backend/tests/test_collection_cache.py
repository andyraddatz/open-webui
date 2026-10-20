import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_webui.retrieval.collection_cache import (  # noqa: E402
    CollectionCache,
    InMemoryVersionStore,
)

MB = 1024 * 1024


def make_result(n_chunks=3, text='lorem ipsum ' * 10):
    return SimpleNamespace(
        ids=[[f'id{i}' for i in range(n_chunks)]],
        documents=[[f'{text}{i}' for i in range(n_chunks)]],
        metadatas=[[{'name': 'f.txt', 'i': i} for i in range(n_chunks)]],
    )


class Loader:
    """Counts loads, like a vector client's get()."""

    def __init__(self, result_factory=make_result, delay=0.0):
        self.calls = 0
        self.result_factory = result_factory
        self.delay = delay

    def __call__(self, name):
        async def load():
            self.calls += 1
            if self.delay:
                await asyncio.sleep(self.delay)
            return self.result_factory()

        return load


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_cache(**kw):
    kw.setdefault('enabled', True)
    kw.setdefault('ttl_seconds', 100)
    kw.setdefault('max_bytes', 100 * MB)
    kw.setdefault('versions', InMemoryVersionStore())
    return CollectionCache(**kw)


async def test_second_request_in_same_chat_is_a_hit():
    cache, loader = make_cache(), Loader()
    a = await cache.get('kb', 'chat1', loader('kb'))
    b = await cache.get('kb', 'chat1', loader('kb'))
    assert loader.calls == 1
    assert a is b


async def test_entry_is_shared_between_chats():
    cache, loader = make_cache(), Loader()
    a = await cache.get('kb', 'chat1', loader('kb'))
    b = await cache.get('kb', 'chat2', loader('kb'))
    assert loader.calls == 1
    assert a is b
    assert set(a.holds) == {'chat1', 'chat2'}


async def test_no_chat_id_or_disabled_never_stores():
    cache, loader = make_cache(), Loader()
    await cache.get('kb', None, loader('kb'))
    await cache.get('kb', None, loader('kb'))
    assert loader.calls == 2
    assert 'kb' not in cache

    off, loader = make_cache(enabled=False), Loader()
    await off.get('kb', 'chat1', loader('kb'))
    await off.get('kb', 'chat1', loader('kb'))
    assert loader.calls == 2
    assert 'kb' not in off


async def test_write_makes_entry_stale_and_reloads():
    versions = InMemoryVersionStore()
    cache, loader = make_cache(versions=versions), Loader()
    await cache.get('kb', 'chat1', loader('kb'))
    versions.bump('other')  # unrelated collection: still a hit
    await cache.get('kb', 'chat1', loader('kb'))
    assert loader.calls == 1
    versions.bump('kb')
    await cache.get('kb', 'chat1', loader('kb'))
    assert loader.calls == 2
    versions.bump_all()
    await cache.get('kb', 'chat1', loader('kb'))
    assert loader.calls == 3


async def test_write_during_load_is_detected_afterwards():
    versions = InMemoryVersionStore()
    cache = make_cache(versions=versions)
    calls = 0

    def loader():
        async def load():
            nonlocal calls
            calls += 1
            versions.bump('kb')  # a write lands while we were loading
            return make_result()

        return load

    await cache.get('kb', 'chat1', loader())
    await cache.get('kb', 'chat1', loader())
    assert calls == 2


async def test_concurrent_requests_share_one_load():
    cache, loader = make_cache(), Loader(delay=0.05)
    entries = await asyncio.gather(*[cache.get('kb', f'chat{i}', loader('kb')) for i in range(5)])
    assert loader.calls == 1
    assert all(e is entries[0] for e in entries)
    assert len(entries[0].holds) == 5


async def test_ttl_expires_entries_without_recent_holders():
    clock = Clock()
    cache, loader = make_cache(ttl_seconds=100, clock=clock), Loader()
    await cache.get('kb', 'chat1', loader('kb'))
    clock.now = 90
    await cache.get('kb', 'chat1', loader('kb'))  # refreshes the hold
    clock.now = 180
    await cache.get('kb', 'chat1', loader('kb'))
    assert loader.calls == 1
    clock.now = 400
    await cache.get('kb', 'chat1', loader('kb'))
    assert loader.calls == 2


async def test_only_expired_holders_are_dropped():
    clock = Clock()
    cache, loader = make_cache(ttl_seconds=100, clock=clock), Loader()
    await cache.get('kb', 'old', loader('kb'))
    clock.now = 90
    entry = await cache.get('kb', 'new', loader('kb'))
    clock.now = 150  # 'old' idle 150 > ttl, 'new' idle 60
    await cache.get('kb', 'new', loader('kb'))
    assert loader.calls == 1
    assert set(entry.holds) == {'new'}


async def test_release_chat_drops_entries_with_no_remaining_holders():
    cache, loader = make_cache(), Loader()
    await cache.get('a', 'chat1', loader('a'))
    await cache.get('b', 'chat1', loader('b'))
    await cache.get('b', 'chat2', loader('b'))
    cache.release_chat('chat1')
    assert 'a' not in cache
    assert 'b' in cache
    cache.release_chat('chat2')
    assert 'b' not in cache
    assert cache.total_bytes == 0


async def test_lru_eviction_keeps_total_under_budget():
    one = Loader(lambda: make_result(50, 'x' * 2000))
    probe = make_cache()
    size = (await probe.get('probe', 'c', one('probe'))).size_bytes
    cache = make_cache(max_bytes=int(size * 2.5))
    for name in ('a', 'b'):
        await cache.get(name, 'c', one(name))
    await cache.get('a', 'c', one('a'))  # 'b' is now least recently used
    await cache.get('c', 'c', one('c'))
    assert 'b' not in cache
    assert 'a' in cache and 'c' in cache
    assert cache.total_bytes <= cache.max_bytes


async def test_oversized_collection_is_served_but_not_kept():
    cache, loader = make_cache(max_bytes=1000), Loader(lambda: make_result(50, 'x' * 2000))
    entry = await cache.get('big', 'chat1', loader('big'))
    assert entry is not None and entry.chunks == 50
    assert 'big' not in cache
    assert cache.total_bytes == 0


async def test_loader_returning_none_is_not_cached():
    cache = make_cache()
    calls = 0

    async def load():
        nonlocal calls
        calls += 1
        return None

    assert await cache.get('kb', 'chat1', load) is None
    assert await cache.get('kb', 'chat1', load) is None
    assert calls == 2


async def test_loader_error_propagates_and_is_not_cached():
    cache = make_cache()
    ok = Loader()

    async def boom():
        raise RuntimeError('db down')

    with pytest.raises(RuntimeError):
        await cache.get('kb', 'chat1', boom)
    await cache.get('kb', 'chat1', ok('kb'))
    assert ok.calls == 1


async def test_derived_data_is_built_once_and_counted_in_budget():
    cache, loader = make_cache(), Loader()
    entry = await cache.get('kb', 'chat1', loader('kb'))
    before = cache.total_bytes
    builds = 0

    def builder():
        nonlocal builds
        builds += 1
        return {'index': True}, 5000

    r1, r2 = await asyncio.gather(entry.derive('bm25:plain', builder), entry.derive('bm25:plain', builder))
    again = await entry.derive('bm25:plain', builder)
    assert builds == 1
    assert r1 is r2 is again
    assert cache.total_bytes == before + 5000


async def test_derived_growth_can_push_entry_out_of_budget():
    probe = make_cache()
    size = (await probe.get('p', 'c', Loader()('p'))).size_bytes
    cache = make_cache(max_bytes=size + 100)
    entry = await cache.get('kb', 'chat1', Loader()('kb'))
    await entry.derive('bm25:plain', lambda: ('x', 10_000))
    assert 'kb' not in cache
    assert cache.total_bytes == 0


async def test_get_all_returns_cached_entry_after_derive_on_hit():
    cache, loader = make_cache(), Loader()
    first = await cache.get('kb', 'chat1', loader('kb'))
    await first.derive('bm25:plain', lambda: ('idx', 10))
    second = await cache.get('kb', 'chat2', loader('kb'))
    assert await second.derive('bm25:plain', lambda: ('other', 10)) == 'idx'
