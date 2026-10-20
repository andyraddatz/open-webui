import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_webui.retrieval.collection_cache import InMemoryVersionStore  # noqa: E402


class FakeClient:
    def __init__(self):
        self.calls = []

    def insert(self, collection_name, items):
        self.calls.append(('insert', collection_name))

    def upsert(self, collection_name, items):
        self.calls.append(('upsert', collection_name))

    def delete(self, collection_name, ids=None, filter=None):
        self.calls.append(('delete', collection_name))
        if ids == 'boom':
            raise RuntimeError('boom')

    def delete_collection(self, collection_name):
        self.calls.append(('delete_collection', collection_name))

    def reset(self):
        self.calls.append(('reset', None))

    def get(self, collection_name):
        return None


def instrumented(monkeypatch):
    from open_webui.retrieval.vector import instrumentation

    versions = InMemoryVersionStore()
    monkeypatch.setattr(instrumentation, 'collection_versions', versions)
    return instrumentation.instrument_client(FakeClient()), versions


def test_writes_bump_only_their_collection(monkeypatch):
    client, versions = instrumented(monkeypatch)
    for call in (
        lambda: client.insert('kb', []),
        lambda: client.upsert(collection_name='kb', items=[]),
        lambda: client.delete('kb', ids=['1']),
        lambda: client.delete_collection(collection_name='kb'),
    ):
        previous = versions.get('kb')
        call()
        assert versions.get('kb') != previous
    assert versions.get('other') == (0, 0)


def test_reads_do_not_bump(monkeypatch):
    client, versions = instrumented(monkeypatch)
    before = versions.get('kb')
    client.get('kb')
    assert versions.get('kb') == before


def test_failed_write_still_bumps(monkeypatch):
    client, versions = instrumented(monkeypatch)
    before = versions.get('kb')
    try:
        client.delete('kb', ids='boom')
    except RuntimeError:
        pass
    assert versions.get('kb') != before


def test_reset_invalidates_everything(monkeypatch):
    client, versions = instrumented(monkeypatch)
    before = versions.get('kb')
    client.reset()
    assert versions.get('kb') != before
