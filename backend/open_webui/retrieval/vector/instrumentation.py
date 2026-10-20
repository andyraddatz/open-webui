"""Hooks that keep the collection cache (retrieval/collection_cache.py) coherent with vector DB writes."""

import functools

from open_webui.retrieval.collection_cache import collection_versions
from open_webui.retrieval.vector.main import VectorDBBase


def instrument_client(client: VectorDBBase) -> VectorDBBase:
    """Bump the collection version after every write so cached copies of the
    collection (see retrieval/collection_cache.py) are detected as stale.

    Methods are wrapped on the instance, so the client keeps its concrete type
    (callers probe e.g. ``hybrid_search`` support through the class).
    """

    def wrap_collection_write(method_name: str):
        original = getattr(client, method_name)

        @functools.wraps(original)
        def wrapper(collection_name, *args, **kwargs):
            try:
                return original(collection_name, *args, **kwargs)
            finally:
                collection_versions.bump(collection_name)

        setattr(client, method_name, wrapper)

    for method_name in ('insert', 'upsert', 'delete', 'delete_collection'):
        wrap_collection_write(method_name)

    original_reset = client.reset

    @functools.wraps(original_reset)
    def reset(*args, **kwargs):
        try:
            return original_reset(*args, **kwargs)
        finally:
            collection_versions.bump_all()

    client.reset = reset
    return client
