"""llm-cache-mem - in-process LRU cache for LLM responses.

Small, zero-dependency LRU cache keyed by request hash (or any string
key you compute), with optional TTL and both sync and async front-ends.

    from llm_cache_mem import LRUCache

    cache = LRUCache(maxsize=1000, ttl_seconds=3600)
    cache.set("hash-abc", {"text": "hello"})
    cache.get("hash-abc")  # -> {"text": "hello"}

Decorator form wraps any callable:

    @cache.cached(key_fn=lambda prompt: sha256(prompt.encode()).hexdigest())
    def call_llm(prompt: str) -> dict:
        ...

`AsyncLRUCache` mirrors the API for `async def` callables.

Sibling to `cachebench` (provider-side cache observability),
`llm-message-hash-py` (canonical request hashing for keys), and
`llm-batch-coalesce` (in-flight dedupe).
"""

from llm_cache_mem.cache import AsyncLRUCache, LRUCache

__version__ = "0.1.0"

__all__ = [
    "AsyncLRUCache",
    "LRUCache",
    "__version__",
]
