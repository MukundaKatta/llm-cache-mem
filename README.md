# llm-cache-mem

[![PyPI](https://img.shields.io/pypi/v/llm-cache-mem.svg)](https://pypi.org/project/llm-cache-mem/)
[![Python](https://img.shields.io/pypi/pyversions/llm-cache-mem.svg)](https://pypi.org/project/llm-cache-mem/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**In-process LRU cache for LLM responses. TTL, sync + async, decorator + direct API, thread-safe.**

Repeated identical LLM calls during dev, evals, and idempotent agent steps
waste tokens and slow the loop. This library is a small, zero-dependency
LRU cache keyed by request hash (or any string key you compute), with
optional TTL, and both sync and async front-ends.

## Install

```bash
pip install llm-cache-mem
```

## Direct get/set

```python
from llm_cache_mem import LRUCache

cache = LRUCache(maxsize=1000, ttl_seconds=3600)

cache.set("hash-abc", {"text": "hello", "tokens": 10})
hit = cache.get("hash-abc")   # -> {"text": "hello", "tokens": 10}
miss = cache.get("hash-xyz")  # -> None
```

## Decorator form

Wrap any callable. By default the key is a hash of `repr((args, sorted_kwargs))`.

```python
import hashlib
from llm_cache_mem import LRUCache

cache = LRUCache(maxsize=500)

@cache.cached(key_fn=lambda prompt: hashlib.sha256(prompt.encode()).hexdigest())
def call_llm(prompt: str) -> dict:
    # expensive call here
    return {"text": "..."}

call_llm("hello")  # miss, calls fn
call_llm("hello")  # hit, returns cached value
```

## Async

```python
from llm_cache_mem import AsyncLRUCache

cache = AsyncLRUCache(maxsize=200, ttl_seconds=60)

@cache.cached()
async def call_llm_async(prompt: str) -> dict:
    return {"text": "..."}

await call_llm_async("hello")  # miss
await call_llm_async("hello")  # hit
```

## TTL

Pass `ttl_seconds` to expire entries. `None` (the default) means entries
never expire on their own and only roll out via LRU eviction.

```python
cache = LRUCache(maxsize=100, ttl_seconds=2.0)
cache.set("k", "v")
# 3 seconds later
cache.get("k")  # -> None (expired, also evicted)
```

## Stats

```python
cache.stats()
# {
#   "hits": 7,
#   "misses": 3,
#   "evictions": 0,
#   "expirations": 1,
#   "size": 6,
#   "maxsize": 100,
#   "hit_ratio": 0.7,
# }
cache.reset_stats()
```

## What it does NOT do

- No persistence. Entries live in-process and vanish when the process exits.
- No HTTP. Doesn't talk to any LLM provider.
- No cross-process or cross-host sharing. For shared cache, wrap a Redis
  or Memcached client behind the same `get`/`set` surface.
- No automatic key derivation from raw LLM request shapes. Pair with
  [`llm-message-hash-py`](https://pypi.org/project/llm-message-hash-py/)
  to canonicalize and hash request structures into stable string keys.

## Siblings in the agent-stack

- [`cachebench`](https://pypi.org/project/cachebench/) - provider-side
  prompt-cache observability (hit ratios from the model API response).
- [`llm-message-hash-py`](https://pypi.org/project/llm-message-hash-py/) -
  canonical request hashing, a natural fit as the `key_fn` for this cache.
- [`llm-batch-coalesce`](https://pypi.org/project/llm-batch-coalesce/) -
  in-flight dedupe so concurrent identical calls share one round-trip.

## License

MIT
