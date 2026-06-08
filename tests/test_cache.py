import asyncio
import threading
import time

import pytest

from llm_cache_mem import AsyncLRUCache, LRUCache

# ---------- basic get/set ----------


def test_get_on_empty_returns_none():
    c = LRUCache()
    assert c.get("nope") is None


def test_set_then_get_returns_value():
    c = LRUCache()
    c.set("k", {"text": "hello"})
    assert c.get("k") == {"text": "hello"}


def test_set_overwrites_existing_key():
    c = LRUCache()
    c.set("k", "v1")
    c.set("k", "v2")
    assert c.get("k") == "v2"
    assert c.size == 1


def test_size_property_tracks_entries():
    c = LRUCache(maxsize=10)
    assert c.size == 0
    c.set("a", 1)
    c.set("b", 2)
    assert c.size == 2


def test_invalid_maxsize_raises():
    with pytest.raises(ValueError):
        LRUCache(maxsize=0)
    with pytest.raises(ValueError):
        LRUCache(maxsize=-5)


def test_invalid_ttl_raises():
    with pytest.raises(ValueError):
        LRUCache(ttl_seconds=0)
    with pytest.raises(ValueError):
        LRUCache(ttl_seconds=-1.0)


# ---------- LRU eviction ----------


def test_lru_evicts_oldest_at_capacity():
    c = LRUCache(maxsize=3)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    c.set("d", 4)  # should evict "a"
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3
    assert c.get("d") == 4
    assert c.stats()["evictions"] == 1


def test_get_refreshes_lru_position():
    c = LRUCache(maxsize=3)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    # touch "a" so it's most-recently-used
    assert c.get("a") == 1
    c.set("d", 4)  # should now evict "b", not "a"
    assert c.get("a") == 1
    assert c.get("b") is None
    assert c.get("c") == 3


def test_set_existing_key_does_not_evict():
    c = LRUCache(maxsize=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("a", 99)  # replace existing, must not evict "b"
    assert c.get("a") == 99
    assert c.get("b") == 2
    assert c.stats()["evictions"] == 0


# ---------- TTL ----------


def test_ttl_expires_entry():
    c = LRUCache(maxsize=10, ttl_seconds=0.05)
    c.set("k", "v")
    time.sleep(0.08)
    assert c.get("k") is None
    s = c.stats()
    assert s["expirations"] == 1
    assert s["misses"] == 1


def test_ttl_none_never_expires():
    c = LRUCache(maxsize=10, ttl_seconds=None)
    c.set("k", "v")
    time.sleep(0.05)
    assert c.get("k") == "v"
    assert c.stats()["expirations"] == 0


def test_set_refreshes_ttl():
    c = LRUCache(maxsize=10, ttl_seconds=0.1)
    c.set("k", "v1")
    time.sleep(0.07)
    c.set("k", "v2")  # refreshes expiry
    time.sleep(0.07)  # total 0.14 from first set, 0.07 from second
    assert c.get("k") == "v2"


def test_contains_removes_expired():
    c = LRUCache(maxsize=10, ttl_seconds=0.05)
    c.set("k", "v")
    time.sleep(0.08)
    assert c.contains("k") is False
    assert c.size == 0


# ---------- stats ----------


def test_stats_counts_hits_and_misses():
    c = LRUCache(maxsize=10)
    c.set("a", 1)
    c.get("a")  # hit
    c.get("a")  # hit
    c.get("missing")  # miss
    s = c.stats()
    assert s["hits"] == 2
    assert s["misses"] == 1
    assert s["hit_ratio"] == pytest.approx(2 / 3)


def test_stats_counts_evictions():
    c = LRUCache(maxsize=2)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)  # evict "a"
    c.set("d", 4)  # evict "b"
    assert c.stats()["evictions"] == 2


def test_stats_size_matches_store():
    c = LRUCache(maxsize=5)
    c.set("a", 1)
    c.set("b", 2)
    s = c.stats()
    assert s["size"] == 2
    assert s["maxsize"] == 5


def test_reset_stats_zeros_counters_but_keeps_entries():
    c = LRUCache(maxsize=5)
    c.set("a", 1)
    c.get("a")
    c.get("missing")
    c.reset_stats()
    s = c.stats()
    assert s["hits"] == 0
    assert s["misses"] == 0
    assert s["evictions"] == 0
    assert s["expirations"] == 0
    assert s["size"] == 1
    assert c.get("a") == 1


def test_hit_ratio_zero_when_no_calls():
    c = LRUCache()
    assert c.stats()["hit_ratio"] == 0.0


# ---------- delete / clear / contains ----------


def test_delete_returns_true_when_present():
    c = LRUCache()
    c.set("k", "v")
    assert c.delete("k") is True
    assert c.get("k") is None


def test_delete_returns_false_when_missing():
    c = LRUCache()
    assert c.delete("k") is False


def test_clear_empties_store():
    c = LRUCache()
    c.set("a", 1)
    c.set("b", 2)
    c.clear()
    assert c.size == 0
    assert c.get("a") is None


def test_contains_does_not_affect_stats():
    c = LRUCache()
    c.set("a", 1)
    before = c.stats()
    assert c.contains("a") is True
    assert c.contains("missing") is False
    after = c.stats()
    assert before["hits"] == after["hits"]
    assert before["misses"] == after["misses"]


def test_contains_does_not_refresh_lru():
    c = LRUCache(maxsize=3)
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    assert c.contains("a") is True  # must not promote "a"
    c.set("d", 4)  # should evict "a"
    assert c.get("a") is None
    assert c.get("b") == 2


# ---------- decorator (sync) ----------


def test_decorator_caches_fn_results():
    c = LRUCache()
    call_count = 0

    @c.cached()
    def fetch(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    assert fetch(5) == 10
    assert fetch(5) == 10  # cached
    assert call_count == 1
    assert fetch(6) == 12  # different arg, miss
    assert call_count == 2


def test_decorator_key_fn_override():
    c = LRUCache()
    calls = []

    @c.cached(key_fn=lambda *args, **kw: "fixed-key")
    def fetch(x: int) -> int:
        calls.append(x)
        return x

    # all calls collapse to the same key
    fetch(1)
    fetch(2)
    fetch(3)
    assert len(calls) == 1
    assert calls[0] == 1


def test_decorator_with_kwargs():
    c = LRUCache()
    calls = []

    @c.cached()
    def fetch(a: int, b: int = 0) -> int:
        calls.append((a, b))
        return a + b

    fetch(1, b=2)
    fetch(1, b=2)
    fetch(1, b=3)
    assert len(calls) == 2


def test_decorator_caches_none_return_value():
    c = LRUCache()
    call_count = 0

    @c.cached()
    def maybe(x: int) -> int | None:
        nonlocal call_count
        call_count += 1
        return None if x == 0 else x

    assert maybe(0) is None
    assert maybe(0) is None  # should NOT re-call
    assert call_count == 1


def test_decorator_none_hit_counts_one_hit_no_miss():
    c = LRUCache()

    @c.cached()
    def maybe(x: int) -> int | None:
        return None

    maybe(0)  # miss + store None
    c.reset_stats()
    maybe(0)  # cache hit on stored None
    s = c.stats()
    assert s["hits"] == 1
    assert s["misses"] == 0
    assert s["hit_ratio"] == pytest.approx(1.0)


# ---------- async ----------


async def test_async_cached_works():
    c = AsyncLRUCache()
    call_count = 0

    @c.cached()
    async def fetch(x: int) -> int:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0)  # cooperative
        return x * 3

    assert await fetch(4) == 12
    assert await fetch(4) == 12  # cached
    assert call_count == 1


async def test_async_get_set_direct():
    c = AsyncLRUCache(maxsize=5)
    c.set("k", {"a": 1})
    assert c.get("k") == {"a": 1}
    assert c.size == 1


async def test_async_decorator_key_fn_override():
    c = AsyncLRUCache()
    calls = []

    @c.cached(key_fn=lambda x: f"k:{x % 2}")
    async def fetch(x: int) -> int:
        calls.append(x)
        return x

    await fetch(1)  # key "k:1"
    await fetch(3)  # key "k:1" - cached
    await fetch(2)  # key "k:0"
    assert len(calls) == 2


async def test_async_decorator_caches_none_return_value():
    c = AsyncLRUCache()
    call_count = 0

    @c.cached()
    async def maybe(x: int) -> int | None:
        nonlocal call_count
        call_count += 1
        return None if x == 0 else x

    assert await maybe(0) is None
    assert await maybe(0) is None  # should NOT re-call
    assert call_count == 1
    s = c.stats()
    assert s["hits"] == 1
    assert s["misses"] == 1  # first call missed, second hit


async def test_async_ttl_expires():
    c = AsyncLRUCache(maxsize=10, ttl_seconds=0.05)
    c.set("k", "v")
    await asyncio.sleep(0.08)
    assert c.get("k") is None


async def test_async_delete_clear_contains():
    c = AsyncLRUCache(maxsize=5)
    c.set("k", "v")
    assert c.contains("k") is True
    assert c.delete("k") is True
    assert c.delete("k") is False
    assert c.contains("k") is False
    c.set("a", 1)
    c.set("b", 2)
    c.clear()
    assert c.size == 0


async def test_async_reset_stats_and_properties():
    c = AsyncLRUCache(maxsize=7, ttl_seconds=1.5)
    assert c.maxsize == 7
    assert c.ttl_seconds == 1.5
    c.set("a", 1)
    c.get("a")  # hit
    c.get("missing")  # miss
    assert c.stats()["hits"] == 1
    c.reset_stats()
    s = c.stats()
    assert s["hits"] == 0
    assert s["misses"] == 0
    assert s["size"] == 1  # entries preserved


# ---------- properties ----------


def test_maxsize_and_ttl_properties():
    c = LRUCache(maxsize=42, ttl_seconds=3.0)
    assert c.maxsize == 42
    assert c.ttl_seconds == 3.0


def test_ttl_seconds_property_none_by_default():
    c = LRUCache(maxsize=10)
    assert c.ttl_seconds is None


# ---------- concurrency ----------


def test_concurrent_set_get_threadsafe():
    c = LRUCache(maxsize=10_000)
    errors: list[BaseException] = []

    def writer(start: int) -> None:
        try:
            for i in range(start, start + 200):
                c.set(f"k{i}", i)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def reader(start: int) -> None:
        try:
            for i in range(start, start + 200):
                c.get(f"k{i}")  # may be hit or miss; just must not crash
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = []
    for i in range(5):
        threads.append(threading.Thread(target=writer, args=(i * 200,)))
        threads.append(threading.Thread(target=reader, args=(i * 200,)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    # final state must be consistent (no exceptions, size within bounds)
    assert c.size <= 10_000


def test_concurrent_eviction_keeps_invariants():
    c = LRUCache(maxsize=50)
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            for j in range(100):
                c.set(f"t{i}-{j}", j)
                c.get(f"t{i}-{j}")
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert c.size <= 50
    # total set calls = 500. capacity = 50. final size <= 50 means at least
    # 450 evictions happened.
    assert c.stats()["evictions"] >= 450
