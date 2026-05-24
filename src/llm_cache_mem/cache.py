"""LRUCache + AsyncLRUCache implementation."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def _default_key_fn(*args: Any, **kwargs: Any) -> str:
    """Default key derivation: repr of (args, sorted-kwargs).

    Not cryptographic. Stable within one Python process for hashable repr
    output. Callers handling secrets or cross-process keys should pass an
    explicit `key_fn` that hashes their request structure.
    """
    return repr((args, tuple(sorted(kwargs.items()))))


class LRUCache(Generic[T]):
    """Thread-safe in-process LRU cache with optional TTL.

    Args:
        maxsize: maximum number of entries before LRU eviction.
        ttl_seconds: per-entry expiry in seconds. `None` means entries
            never expire on their own and only roll out via LRU.

    Storage is a `collections.OrderedDict` of `key -> (value, expire_at)`.
    `expire_at` is `None` when `ttl_seconds is None`.
    """

    def __init__(self, maxsize: int = 1000, ttl_seconds: float | None = None) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0 or None")
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[T, float | None]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    # ---- core ops ----

    def get(self, key: str) -> T | None:
        """Return value for `key`, or `None` on miss / expiry.

        Refreshes LRU position on hit. Counts as a hit or miss in stats.
        Expired entries are removed and counted as an expiration plus a miss.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, expire_at = entry
            if expire_at is not None and time.monotonic() >= expire_at:
                del self._store[key]
                self._expirations += 1
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: T) -> None:
        """Insert or replace `key`. Refreshes TTL. Evicts LRU if over capacity."""
        expire_at = time.monotonic() + self._ttl if self._ttl is not None else None
        with self._lock:
            if key in self._store:
                self._store[key] = (value, expire_at)
                self._store.move_to_end(key)
                return
            self._store[key] = (value, expire_at)
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)
                self._evictions += 1

    def delete(self, key: str) -> bool:
        """Remove `key`. Returns True if it was present, False otherwise."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Drop all entries. Stats are preserved (use `reset_stats()` to zero)."""
        with self._lock:
            self._store.clear()

    def contains(self, key: str) -> bool:
        """Whether `key` is present and not expired.

        Does NOT count as a hit or miss in stats and does NOT refresh LRU
        position. Expired entries are still removed when discovered.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return False
            _, expire_at = entry
            if expire_at is not None and time.monotonic() >= expire_at:
                del self._store[key]
                self._expirations += 1
                return False
            return True

    # ---- introspection ----

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def ttl_seconds(self) -> float | None:
        return self._ttl

    def stats(self) -> dict[str, Any]:
        """Snapshot of counters. Includes derived `hit_ratio`."""
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = (self._hits / total) if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "size": len(self._store),
                "maxsize": self._maxsize,
                "hit_ratio": hit_ratio,
            }

    def reset_stats(self) -> None:
        """Zero hits, misses, evictions, expirations. Entries are untouched."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0

    # ---- decorator ----

    def cached(
        self,
        key_fn: Callable[..., str] | None = None,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator that wraps a sync callable and memoizes by computed key.

        Args:
            key_fn: function `(*args, **kwargs) -> str`. Defaults to
                `repr((args, sorted_kwargs))`.
        """
        kf = key_fn if key_fn is not None else _default_key_fn

        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            @wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                key = kf(*args, **kwargs)
                hit = self.get(key)
                if hit is not None:
                    return hit
                # Also handle the case where the cached value is exactly None:
                # `get` would return None for both miss and stored-None. We use
                # `contains` here to disambiguate without double-counting stats.
                if self.contains(key):
                    # rare path: stored None. Re-fetch through get to refresh
                    # LRU and record a hit. The previous get already counted a
                    # miss, so adjust counters to keep totals consistent.
                    with self._lock:
                        self._misses -= 1
                    return self.get(key)  # type: ignore[return-value]
                value = fn(*args, **kwargs)
                self.set(key, value)
                return value

            return wrapper

        return decorator


class AsyncLRUCache(Generic[T]):
    """Async-friendly LRU cache.

    The underlying storage operations are the same as `LRUCache` and remain
    thread-safe via an `RLock`. CPython dict ops are atomic for single
    statements, so synchronous get/set on this class are safe to call from
    async code without holding the event loop. The `cached` decorator is
    `async def`-aware: it `await`s the wrapped function on a miss.
    """

    def __init__(self, maxsize: int = 1000, ttl_seconds: float | None = None) -> None:
        self._inner: LRUCache[T] = LRUCache(maxsize=maxsize, ttl_seconds=ttl_seconds)

    def get(self, key: str) -> T | None:
        return self._inner.get(key)

    def set(self, key: str, value: T) -> None:
        self._inner.set(key, value)

    def delete(self, key: str) -> bool:
        return self._inner.delete(key)

    def clear(self) -> None:
        self._inner.clear()

    def contains(self, key: str) -> bool:
        return self._inner.contains(key)

    @property
    def size(self) -> int:
        return self._inner.size

    @property
    def maxsize(self) -> int:
        return self._inner.maxsize

    @property
    def ttl_seconds(self) -> float | None:
        return self._inner.ttl_seconds

    def stats(self) -> dict[str, Any]:
        return self._inner.stats()

    def reset_stats(self) -> None:
        self._inner.reset_stats()

    def cached(
        self,
        key_fn: Callable[..., str] | None = None,
    ) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
        """Decorator for an `async def` callable.

        Args:
            key_fn: function `(*args, **kwargs) -> str`. Defaults to
                `repr((args, sorted_kwargs))`.
        """
        kf = key_fn if key_fn is not None else _default_key_fn
        inner = self._inner

        def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> T:
                key = kf(*args, **kwargs)
                hit = inner.get(key)
                if hit is not None:
                    return hit
                if inner.contains(key):
                    with inner._lock:
                        inner._misses -= 1
                    return inner.get(key)  # type: ignore[return-value]
                value = await fn(*args, **kwargs)
                inner.set(key, value)
                return value

            return wrapper

        return decorator
