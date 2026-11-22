"""One cache for everything worth keeping for a while.

The bot asks the same questions over and over. People repeat themselves,
the model searches twice for the same thing when it is unsure, the same
paper comes up in three conversations in a row, and the same Hugging Face
card is fetched every time somebody asks about that model. Each of those
is a network call that somebody waits for.

There were already caches in three different places, each with its own
shape - one keyed by message id, one with a time limit, one that never
expired at all. This module gives them one shape instead:

    from cache import CACHES
    value = CACHES.get("web_search", key)
    CACHES.put("web_search", key, value)

with the time limit and the size limit set per cache, so a search result
and an embedding can have different rules without either of them knowing
about the other.

Two things it deliberately does not do:

**It does not cache failures.** An error is worth retrying, and a cached
error would turn a blip into a dead feature for the length of the TTL.

**It does not cache across people.** A cache key that does not include who
asked is a cache that can answer Jade's question with Damian's reply. The
callers put the scope in the key; this module does not second-guess them,
but it does make it easy to see what the key has to contain.

There is also a small single-flight helper, so that two people asking the
same question at the same moment cause one lookup rather than two.
"""

import asyncio
import threading
import time

# How long each kind of thing is worth keeping.
#
# The numbers are not arbitrary:
#
# - A web search result is a snapshot of pages that rarely change within
#   an hour, and the bot has already been told to say when a source was
#   published.
# - A paper does not change at all. A day is safe.
# - An embedding is a pure function of the text and the model, so it only
#   needs to expire when the model does.
# - The answer itself is the delicate one. Ten minutes is long enough for
#   "what did you just say?" to be instant, and short enough that a
#   question about the weather does not get a stale answer.

DEFAULT_TTL = 600.0

TTLS = {
    "web_search": 900.0,
    "papers": 86400.0,
    "hub": 3600.0,
    "embed": 3600.0,
    "answer": 600.0,
    "kaggle_hint": 600.0,
}

MAX_ENTRIES = {
    "web_search": 200,
    "papers": 200,
    "hub": 300,
    "embed": 500,
    "answer": 150,
    "kaggle_hint": 4,
}


class TTLCache:
    """A dict of values that expire, with a ceiling on how many are kept."""

    def __init__(self, name: str = "", ttl: float = DEFAULT_TTL,
                 max_entries: int = 256):
        self.name = name
        self.ttl = ttl
        self.max_entries = max_entries
        self._data: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str, default=None):
        """The cached value, or None if there isn't a live one."""
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return default
            expires, value = entry
            if expires <= now:
                # Expired: drop it and let the caller do the work again.
                self._data.pop(key, None)
                self.misses += 1
                return default
            self.hits += 1
            return value

    def put(self, key: str, value, ttl: float = None) -> None:
        """Keep a value for a while."""
        if ttl is None:
            ttl = self.ttl
        expires = time.monotonic() + ttl
        with self._lock:
            if key not in self._data and len(self._data) >= self.max_entries:
                # Drop the oldest. dicts keep insertion order, and the
                # oldest is the one closest to expiring anyway.
                oldest = next(iter(self._data))
                self._data.pop(oldest, None)
            self._data[key] = (expires, value)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        with self._lock:
            live = sum(1 for expires, _ in self._data.values()
                       if expires > time.monotonic())
            return {
                "name": self.name,
                "entries": live,
                "hits": self.hits,
                "misses": self.misses,
                "ttl": self.ttl,
            }


class CacheRegistry:
    """The named caches, created on first use."""

    def __init__(self):
        self._caches: dict[str, TTLCache] = {}
        self._lock = threading.Lock()

    def _cache(self, name: str) -> TTLCache:
        cache = self._caches.get(name)
        if cache is None:
            with self._lock:
                cache = self._caches.get(name)
                if cache is None:
                    cache = TTLCache(
                        name=name,
                        ttl=TTLS.get(name, DEFAULT_TTL),
                        max_entries=MAX_ENTRIES.get(name, 256),
                    )
                    self._caches[name] = cache
        return cache

    def get(self, name: str, key: str, default=None):
        return self._cache(name).get(key, default)

    def put(self, name: str, key: str, value, ttl: float = None) -> None:
        self._cache(name).put(key, value, ttl)

    def clear(self, name: str = "") -> None:
        if name:
            self._cache(name).clear()
            return
        with self._lock:
            for cache in self._caches.values():
                cache.clear()

    def stats(self) -> list[dict]:
        with self._lock:
            return [c.stats() for c in self._caches.values()]


# The one registry the bot uses.
CACHES = CacheRegistry()


def normalize_key(text: str) -> str:
    """Tidy a query so that the same question gets the same key.

    "What is filtering theory?" and "what is filtering theory" are the
    same question, and the people typing them are waiting for the same
    answer. The trailing punctuation is dropped and the whitespace is
    collapsed; nothing else is touched, because two different queries
    that happen to share a key would be the wrong kind of clever.
    """
    lowered = " ".join((text or "").split()).lower()
    return lowered.strip(" ?.!.,:;")


class SingleFlight:
    """Make sure one job runs at a time, however many callers want it.

    Two people asking the same question in the same second should cause one
    lookup, not two. The first caller does the work; the others wait for
    its answer.
    """

    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = threading.Lock()

    async def run(self, key: str, factory):
        """Run `factory()` for this key, sharing the result."""
        lock = self._locks.get(key)
        if lock is None:
            with self._guard:
                lock = self._locks.get(key)
                if lock is None:
                    lock = asyncio.Lock()
                    self._locks[key] = lock
        async with lock:
            # Once the lock is taken, the first caller may already have
            # filled the cache while the others were queueing.
            return await factory()


FLIGHTS = SingleFlight()
