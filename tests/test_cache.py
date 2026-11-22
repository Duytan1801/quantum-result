"""Tests for the cache, and for the caching that was wired into the bot.

The point of these is not that a dict works. It is that the things which
must NOT be cached are not cached, and that a cache hit really does skip
the expensive work.
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT, need  # noqa: E402

sys.path.insert(0, ROOT)

need("SUPABASE_KEY")


from cache import CACHES, TTLCache  # noqa: E402

FAILED = []
PASSED = 0


def check(label, got, want):
    global PASSED
    if got == want:
        PASSED += 1
        print(f"  ok   {label}")
    else:
        FAILED.append(label)
        print(f"  FAIL {label}\n         got:  {got!r}\n         want: {want!r}")


def check_true(label, cond):
    check(label, bool(cond), True)


print("=" * 70)
print("THE CACHE ITSELF")
print("=" * 70)

c = TTLCache("test", ttl=1.0, max_entries=3)
c.put("a", 1)
check("a value comes back", c.get("a"), 1)
check("a missing value is None", c.get("nope"), None)
check("a missing value can have a default", c.get("nope", "d"), "d")

time.sleep(1.1)
check("an expired value is gone", c.get("a"), None)

stats = c.stats()
check_true("the stats count hits", stats["hits"] > 0)
check_true("the stats count misses", stats["misses"] > 0)

c = TTLCache("test", ttl=60, max_entries=3)
for i in range(5):
    c.put(f"k{i}", i)
check("the size limit is respected", len(c._data), 3)
check("the newest entry survived", c.get("k4"), 4)

print()
print("=" * 70)
print("THE ANSWER CACHE RULES")
print("=" * 70)

from agent import _answer_cache_key, _cacheable, needs_fresh_data  # noqa: E402

JADE = {"id": "909453181916938301", "name": "jade", "plain": "jade"}
DAMIAN = {"id": "1499211380052721684", "name": "lain", "plain": "lain"}

key_a = _answer_cache_key("What is filtering theory?", JADE, "channel:1", "x")
key_b = _answer_cache_key("what is filtering theory?", JADE, "channel:1", "x")
key_c = _answer_cache_key("What is filtering theory?", DAMIAN, "channel:1", "x")
key_d = _answer_cache_key("What is filtering theory?", JADE, "channel:2", "x")

check("the same question by the same person has the same key", key_a, key_b)
check_true("a different person has a different key", key_a != key_c)
check_true("a different room has a different key", key_a != key_d)

FACTUAL = {"answer": "Filtering theory is the study of signals...", "tools_used": ["web_search"], "files": []}
CHATTY = {"answer": "That sounds rough, honestly.", "tools_used": [], "files": []}
WITH_FILE = {"answer": "Here is the plot.", "tools_used": ["code"], "files": ["/tmp/x.png"]}
DELEGATED = {"answer": "I started the agent.", "tools_used": ["start_agent_task"], "files": []}
ERRORED = {"answer": "LLM API error", "tools_used": ["web_search"], "files": []}

check("a factual answer is cacheable", _cacheable(FACTUAL, "what is filtering theory?"), True)
check("a personal reply is not cacheable", _cacheable(CHATTY, "i feel rough"), False)
check("an answer with a file is not cacheable", _cacheable(WITH_FILE, "plot it"), False)
check("a delegated answer is not cacheable", _cacheable(DELEGATED, "write a report"), False)
check("an error is not cacheable", _cacheable(ERRORED, "what is x"), False)
check("a question about today is not cacheable",
      _cacheable(FACTUAL, "what is the share price today"), False)

print()
print("=" * 70)
print("THE CACHE IN THE BOT")
print("=" * 70)

import tools  # noqa: E402

calls = {"n": 0}


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload

    @property
    def status_code(self):
        return 200

    @property
    def text(self):
        return "ok"


async def counting_post(self, *a, **kw):
    calls["n"] += 1
    return FakeResponse({"data": {"webPages": {"value": [
        {"name": "France", "url": "https://example.org/fr",
         "text": "Paris is the capital."},
    ]}}})


_real_post = tools.httpx.AsyncClient.post
tools.httpx.AsyncClient.post = counting_post
tools.LANGSEARCH_API_KEY = "test-key"
CACHES.clear("web_search")

result_a = asyncio.run(tools.web_search("what is the capital of France"))
calls_after_first = calls["n"]
result_b = asyncio.run(tools.web_search("what is the capital of France"))
calls_after_second = calls["n"]
result_c = asyncio.run(tools.web_search("what is the capital of France?"))
calls_after_third = calls["n"]

print(f"  search calls: {calls_after_first} then {calls_after_second} then {calls_after_third}")
check("the second search made no call", calls_after_second, calls_after_first)
check("the third search made no call", calls_after_third, calls_after_first)
check("both searches returned the same thing", result_a, result_b)
check_true("the search actually found something", "Paris" in result_a)

# A different question is a different key.
asyncio.run(tools.web_search("what is the capital of Spain"))
check("a different question is not cached", calls["n"], calls_after_third + 1)

# A failure is not cached: the next call tries again.
async def failing_post(self, *a, **kw):
    calls["n"] += 1
    raise RuntimeError("network is down")


tools.httpx.AsyncClient.post = failing_post
CACHES.clear("web_search")
before_fail = calls["n"]
fail_a = asyncio.run(tools.web_search("a question that fails"))
fail_b = asyncio.run(tools.web_search("a question that fails"))
check_true("a failure is reported", "error" in fail_a.lower())
check("a failure is not cached, so it is retried", calls["n"] - before_fail, 2)

tools.httpx.AsyncClient.post = _real_post

print()
print("=" * 70)
print(f"{PASSED}/{PASSED + len(FAILED)} checks passed")
if FAILED:
    for f in FAILED:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("ALL GREEN")
