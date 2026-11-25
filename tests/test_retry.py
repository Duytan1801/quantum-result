"""What happens when a turn produces no answer.

Three things can go wrong between the question and the reply, and none of
them is an answer:

  * the provider says "not now" - a 429, a service that is full
  * the stream died half way through
  * the model finished the step without saying anything

The first two clear on their own, so the turn keeps trying, five seconds
apart, for as long as it takes. The third is tried a few times. A failure
that no wait can fix - a bad key, a model that does not exist - is given
up on at once.

    python tests/test_retry.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT  # noqa: E402

sys.path.insert(0, ROOT)

import agent  # noqa: E402
from config import ANSWER_RETRY_ATTEMPTS, ANSWER_RETRY_DELAY  # noqa: E402

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


# --- the fakes --------------------------------------------------------------

class APIError(Exception):
    """An error that carries a status code, like the ones the SDK raises."""

    def __init__(self, status_code, message="failed"):
        super().__init__(message)
        self.status_code = status_code


class _Delta:
    def __init__(self, content=""):
        self.content = content
        self.tool_calls = None
        self.reasoning_content = None


class _Choice:
    def __init__(self, content="", finish_reason=None):
        self.delta = _Delta(content)
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, content="", finish_reason=None):
        self.choices = [_Choice(content, finish_reason)]


def a_stream(text):
    async def gen():
        yield _Chunk(text, finish_reason="stop")
    return gen()


class _Completions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if not self.outcomes:
            raise RuntimeError("the fake ran out of outcomes")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class _Client:
    def __init__(self, outcomes):
        self.completions = _Completions(outcomes)
        self.chat = _Chat(self.completions)


class _Clock:
    """Records the waits instead of making the test slow."""

    def __init__(self):
        self.waits = []


async def run(outcomes, messages=None):
    """Run the loop against a fake provider, and say what happened."""
    client = _Client(outcomes)
    clock = _Clock()

    old_chain = agent.provider_chain
    old_sleep = agent.asyncio.sleep
    old_delay = agent.ANSWER_RETRY_DELAY

    async def fake_sleep(seconds):
        clock.waits.append(seconds)

    agent.provider_chain = lambda: [("NVIDIA", client, "fake-model")]
    agent.asyncio.sleep = fake_sleep
    agent.ANSWER_RETRY_DELAY = 5.0
    try:
        result = await agent.run_agent_loop(
            messages or [{"role": "user", "content": "hello"}],
            user_id=None, channel=None, thinking=False)
    finally:
        agent.provider_chain = old_chain
        agent.asyncio.sleep = old_sleep
        agent.ANSWER_RETRY_DELAY = old_delay
    return result, client.completions.calls, clock


async def main():
    print("=" * 70)
    print("THE SETTINGS")
    print("=" * 70)

    check("the wait is five seconds", ANSWER_RETRY_DELAY, 5.0)
    check("an answer-less turn is tried again three times",
          ANSWER_RETRY_ATTEMPTS, 3)

    print()
    print("=" * 70)
    print("WHAT COUNTS AS WORTH WAITING OUT")
    print("=" * 70)

    check("a rate limit is worth waiting out",
          agent.is_transient(APIError(429, "rate limit exceeded")), True)
    check("a full service is worth waiting out",
          agent.is_transient(APIError(503, "service unavailable")), True)
    check("an overloaded model is worth waiting out",
          agent.is_transient(APIError(400, "the model is overloaded")), True)
    check("a dropped connection is worth waiting out",
          agent.is_transient(RuntimeError("Connection error.")), True)
    check("a timeout is worth waiting out",
          agent.is_transient(asyncio.TimeoutError()), True)
    check("a bad key is not worth waiting out",
          agent.is_transient(APIError(401, "invalid api key")), False)
    check("a model that does not exist is not worth waiting out",
          agent.is_transient(APIError(404, "model not found")), False)
    check("a broken request is not worth waiting out",
          agent.is_transient(APIError(400, "context length exceeded")), False)

    print()
    print("=" * 70)
    print("THE PROVIDER SAYS NOT NOW")
    print("=" * 70)

    # More failures than the number of tries an answer-less turn gets:
    # a provider that says "not now" is not given up on.
    result, calls, clock = await run(
        [APIError(429, "rate limit exceeded")] * 5 + [a_stream("here you go")])
    check("the call is made again after every failure", calls, 6)
    check("it waits five seconds between the tries",
          clock.waits, [5.0] * 5)
    check("the answer that finally came is used",
          result["answer"], "here you go")

    result, calls, clock = await run(
        [APIError(503, "service full")] * 3 + [a_stream("at last")])
    check("a full service is waited out too", result["answer"], "at last")
    check("three waits for three tries", clock.waits, [5.0, 5.0, 5.0])

    result, calls, clock = await run([APIError(401, "invalid api key")])
    check("a bad key is not tried again", calls, 1)
    check("a bad key is not waited out", clock.waits, [])
    check("the turn gives up with an error", result["answer"], "LLM API error")

    result, calls, clock = await run([APIError(404, "model not found")])
    check("a missing model is not tried again", calls, 1)

    print()
    print("=" * 70)
    print("THE MODEL SAID NOTHING")
    print("=" * 70)

    result, calls, clock = await run([a_stream("")] * 4)
    check("an empty answer is tried again and again", calls, 4)
    check("it waits five seconds between the tries",
          clock.waits, [5.0, 5.0, 5.0])
    check("the turn gives up rather than going for ever",
          result["answer"], "(no response)")

    result, calls, clock = await run([a_stream(""), a_stream("at last")])
    check("the second answer is used", result["answer"], "at last")
    check("one wait for one retry", clock.waits, [5.0])

    print()
    print("=" * 70)
    print(f"{PASSED} passed, {len(FAILED)} failed")
    print("=" * 70)
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        sys.exit(1)


asyncio.run(main())
