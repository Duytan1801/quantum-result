"""Which failures are worth waiting out.

A provider that says "not now" is different from a provider that says "no".
The first is a rate limit, a service that is full, a connection that
dropped: it clears on its own, and the right thing to do is to wait and
ask again. The second is a bad key or a model that does not exist: waiting
does not help, and the answer to that is to say so.

So this is the one place that decides, and everything that makes a call -
the answer path, the presence scanner - asks it first.

The status code decides it where there is one. The message decides it
where there is not, because a busy service is sometimes reported as a bad
request with "the model is overloaded" in the text - and that is exactly
the case worth waiting out.
"""

import asyncio

# The codes that mean "come back in a moment".
TRANSIENT_STATUS = frozenset({
    408,   # request timeout
    409,   # conflict
    425,   # too early
    429,   # rate limited
    500,   # internal server error
    502,   # bad gateway
    503,   # service unavailable / full
    504,   # gateway timeout
    529,   # overloaded
})

# The phrases that mean the same thing when the code does not say it.
TRANSIENT_TEXT = (
    "rate limit", "too many requests", "rate_limited", "overloaded",
    "service full", "server is full", "no capacity", "capacity",
    "all servers are busy", "server is busy", "service unavailable",
    "temporarily unavailable", "try again later", "please try again",
    "connection error", "connection reset", "connection aborted",
    "server disconnected", "remote end closed", "timed out", "timeout",
    "incomplete chunked read", "read timeout",
)


def status_of(exc: Exception) -> int | None:
    """The HTTP status an exception carries, if it carries one."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def is_transient(exc: Exception) -> bool:
    """Is this failure one that a wait can fix?"""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True

    text = str(exc).lower()
    status = status_of(exc)

    if status is not None and status in TRANSIENT_STATUS:
        return True

    # The message decides it where the code does not.
    if any(marker in text for marker in TRANSIENT_TEXT):
        return True

    # Any other code is the provider saying no, not saying not now.
    return False
