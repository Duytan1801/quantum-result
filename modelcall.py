"""One call to a model, with the answer that never came asked for again.

Some reasoning models spend the whole output on thinking and finish
without saying anything. The request succeeds - the endpoint has no
complaint to make - the reasoning comes back, the answer does not, and
the stop reason is an ordinary end-of-turn.

Nothing in the response says "this went wrong", so the only way to
notice is to look: no answer, but some thinking. That is what this
module is for. The call is made again with a line asking for the answer
and nothing else, and the second answer is the one that is used.

The step-5 model does this more readily than the others, and no flag
turns its thinking off - the endpoint takes `enable_thinking`,
`thinking` and `reasoning` as request options, and ignores all three -
so the retry is the only thing that can be done about it.
"""

from config import extra_body_for

# What the second attempt says. It is short on purpose: the model has
# already thought about the question, and the thing missing is the
# answer, not more thinking.
NUDGE = ("You thought it through and did not answer. Do not think it "
         "through again - give the answer now, in a few sentences.")


async def _one(client, model, messages, *, provider, thinking, max_tokens,
               temperature, top_p, extra_body, **kwargs):
    """The call itself: what the model said, and what it thought."""
    body = extra_body if extra_body is not None else extra_body_for(
        provider, thinking=thinking)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=top_p,
        extra_body=body,
        **kwargs,
    )
    message = response.choices[0].message
    content = (message.content or "").strip()
    reasoning = (getattr(message, "reasoning_content", "") or "").strip()
    return content, reasoning


async def complete(client, model, messages, *, provider: str = "NVIDIA",
                   thinking: bool = True, max_tokens: int = 1024,
                   temperature: float = 0.6, top_p: float = 0.95,
                   extra_body=None, retries: int = 1, **kwargs) -> str:
    """The answer, with a second try when the model reasoned and said nothing.

    Returns the answer, or "" if there was none. An exception from the
    endpoint is raised as usual - the caller decides whether that is
    worth waiting out.
    """
    content, reasoning = await _one(
        client, model, messages, provider=provider, thinking=thinking,
        max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        extra_body=extra_body, **kwargs)

    attempts = 0
    while not content and reasoning and attempts < retries:
        # It thought and said nothing. Ask again, and say what is wanted.
        attempts += 1
        ask = list(messages) + [{"role": "user", "content": NUDGE}]
        try:
            content, reasoning = await _one(
                client, model, ask, provider=provider, thinking=thinking,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                extra_body=extra_body, **kwargs)
        except Exception:
            break
        if content:
            print(f"the model reasoned without answering ({model}) - "
                  f"asked again and got an answer")
    return content
