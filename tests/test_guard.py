"""Tests for the guard and the relationship impressions.

The guard is the last thing a reply passes through, so the dangerous cases
are the ones where it changes a good answer. Each of those is tested
explicitly: the substance of a reply must survive it.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT, need  # noqa: E402

sys.path.insert(0, ROOT)

need("SUPABASE_KEY")


from guard import sanitize_reply  # noqa: E402

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
print("THE GUARD: what it removes")
print("=" * 70)

check("the AI disclaimer goes",
      sanitize_reply("I'm just an AI, so I can't help with that."),
      "I can't help with that.")

check("the disclaimer goes mid-sentence",
      sanitize_reply("I cannot do that. I'm just an AI, sorry."),
      "I cannot do that. sorry.")

check("a plain refusal survives",
      sanitize_reply("I don't have access to that account."),
      "I don't have access to that account.")

check("a false capability claim is removed whole",
      sanitize_reply("I can't access Discord from here."),
      "")

check("the 'hope this helps' closer goes",
      sanitize_reply("The answer is 42. I hope this helps!"),
      "The answer is 42.")

check("the 'let me know' closer goes",
      sanitize_reply("Here is the result.\n\nLet me know if you need anything else!"),
      "Here is the result.")

check("the 'here is' frame goes",
      sanitize_reply("Here's a helpful response: the value is 42."),
      "the value is 42.")

check("the sycophancy opener goes",
      sanitize_reply("Great question! The answer is 42."),
      "The answer is 42.")

check("the Vietnamese closer goes",
      sanitize_reply("Kết quả là 42. Hy vọng điều này giúp ích!"),
      "Kết quả là 42.")

check("the Vietnamese praise goes",
      sanitize_reply("Câu hỏi hay! Đáp án là 42."),
      "Đáp án là 42.")

check("the prompt echo goes",
      sanitize_reply("TALKING - the default. Somebody is telling you how they are."),
      "")

check("the status line goes when there is a real answer",
      sanitize_reply("Let me look that up.\n\nThe answer is 42, and here is why it matters."),
      "The answer is 42, and here is why it matters.")

check("the status line stays when it is all there is",
      sanitize_reply("Let me look that up."),
      "Let me look that up.")

print()
print("=" * 70)
print("THE GUARD: what it must NOT touch")
print("=" * 70)

check("a plain answer is untouched",
      sanitize_reply("42"),
      "42")

check("code survives",
      sanitize_reply("```python\nprint('hi')\n```"),
      "```python\nprint('hi')\n```")

check("numbers survive",
      sanitize_reply("Anthropic's float is 7.4%, and the IPO priced at $83."),
      "Anthropic's float is 7.4%, and the IPO priced at $83.")

check("a quote containing 'I hope' survives",
      sanitize_reply('She said "I hope you feel better", and I agreed.'),
      'She said "I hope you feel better", and I agreed.')

check("a Vietnamese reply is untouched",
      sanitize_reply("Mình ổn, cảm ơn cậu."),
      "Mình ổn, cảm ơn cậu.")

check("a long answer keeps its shape",
      sanitize_reply("Here are three things:\n1. First\n2. Second\n3. Third"),
      "Here are three things:\n1. First\n2. Second\n3. Third")

check("empty stays empty",
      sanitize_reply(""),
      "")

print()
print("=" * 70)
print("THE RELATIONSHIP IMPRESSIONS")
print("=" * 70)

from memory import build_impressions  # noqa: E402

JADE = {"id": "909453181916938301", "name": "𝐣𝐚𝐝𝐞_𝐦𝟖", "plain": "jade_m8"}
DAMIAN = {"id": "1499211380052721684", "name": "Lain", "plain": "Lain"}


async def impressions_probe():
    scopes = ["1499211380052721684", "channel:1569657875368902666"]
    text = await build_impressions(scopes, speaker=JADE,
                                   roster=[JADE, DAMIAN])
    print(text if text else "(no impressions)")
    print()
    return text


text = asyncio.run(impressions_probe())

check_true("the impressions name Jade", "jade" in text.lower())
check_true("the impressions mention the warmth instruction",
           "warm" in text.lower() or "gentle" in text.lower())
check_true("the impressions say not to say it out loud",
           "never say" in text.lower())
check_true("the impressions say not to name the relationship",
           "relationship" in text.lower())
check_true("the impressions say the bot is not Lain",
           "not lain" in text.lower())

print()
print("=" * 70)
print(f"{PASSED}/{PASSED + len(FAILED)} checks passed")
if FAILED:
    for f in FAILED:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("ALL GREEN")
