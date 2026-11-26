"""Choosing the model that answers, with !model.

The command is small, but the wiring around it is what matters: a model
somebody chose has to go first in the chain, the others have to stay
behind it as fallbacks, and the choice has to be per room - two rooms
can want different models, and a room that never chose one has to get
the default chain back.

Nothing here touches the network: the storage is stubbed out, and the
Discord objects are fakes.

    python tests/test_model_command.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _env import ROOT  # noqa: E402

sys.path.insert(0, ROOT)

import main as main_module  # noqa: E402
from config import (GROQ_MODEL, MINIMAX_MODEL, MODEL_CHOICES,  # noqa: E402
                    MODEL_LABELS, BOLT_MODEL, chain_with_model,
                    groq_client, provider_for_model, provider_chain)

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


CHANNEL_ID = "1569657875368902666"


class FakeChannel:
    id = int(CHANNEL_ID)
    name = "model"


class FakeCtx:
    def __init__(self, channel_id=CHANNEL_ID):
        self.channel = FakeChannel()
        self.channel.id = channel_id
        self.sent = []

    async def send(self, text):
        self.sent.append(text)


async def run_tests():
    print("=" * 70)
    print("THE MODELS ON OFFER")
    print("=" * 70)

    check("every model has a name to show",
          sorted(MODEL_LABELS), sorted(set(MODEL_CHOICES) - {"nemotron"}))
    check_true("super is the strong one",
               "super" in MODEL_CHOICES and "super" in MODEL_LABELS)
    check_true("minimax is on offer",
               "minimax" in MODEL_CHOICES and "minimax" in MODEL_LABELS)
    check_true("groq is on offer",
               "groq" in MODEL_CHOICES and "groq" in MODEL_LABELS)
    check_true("bolt is on offer",
               "bolt" in MODEL_CHOICES and "bolt" in MODEL_LABELS)

    print()
    print("=" * 70)
    print("THE CHAIN")
    print("=" * 70)

    # The default: Nemotron first, then MiniMax, then Groq. The order is
    # the whole point of the chain - the strongest model goes first.
    default = [entry[2] for entry in provider_chain()]
    check_true("nemotron goes first", default[0].startswith("nvidia/"))
    check_true("minimax is behind it", MINIMAX_MODEL in default)
    check("groq is the last resort", GROQ_MODEL in default,
          groq_client is not None)

    # A chosen model goes first, and the rest stay behind it as
    # fallbacks - so choosing a model changes which one answers, not
    # whether there is a backup when it is down.
    chosen = [entry[2] for entry in chain_with_model(MINIMAX_MODEL)]
    check("the chosen model goes first", chosen[0], MINIMAX_MODEL)
    check("the rest stay as fallbacks", sorted(chosen), sorted(default))

    # Bolt is not in the default chain - it is the small model the
    # proactive scan uses - but somebody can still pick it to answer.
    chosen = [entry[2] for entry in chain_with_model(BOLT_MODEL)]
    check("bolt can go first too", chosen[0], BOLT_MODEL)
    check("the rest still stand behind it",
          sorted(chosen[1:]), sorted(default))

    check("an unknown model is not a provider", provider_for_model("nope"), None)
    check("an unknown model changes nothing",
          [e[2] for e in chain_with_model("nope")], default)

    print()
    print("=" * 70)
    print("THE COMMAND")
    print("=" * 70)

    stored = {}

    async def fake_set(channel_id, model):
        stored[channel_id] = model

    async def fake_get(channel_id):
        return stored.get(channel_id, "")

    async def fake_clear(channel_id):
        stored.pop(channel_id, None)

    old = (main_module.db_set_room_model, main_module.db_get_room_model,
           main_module.db_clear_room_model)
    main_module.db_set_room_model = fake_set
    main_module.db_get_room_model = fake_get
    main_module.db_clear_room_model = fake_clear
    main_module.ROOM_MODELS.clear()
    try:
        callback = main_module.model_cmd.callback

        # With no argument it lists the choices and says which one is in
        # use in this room.
        ctx = FakeCtx()
        await callback(ctx, "")
        check_true("it lists the choices",
                   "minimax" in ctx.sent[0] and "auto" in ctx.sent[0])

        # Choosing one stores the model id for this room.
        ctx = FakeCtx()
        await callback(ctx, "minimax")
        check("the choice is stored", stored.get(CHANNEL_ID), MINIMAX_MODEL)
        check_true("it says what it chose",
                   MINIMAX_MODEL in ctx.sent[0])

        # The listing then marks the model in use.
        ctx = FakeCtx()
        await callback(ctx, "")
        check_true("the listing marks the model in use",
                   "<- in use" in ctx.sent[0])

        # A room is a room: another room has its own choice.
        other = FakeCtx(channel_id="999")
        await callback(other, "groq")
        check("another room has its own choice",
              stored.get("999"), GROQ_MODEL)
        check("the first room keeps its choice",
              stored.get(CHANNEL_ID), MINIMAX_MODEL)

        # auto puts the default chain back.
        ctx = FakeCtx()
        await callback(ctx, "auto")
        check_true("auto clears the choice", CHANNEL_ID not in stored)

        # A model nobody has heard of is not an error, it is a list.
        ctx = FakeCtx()
        await callback(ctx, "gpt5")
        check_true("an unknown model is refused",
                   "do not know that model" in ctx.sent[0])
        check_true("the refusal lists the real ones",
                   "minimax" in ctx.sent[0])
    finally:
        (main_module.db_set_room_model, main_module.db_get_room_model,
         main_module.db_clear_room_model) = old

    print()
    print("=" * 70)
    print(f"{PASSED} passed, {len(FAILED)} failed")
    print("=" * 70)
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        sys.exit(1)


asyncio.run(run_tests())
