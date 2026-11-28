"""Discord message helpers: splitting long replies and sending them."""

import os
import shutil
import tempfile

from people import address_mentions, neutralise_mass_mentions

# Discord rejects messages over 2000 characters, so long replies are split
# into several messages instead of being cut off mid-sentence.
DISCORD_MAX_LEN = 2000

# How many rendered equation images to attach per reply. Math-heavy answers
# were silently degrading every equation past the 5th into plain text, which
# is where the mangled "tE = -4/3" in #model came from.
MAX_EQUATION_IMAGES = 12


def fit_discord(text: str, limit: int = DISCORD_MAX_LEN) -> str:
    """Shorten text so it fits in a single Discord message.

    Discord rejects anything over 2000 characters with "Invalid Form Body",
    so anything that must be one message goes through here.
    """
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:max(0, limit - 1)].rstrip() + "…"


def split_message(text: str, max_len: int = DISCORD_MAX_LEN) -> list[str]:
    """Split text into Discord-sized chunks, preferring natural break points.

    Tries to break on a paragraph, then a line, then a word boundary, so we
    don't cut words in half. Very long unbroken tokens (e.g. a base64 blob)
    are hard-split as a last resort.
    """
    if len(text) <= max_len:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > max_len:
        window = remaining[:max_len]

        # Prefer a paragraph break, then a newline, then a space.
        cut = window.rfind("\n\n")
        if cut < max_len // 2:
            cut = window.rfind("\n")
        if cut < max_len // 2:
            cut = window.rfind(" ")
        if cut < max_len // 2:
            cut = max_len  # no usable break: hard split

        # Don't leave a tiny fragment behind.
        if cut < 1:
            cut = max_len

        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip("\n")

    if remaining:
        chunks.append(remaining)

    return chunks


async def send_long(target, content: str, code_block: bool = False, footer: str = None,
                   render_math: bool = True, allowed_mentions=None):
    """Send content to a channel/context, splitting it across messages if needed.

    `target` may be a CommandContext (uses .send) or a Message (first chunk is
    a reply, continuations are plain channel messages).

    Unless `code_block` is set, LaTeX math is rendered: display equations
    become images attached next to the text they belong to, and inline maths
    becomes Unicode.

    `allowed_mentions` overrides who may be pinged. Without it, the tags in
    the message decide: a tag that addresses somebody pings them, and
    nothing else pings anybody - the server-wide mentions are taken out of
    the message before it is sent.

    Returns the list of messages that were sent, so the caller can remember
    that they were the bot's own (a reply to one of them is for the bot).
    """
    import discord

    sent = []
    render_dir = None

    # Nothing to say. Discord rejects an empty message, and sending one
    # would be worse than saying nothing at all - the guard can empty a
    # reply when the whole of it was scaffolding.
    if not (content or "").strip():
        return sent

    # The tags. A tag at the front of the message is an address and pings;
    # one further down is somebody being talked about, and is written out
    # as a name. The server-wide pings never survive.
    content, pings = address_mentions(content)
    content = neutralise_mass_mentions(content)
    if allowed_mentions is None:
        # The ids have to be objects with an ".id" - discord.py reads the
        # attribute, not the value, so a list of plain strings raises
        # when the message is sent.
        allowed_mentions = discord.AllowedMentions(
            users=[discord.Object(id=int(user_id))
                   for user_id in pings if str(user_id).isdigit()],
            roles=False, everyone=False, replied_user=True)

    parts = [("text", content)]
    if render_math and not code_block:
        from mathrender import process_math

        render_dir = tempfile.mkdtemp(prefix="eq_")
        try:
            parts = process_math(content, render_dir, MAX_EQUATION_IMAGES)
        except Exception:
            # Never let rendering break delivery.
            parts = [("text", content)]

    try:
        footer_text = f"\n-# {footer}" if footer else ""
        footer_room = len(footer_text) + 4

        # Join neighbouring text sections back together.
        #
        # process_math cuts the message apart wherever an equation was. When
        # an equation cannot be drawn it becomes text too, so without this the
        # answer arrives as a dozen unrelated fragments instead of one
        # message - and, worse, a blank fragment in between, which Discord
        # rejects outright.
        merged = []
        for kind, value in parts:
            if kind == "text" and merged and merged[-1][0] == "text":
                merged[-1] = ("text", merged[-1][1] + value)
            else:
                merged.append((kind, value))

        # Split each text section into Discord-sized chunks, keeping images in
        # place between the sections they came from.
        outgoing = []
        for kind, value in merged:
            if kind == "image":
                outgoing.append(("image", value))
                continue
            if code_block:
                chunks = split_message(value, DISCORD_MAX_LEN - 10 - footer_room)
                for chunk in chunks:
                    outgoing.append(("text", f"```\n{chunk}\n```"))
            else:
                chunks = split_message(value, DISCORD_MAX_LEN - footer_room)
                for chunk in chunks:
                    outgoing.append(("text", chunk))

        # The footer goes on the last text message.
        text_positions = [i for i, (k, _) in enumerate(outgoing) if k == "text"]
        if footer_text and text_positions:
            i = text_positions[-1]
            outgoing[i] = ("text", outgoing[i][1] + footer_text)

        first = True
        for kind, value in outgoing:
            if kind == "image":
                try:
                    await _send(target, file=discord.File(value))
                except Exception as exc:
                    await _send(target, f"⚠️ Could not attach {os.path.basename(value)}: {exc}")
                continue

            # Discord rejects a message with no visible content:
            #   HTTP 400 {"message": "Cannot send an empty message"}
            # A blank chunk used to raise out of this function, which threw
            # away every remaining part of the answer - so a reply stopped
            # halfway, with no footer and no explanation.
            if not (value or "").strip():
                continue

            try:
                if first and hasattr(target, "reply"):
                    sent.append(await target.reply(value,
                                                   allowed_mentions=allowed_mentions))
                else:
                    sent.append(await _send(target, value,
                                            allowed_mentions=allowed_mentions))
            except Exception as exc:
                # One message that Discord will not accept must not cost the
                # user the rest of the answer.
                print(f"could not send a message part: {exc}")
                try:
                    sent.append(await _send(
                        target, f"⚠️ A part of this reply failed to send: {exc}",
                        allowed_mentions=allowed_mentions))
                except Exception:
                    pass
            first = False
    finally:
        if render_dir:
            shutil.rmtree(render_dir, ignore_errors=True)

    return sent


async def _send(target, content=None, **kwargs):
    """Send to a context, channel or message."""
    if hasattr(target, "send"):
        return await target.send(content, **kwargs)
    return await target.channel.send(content, **kwargs)
