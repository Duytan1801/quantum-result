"""Discord message helpers: splitting long replies and sending them."""

import os
import shutil
import tempfile

# Discord rejects messages over 2000 characters, so long replies are split
# into several messages instead of being cut off mid-sentence.
DISCORD_MAX_LEN = 2000

# How many rendered equation images to attach to one message.
MAX_EQUATION_IMAGES = 5


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
                   render_math: bool = True):
    """Send content to a channel/context, splitting it across messages if needed.

    `target` may be a CommandContext (uses .send) or a Message (first chunk is
    a reply, continuations are plain channel messages).

    Unless `code_block` is set, LaTeX math is rendered: display equations
    become images attached next to the text they belong to, and inline maths
    becomes Unicode.
    """
    import discord

    render_dir = None
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
        # Reserve room for the continuation marker so no message exceeds the limit.
        marker_room = 16
        footer_text = f"\n-# {footer}" if footer else ""
        footer_room = len(footer_text) + 4

        # Split each text section into Discord-sized chunks, keeping images in
        # place between the sections they came from.
        outgoing = []
        for kind, value in parts:
            if kind == "image":
                outgoing.append(("image", value))
                continue
            if code_block:
                chunks = split_message(value, DISCORD_MAX_LEN - marker_room - 10 - footer_room)
                for chunk in chunks:
                    outgoing.append(("text", f"```\n{chunk}\n```"))
            else:
                chunks = split_message(value, DISCORD_MAX_LEN - marker_room - footer_room)
                for chunk in chunks:
                    outgoing.append(("text", chunk))

        # Number the text messages so a split reply reads "(2/3)".
        text_positions = [i for i, (k, _) in enumerate(outgoing) if k == "text"]
        total_text = len(text_positions)
        for n, i in enumerate(text_positions, 1):
            chunk = outgoing[i][1]
            if total_text > 1:
                chunk += f"… ({n}/{total_text})"
            outgoing[i] = ("text", chunk)

        # The footer goes on the last text message.
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
            if first and hasattr(target, "reply"):
                await target.reply(value)
            else:
                await _send(target, value)
            first = False
    finally:
        if render_dir:
            shutil.rmtree(render_dir, ignore_errors=True)


async def _send(target, content=None, **kwargs):
    """Send to a context, channel or message."""
    if hasattr(target, "send") and not hasattr(target, "channel"):
        return await target.send(content, **kwargs)
    if hasattr(target, "send"):
        return await target.send(content, **kwargs)
    return await target.channel.send(content, **kwargs)
