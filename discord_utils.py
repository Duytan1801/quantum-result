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

    Unless `code_block` is set, LaTeX math is rendered: inline math becomes
    Unicode text and display math becomes an image attachment.
    """
    images = []
    if render_math and not code_block:
        from mathrender import process_math

        render_dir = tempfile.mkdtemp(prefix="eq_")
        try:
            content, images = process_math(content, render_dir, MAX_EQUATION_IMAGES)
        except Exception:
            # Never let rendering break delivery.
            pass

    try:
        # Reserve room for the continuation marker so no message exceeds the limit.
        marker_room = 16
        footer_text = f"\n-# {footer}" if footer else ""
        footer_room = len(footer_text) + 4

        if code_block:
            chunks = split_message(content, DISCORD_MAX_LEN - marker_room - 10 - footer_room)
            formatted = [
                f"```\n{chunk}{f'\n… ({i + 1}/{len(chunks)})' if len(chunks) > 1 else ''}\n```"
                for i, chunk in enumerate(chunks)
            ]
        else:
            chunks = split_message(content, DISCORD_MAX_LEN - marker_room - footer_room)
            formatted = [
                f"{chunk}{f'… ({i + 1}/{len(chunks)})' if len(chunks) > 1 else ''}"
                for i, chunk in enumerate(chunks)
            ]

        # The footer goes on the last chunk so the whole reply reads as one unit.
        if footer_text and formatted:
            formatted[-1] = formatted[-1] + footer_text

        for i, text in enumerate(formatted):
            if i == 0 and hasattr(target, "reply"):
                await target.reply(text)
            elif hasattr(target, "send"):
                await target.send(text)
            else:
                # A Context has .send; a Message has .channel.send.
                await target.channel.send(text)

        # Attach any rendered equations.
        if images:
            sender = target.send if hasattr(target, "send") else target.channel.send
            for path in images:
                try:
                    import discord
                    await sender(file=discord.File(path))
                except Exception:
                    pass
    finally:
        if images:
            shutil.rmtree(os.path.dirname(images[0]), ignore_errors=True)
