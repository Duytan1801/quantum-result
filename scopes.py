"""Where a turn reads from, and where it writes to.

The bot's memory is a set of scopes. There is one per person - the things
that are true about them wherever they are - and one per room - the things
that were said in that room. A reply reads the scopes of the room it is in,
so a room only ever knows what was said in it.

That is the right default, and it is what keeps one room's conversation out
of another room's answers.

But some rooms need to know more than their own. A private room may need
the whole picture - every channel, every person - to do its job. What it
learns there, though, must not travel: the memories that come out of such
a room stay in it, and no other room's reply ever loads them.

That is a detached room: it reads the whole, and writes only itself.

The two halves have to be kept apart, and they used to be the same list.
A memory learned in a shared room was written to every scope that was
read, which is right - the room and the person both know it - but it means
the write list has to be worked out on its own. A detached room reads
everything and writes one scope; a normal room reads two and writes two.

This module is the one place that works this out, so the message handler,
the slash commands and the background jobs all agree.
"""

from dataclasses import dataclass, field

from config import is_detached
from presence import channel_scope


@dataclass
class ScopePlan:
    """The scopes one turn may read, and the ones it may write.

    `read` is what the reply is allowed to know: the room's own scope,
    the person's own scope, and - for a detached room - everything the
    bot has ever stored.

    `write` is where the new rows go. This is the list of scopes a memory
    learned in this turn is stored under. For a normal room it matches
    `read`; for a detached room it is its own scope and nothing else.

    `primary` is the scope the conversation itself is written to - the one
    the next turn in this room will read as "this room".
    """

    read: list[str] = field(default_factory=list)
    write: list[str] = field(default_factory=list)
    primary: str = ""
    shared: bool = False
    detached: bool = False

    def __iter__(self):
        """So the old (scopes, write_scope, shared) shape still works."""
        return iter((self.read, self.primary, self.shared))


async def scope_plan(channel_id, user_id, tracker=None, is_dm: bool = False) -> ScopePlan:
    """Work out where this turn reads from and writes to.

    `tracker` is the ChannelTracker, which knows whether a room is a room
    (two or more people in it) or a private chat. It is passed in rather
    than imported, because the tracker lives with the bot.

    A detached room is always a room, whatever the tracker says. That
    matters: if it looked like a private chat, the turn would be written
    to the person's own scope, and the person's own scope is read in
    every other room. The whole point is that it does not travel.
    """
    person = str(user_id or "")
    own = channel_scope(channel_id) if channel_id else ""

    if is_detached(channel_id):
        # Read the whole picture. The scopes are whatever the bot has
        # stored, plus this room and this person, so a room that has not
        # been written to yet still reads properly.
        try:
            from storage import db_all_scopes

            everything = await db_all_scopes()
        except Exception:
            everything = []
        read = []
        for scope in list(everything) + [own, person]:
            if scope and scope not in read:
                read.append(scope)
        # Write only here. Nothing learned in this room leaves it.
        return ScopePlan(read=read, write=[own] if own else [],
                         primary=own or person, shared=True, detached=True)

    # An ordinary room or a private chat.
    read = []
    if person:
        read.append(person)
    shared = False
    if tracker is not None and not is_dm:
        scope = tracker.scope_for(channel_id)
        if scope:
            read.append(scope)
            shared = True

    primary = own if (shared and own) else person
    return ScopePlan(read=read, write=list(read), primary=primary,
                     shared=shared, detached=False)
