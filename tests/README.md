# The tests

The bot talks to a real model and a real database, so the tests do too.
That means they need keys in the environment - never in the files.

```bash
export SUPABASE_KEY=... NVIDIA_API_KEY=... KAGGLE_API_TOKEN=...
python tests/run_all.py quick      # no network calls to a model
python tests/run_all.py            # everything
```

A `.env` file next to the bot is read too, so the keys do not have to be
exported by hand.

## What each one is for

| file | what it covers |
| --- | --- |
| `regress.py` | every module imports, the provider chain, no credential has leaked into the source, the file tools, the prompt, and the whole presence decision - the structural checks, the three states, the name collisions, the final pass |
| `test_guard.py` | the reply sanitiser: the bot-speak, the prompt echoes, the closers, and that the substance of an answer is never touched |
| `test_cache.py` | the caches: the TTLs, the single-flight, that a failure is not cached |
| `test_handler.py` | the `on_message` handler itself, driven with fake Discord objects: the shape reaches the agent, a message meant for the person next to it never gets that far, and an unfinished message is held and then answered |
| `test_addresses.py` | the name check: a name at the front of a message is an address, a name in the middle of a sentence is not, and the bot's short name clashes with somebody in the room |
| `test_decisions.py` | the classifier itself, on real messages from the room: answer, wait, or quiet |
| `live_probe_laguna.py` | a real end-to-end turn: a personal message in Vietnamese, a technical question that has to use a tool, the answer cache, and a passing remark that gets a short reply |
| `probe_tool_retry.py` | that a real job is looked up rather than answered from memory, every time |
| `probe_personal_deep.py` | that a message about somebody's day is *not* turned into a search, and that the reply is still warm |
| `decision_on_log.py` | the whole decision over the real channel history, with what the bot actually did next to it |

`decision_on_log.py` needs `DISCORD_TOKEN` and a copy of the channel
history. It reads one from `CHANNEL_LOG` (default
`/tmp/opencode/channel_log.json`) and fetches it from Discord if it is
not there.

## The one that matters most

`decision_on_log.py` is the honest test of the presence decision. Every
other one checks that a piece works; this one puts a hundred real
messages from the room through the real decision and prints what the bot
did next to what the new code would do. The differences are the point.
