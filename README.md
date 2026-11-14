# Quantum Discord Bot

A Discord AI agent with code execution, web search, and persistent file storage.
Conversations are stored in Supabase so the bot remembers you between sessions.

## Features

| Capability | Provider |
|---|---|
| LLM (primary) | NVIDIA NIM — `nvidia/nemotron-3.5-super-vl-120b-a12b` |
| LLM (fallback) | Groq — `qwen/qwen3.8-27b` (used only if NVIDIA fails) |
| Math / science | WolframAlpha (derivatives, integrals, stats, conversions) |
| Code execution | Daytona sandbox (Python, JavaScript, TypeScript, Bash) |
| Web search | LangSearch (returns full page text, not just snippets) |
| History / facts | Supabase (`conversation_history`, `user_facts`, `chat_mode`) |
| Long-term memory | Supabase + pgvector, embedded with NVIDIA `nvidia/nemotron-3.5-embed-1b` |
| Math rendering | KaTeX + MathJax → PNG (via Node) |
| Diagrams | Excalidraw → PNG |
| Vision | Images attached to a message are read by the model |

The agent can call tools in a loop: it computes with WolframAlpha, runs code,
searches the web, verifies its own arithmetic, and reads/writes/edits files in
a sandboxed workspace before answering. Every reply ends with a small grey
footer naming the model and the tools that were used.

## Sourcing and verification

The bot is built to not answer from stale memory or from arithmetic it did in
its head.

**Always source.** Questions that could have a different answer today than last
year trigger a mandatory search — prices, financials, earnings, product
versions, who holds a role, anything "latest" or "current". The prompt requires
primary sources (filings, official announcements, the regulator) over coverage
of them, and dates every figure it quotes.

**Always recompute.** Derived numbers — growth rates, percentages, totals,
differences, ratios, sums — are never stated from the model's head. The
`verify_claims` tool recomputes the expression behind each figure with a
whitelisted AST evaluator and reports a pass/mismatch verdict, so the model
catches its own arithmetic errors before publishing them. If a source's printed
percentage disagrees with the recomputation, the bot trusts the recomputation
and flags the discrepancy.

```
verify_claims({"checks": [
  {"expression": "(127.6 - 90.2) / 90.2 * 100", "claimed": "41.5"}
]})
→ 1. OK  (127.6 - 90.2) / 90.2 * 100 = 41.4634 (claimed 41.5)
```

The evaluator only allows numbers, `+ - * / // % **`, parentheses, and
functions like `sqrt`/`log`/`abs` — no names, no attribute access, no calls
outside a whitelist.

## Diagrams

The bot can draw flowcharts, architecture sketches and concept maps, and send
them as images. It uses
[Excalidraw-Interface](https://pypi.org/project/Excalidraw-Interface/) to
build the sketch, which is then rendered to a PNG on Discord's dark background
and attached to the reply automatically.

The model only describes *what* the diagram says — the layout is computed from
each node's row and column, so it never has to work out coordinates:

```json
{
  "title": "Agent pipeline",
  "nodes": [
    {"id": "q", "text": "User question", "row": 0, "col": 0, "color": "blue"},
    {"id": "d", "text": "Needs current data?", "shape": "diamond", "row": 1, "col": 0},
    {"id": "s", "text": "web_search", "row": 2, "col": -1, "color": "violet"}
  ],
  "edges": [
    {"from": "q", "to": "d"},
    {"from": "d", "to": "s", "label": "yes"}
  ]
}
```

Shapes are `rectangle` (a step), `diamond` (a decision) and `ellipse` (a
start/end). Edges can carry a label, be dashed, or point both ways.

## Math rendering

Discord does not render LaTeX in chat, so the bot renders it for you:

| Input | Output |
|---|---|
| `$x^2$`, `$\alpha$`, `$\frac{1}{2}$` | Unicode text inline — `x²`, `α`, `1/2` |
| `$$...$$`, `\[...\]` | A rendered PNG attached next to the text it belongs to |

Display equations are typeset by **KaTeX** (which parses and validates the
LaTeX) and **MathJax** (which lays it out — `aligned`, `matrix`, `cases` and
friends all come out correctly), then rasterised to PNG on Discord's
dark-theme background (`#313338`) so they blend in rather than floating as a
white rectangle.

Environments are found whether or not they are wrapped in `$$...$$` or
`\[...\]`. Models routinely write a bare
```
\begin{aligned}
a &= b \\
c &= d
\end{aligned}
```
with no delimiters at all, and that used to be posted to Discord as raw
LaTeX. `aligned`, `align`, `align*`, `alignedat`, `gathered`, `split`,
`cases`, `eqnarray`, `multline`, `array` and the `matrix` family are all
handled. If KaTeX rejects an expression that MathJax understands
(`eqnarray`, `multline`, `align*`), MathJax gets the final say; if MathJax
reports an error too, the equation falls back to Unicode text instead of
being posted broken.

Images are interleaved with the text: an equation appears immediately after the
paragraph that introduces it, not piled up at the end. Each reply carries at
most 5 equation images; beyond that the remaining equations fall back to
Unicode text so a message never turns into an image dump.

Fenced code blocks are never touched — Python source containing `$` or
`\frac` is left exactly as written.

Requires Node.js plus `katex` and `mathjax-full` (see `render/`). If those
are missing the bot falls back to matplotlib's `mathtext`, which handles simple
expressions but not multi-line environments.

## Vision

Attach an image to a message and the bot looks at it. The attachment is
downloaded, turned into a data URL, and passed to the model as an image content
part alongside your text, so you can ask "what's wrong with this chart?" or
"summarise this screenshot" without leaving Discord.

Images are capped at 25 MB each (Discord's limit) and the total request stays
inside the model's context budget.

## Semantic memory

Two kinds of memory, working together:

- **Conversation history** — every exchange is stored in Supabase, so the bot
  remembers the thread of a conversation. `!remember` / `!recall` / `!forget`
  manage the small key–value fact list on top of that.
- **Long-term semantic memory** — durable facts are extracted from each
  exchange in the background, embedded, and stored as *atomic* memories (one
  self-contained sentence each). When you ask a question, the bot embeds the
  question and retrieves only the memories that are actually relevant, then
  injects them into the prompt.

Atomic memories beat one long summary: a question about SpaceX pulls the SpaceX
memory out of the noise instead of retrieving a blob that mentions it in
passing.

### How a memory is born

```
exchange ---> worth extracting? ---> extract (LLM) ---> embed ---> store
                   |
                   +--- no: stop (a greeting costs nothing)
```

Only exchanges that could hold a durable fact reach the extractor. A "hi",
an acknowledgement, or a one-word answer is skipped, so the background work
costs nothing on the vast majority of messages.

### How a memory is found

```
question ---> embed (cached) ---> cosine search ---> recency bias
                                              +--> revision rule
                                              +--> relevance floor
                                              +--> injected into the prompt
```

- **Recency** - the score blends similarity with how recent the memory is,
  so "I now work at X" (this morning) outranks "I worked at Y" (last year).
- **Revisions** - when two memories are about the same thing but say
  different things, only the newer one is kept. The comparison is between
  the memories themselves, not their scores against the question: measured
  on real memories, a revision pair sits at cosine similarity 0.83 while two
  unrelated facts sit at 0.63.
- **Relevance floor** - a memory that scores below 0.15 against the question
  is not injected. Measured over 15 real questions, correct matches scored
  0.33-0.68 while questions with no matching memory topped out at 0.14, so
  0.15 separates them cleanly and keeps irrelevant memories out of the
  prompt, where the model would treat them as fact.

### Versioning

Every row records which embedding model produced its vector, and its
dimension. Vectors from different models are not comparable, so retrieval
filters on the model - if `EMBED_MODEL` is ever changed, the old rows are
identified rather than silently mixed into the same vector space, and
`reembed_stale_memories()` puts them back into the new one.

Embeddings come from NVIDIA's API - `nvidia/nemotron-3.5-embed-1b`, 2048
dimensions. It was benchmarked against the other NVIDIA embedding models on
real memories from actual conversations: it matches the fastest model on
retrieval quality (R@1 0.933, MRR 0.943) and keeps a much better latency tail
than the 8B model, whose batch p90 was 30 seconds. Override with
`EMBED_MODEL` if you want to try another.

### Keeping it tidy

- Memories are upserted on `(user_id, content)`, so a fact learned twice is
  stored once.
- Retrieved memories are marked as used; the store is pruned to a cap
  (`MEMORY_CAP`, 2000 by default), dropping the least used and oldest first.
- `!forget <key>` deletes the key-value fact **and** any long-term memory
  that mentions it, so there is a way to get rid of something the bot
  remembers.
- Query embeddings are cached, so asking the same question twice only pays
  for the embedding once.

**One-time setup.** The memories table and its search function must exist
before memory works. Run [`supabase_memory.sql`](supabase_memory.sql) in the
Supabase SQL editor (Dashboard -> SQL Editor -> New query -> paste -> Run).
Until you do, the bot detects the missing table and runs without long-term
memory - no errors, it just does not remember anything across sessions. It
picks the table up on its own within 5 minutes of you running the SQL, no
restart needed. The bot prints `long-term memory: on` or `long-term memory:
OFF` when it starts, so you can tell which state you are in.

## Conversation history

The whole conversation is kept, not just the last few messages. Every turn is
stored in Supabase and pulled back on the next one, so the bot keeps the thread
of a long conversation instead of forgetting what it just said.

When the history outgrows the context budget, the oldest part is **summarised**
rather than dropped, and the summary is carried forward on later turns. The
summariser is told explicitly that it is reading a record of a past
conversation, not answering a question — otherwise it responds to the last
thing it read instead of summarising it.

The summary is incremental: the first time the history is too big it takes a
full pass, and after that only the messages that arrived since the last
summary are summarised and appended. That turns a 30-second step into a couple
of seconds, and it keeps the bot responsive once the conversation is long.

## Commands

| Command | Description |
|---|---|
| `!ask <question>` | Ask the agent anything |
| `!research <topic>` | Start a long background research job (up to 100 steps) |
| `!agent <name> <topic>` | Delegate to a specialist agent (`research`, `analyze`, `code`, `explore`) |
| `!agents` | List the specialist agents |
| `!jobs` | List background jobs |
| `!cancel #3` | Cancel a running job |
| `!run [lang] <code>` | Run code in the sandbox (default `python`) |
| `!calc <query>` | Compute with WolframAlpha |
| `!search <query>` | Search the web |
| `!remember <key> <value>` | Store a fact about yourself |
| `!recall <key>` | Recall a stored fact |
| `!forget <key>` | Delete a stored fact |
| `!listfiles [dir]` | List files in the workspace |
| `!sendfile <path>` | Send a workspace file to Discord |
| `!chatmode on\|off` | Toggle auto-reply to every message in this channel |
| `!test_piston` | Check the code sandbox is working |

Mentioning the bot (`@Bot hello`) or DMing it also works. `!chatmode on` makes
the bot reply to every message in a channel without needing a mention.

### Background research jobs

The bot decides for itself when a request needs deep work. Ask it to "deep
research" something, or ask for a report, analysis, or comparison, and it
starts a background job instead of answering in one pass:

```
you: can you do a deep research on Alphabet's financial position if the AI bubble bursts?
bot: A deep research job on Alphabet's financial position and AI bubble exposure
     has started. The report, with sources and a verdict, will be posted here.
⏳ Job #1 started — Alphabet/Google's financial position
-# step 4 of 100 · 2m14s elapsed · 🔍 Searching the web...
📄 report_1.md  📄 report_2.md
```

The job runs in the background while you keep chatting, reports progress as a
small grey line every 100 seconds, writes its report to markdown files as it
goes, and delivers those files as attachments when it finishes.

Every specialist agent gets 100 steps. If a job does use them all up, the
agent is asked for a final summary before it stops, so the work it did is
never thrown away.

You can also start one explicitly with `!research <topic>`, list them with
`!jobs`, and stop one with `!cancel #3`.

Jobs use a larger step budget and token limit than an interactive reply, so
they can dig much deeper. Quick questions, math, and simple lookups are
answered directly without starting a job.

## Files

| File | Purpose |
|---|---|
| `main.py` | Entrypoint: Discord bot setup, commands, event handlers |
| `config.py` | Environment configuration and provider clients |
| `agent.py` | The agent loop: prompt, context assembly, tool dispatch |
| `context.py` | Context management: rebuilding history, sizing it, summarising it |
| `tools.py` | Tool schemas and implementations (WolframAlpha, search, files, verify) |
| `agents.py` | Specialist agent registry (research, analyze, code, explore) |
| `jobs.py` | Background job runner |
| `sandbox.py` | Sandboxed code execution via Daytona |
| `storage.py` | Supabase persistence (history, facts, chat mode) |
| `memory.py` | Semantic long-term memory (embeddings, extraction, retrieval) |
| `vision.py` | Downloads Discord attachments for the vision model |
| `discord_utils.py` | Message splitting, sending, and math rendering |
| `mathrender.py` | LaTeX → Unicode text, and message-level math processing |
| `latex_render.py` | LaTeX → PNG via KaTeX + MathJax |
| `render/` | Node renderer script and npm dependencies |
| `diagrams.py` | Node/edge spec → Excalidraw sketch, with grid layout |
| `excalidraw_render.py` | Excalidraw JSON → PNG |
| `supabase_memory.sql` | Creates the `memories` table and its search function |
| `requirements.txt` | Python dependencies |
| `.env.example` | Copy to `.env` and add your keys |
| `render.yaml` | Render service definition (optional) |
| `audioop.py` | Stub so `discord.py` imports on Python 3.13+ |

## Local setup

1. **Create the environment file**
   ```bash
   cp .env.example .env
   ```

2. **Fill in `.env`** — see [Configuration](#configuration) below.

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run**
   ```bash
   python main.py
   ```

## Configuration

| Variable | Required | Description |
|---|---|---|
| `DISCORD_TOKEN` | yes | From the [Discord Developer Portal](https://discord.com/developers/applications) |
| `NVIDIA_API_KEY` | yes* | NVIDIA NIM key — <https://build.nvidia.com> |
| `NVIDIA_MODEL` | no | Defaults to `nvidia/nemotron-3.5-super-vl-120b-a12b` |
| `GROQ_API_KEY` | no | Groq fallback key — <https://console.groq.com> |
| `GROQ_MODEL` | no | Defaults to `qwen/qwen3.8-27b` |
| `SUPABASE_URL` | yes | Supabase project URL |
| `SUPABASE_KEY` | yes | Supabase service-role key |
| `DAYTONA_API_KEY` | no | Enables the code execution sandbox — <https://daytona.io> |
| `LANGSEARCH_API_KEY` | no | Enables web search — <https://langsearch.com> |
| `WOLFRAM_APP_ID` | no | Enables WolframAlpha — <https://wolframalpha.com/api> |
| `EMBED_MODEL` | no | Embedding model for semantic memory — defaults to `nvidia/nemotron-3.5-embed-1b` |
| `MEMORY_MIN_SCORE` | no | Relevance floor for memory retrieval — defaults to `0.15` |
| `NODE_BINARY` | no | Path to `node` if it is not on `PATH` (needed for math rendering) |

\* At least one of `NVIDIA_API_KEY` or `GROQ_API_KEY` must be set. If both are
present, NVIDIA is used first and Groq takes over if NVIDIA errors.

### Supabase tables

Create these three tables in Supabase (SQL editor):

```sql
create table conversation_history (
  id            bigserial primary key,
  user_id       text not null,
  role          text not null,
  content       text default '',
  created_at    timestamptz default now(),
  tool_calls    jsonb,
  tool_call_id  text,
  name          text
);
create index on conversation_history (user_id, created_at desc);

create table user_facts (
  id            bigserial primary key,
  user_id       text not null,
  fact_key      text not null,
  fact_value    text not null,
  updated_at    timestamptz default now(),
  unique (user_id, fact_key)
);

create table chat_mode (
  id            bigserial primary key,
  channel_id    text not null unique,
  enabled       boolean default true,
  created_at    timestamptz default now()
);
```

And a fourth, for long-term semantic memory — run
[`supabase_memory.sql`](supabase_memory.sql) (it enables the `vector`
extension, creates the `memories` table and the `match_memories` search
function):

```sql
create extension if not exists vector;

create table if not exists memories (
    id         bigserial primary key,
    user_id    text        not null,
    kind       text        not null default 'fact',
    content    text        not null,
    embedding  vector(2048),
    created_at timestamptz default now(),
    unique (user_id, content)
);
```

## Discord Developer Portal settings

1. Go to **Bot** → **Privileged Gateway Intents**
2. Enable **Message Content Intent** (required to read messages)
3. Save changes

## Deploy to Render (Free Web Service)

1. Push this folder to GitHub.
2. [Render Dashboard](https://dashboard.render.com/) → **New → Web Service**
3. Connect GitHub → select your repo
4. Settings:
   - **Runtime:** Python
   - **Build Command:**
     ```bash
     curl -fsSL https://nodejs.org/dist/v20.18.0/node-v20.18.0-linux-x64.tar.xz | tar -xJ -C /opt && export PATH=/opt/node-v20.18.0-linux-x64/bin:$PATH && node /opt/node-v20.18.0-linux-x64/lib/node_modules/npm/bin/npm-cli.js install --prefix ./render --no-audit --no-fund && pip install -r requirements.txt
     ```
   - **Start Command:** `python main.py`
   - **Instance Type:** Free
5. **Environment Variables** (in the Render dashboard, not `.env`):
   `DISCORD_TOKEN`, `NVIDIA_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`,
   and optionally `GROQ_API_KEY`, `DAYTONA_API_KEY`, `LANGSEARCH_API_KEY`,
   `WOLFRAM_APP_ID`, `EMBED_MODEL`.
   Also set `NODE_BINARY=/opt/node-v20.18.0-linux-x64/bin/node`.
6. **Before you deploy**: run `supabase_memory.sql` in the Supabase SQL
   editor, or long-term memory will stay off.

> **Node is not optional.** The math renderer is a Node script (KaTeX +
> MathJax). If the build command above is left as plain
> `pip install -r requirements.txt`, Node is never installed and the bot
> silently falls back to matplotlib's mathtext — which cannot typeset
> `\begin{aligned}`, `\begin{cases}` or any other multi-line environment, so
> equations come out as raw LaTeX. The bot prints whether the proper
> renderer is in use when it starts; check the Render logs for
> `math renderer: KaTeX + MathJax` (good) or `math renderer: matplotlib
> fallback` (Node is missing).

### Local setup for math rendering

Math needs Node.js. From the project folder:

```bash
cd render && npm install && cd ..
```

Then set `NODE_BINARY` if `node` is not already on your `PATH`:

```bash
export NODE_BINARY=$(which node)
```

### Keep it awake

Render free web services sleep after ~15 minutes of inactivity.

1. Get your Render URL (e.g. `https://your-bot.onrender.com`)
2. Create a free monitor at [UptimeRobot](https://uptimerobot.com)
3. Type: **HTTP(s)**, URL: `https://your-bot.onrender.com/`, Interval: **5 minutes**

## License

MIT
