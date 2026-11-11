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
| Memory / history | Supabase (`conversation_history`, `user_facts`, `chat_mode`) |

The agent can call tools in a loop: it computes with WolframAlpha, runs code,
searches the web, and reads/writes/edits files in a sandboxed workspace before
answering. Every reply ends with a small grey footer naming the model and the
tools that were used.

## Commands

| Command | Description |
|---|---|
| `!ask <question>` | Ask the agent anything |
| `!research <topic>` | Start a long background research job (up to 30 steps) |
| `!jobs` | List background research jobs |
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

`!research <topic>` starts a long job that runs in the background while the bot
stays responsive. It reports progress as small grey lines, writes its report to
markdown files as it goes, and delivers those files as attachments when it
finishes:

```
⏳ Job #1 started — AI capex financing
-# step 4 of 30 · 2m14s elapsed · 🔍 Searching the web...
📄 report_1.md  📄 report_2.md
```

Jobs use a larger step budget and token limit than an interactive reply, so
they can dig much deeper.

## Files

| File | Purpose |
|---|---|
| `main.py` | Discord bot, agent loop, tools, and Flask keep-alive server |
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
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python main.py`
   - **Instance Type:** Free
5. **Environment Variables** (in the Render dashboard, not `.env`):
   `DISCORD_TOKEN`, `NVIDIA_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`,
   and optionally `GROQ_API_KEY`, `DAYTONA_API_KEY`, `LANGSEARCH_API_KEY`,
   `WOLFRAM_APP_ID`.

### Keep it awake

Render free web services sleep after ~15 minutes of inactivity.

1. Get your Render URL (e.g. `https://your-bot.onrender.com`)
2. Create a free monitor at [UptimeRobot](https://uptimerobot.com)
3. Type: **HTTP(s)**, URL: `https://your-bot.onrender.com/`, Interval: **5 minutes**

## License

MIT
