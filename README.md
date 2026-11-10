# Quantum Groq Discord Bot

A Discord bot that forwards your messages to [Groq](https://groq.com/) (LLM API) and replies with the answer.

## Features
- `!ask <question>` — Ask Groq anything
- Mention the bot (`@Bot hello`) — It replies via Groq
- DM the bot directly — It replies via Groq
- Flask keep-alive server for Render free tier

## Files
| File | Purpose |
|------|---------|
| `main.py` | Discord bot + Groq integration + Flask web server |
| `requirements.txt` | Python dependencies |
| `.env.example` | Copy to `.env` and add your tokens |
| `render.yaml` | Render service definition (optional) |
| `.gitignore` | Ignores secrets and caches |

## Local setup

1. **Create the environment file**
   ```bash
   cp .env.example .env
   ```

2. **Fill in `.env`**
   - `DISCORD_TOKEN` — From [Discord Developer Portal](https://discord.com/developers/applications)
   - `GROQ_API_KEY` — From [Groq Console](https://console.groq.com/)
   - `GROQ_MODEL` — Optional (default: `llama3-8b-8192`)

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run locally**
   ```bash
   python main.py
   ```

## Discord Developer Portal settings
1. Go to **Bot** → **Privileged Gateway Intents**
2. Enable **Message Content Intent** (required to read messages)
3. Save Changes

## Deploy to Render (Free Web Service)

1. Push this folder to GitHub.
2. [Render Dashboard](https://dashboard.render.com/) → **New → Web Service**
3. Connect GitHub → select your repo
4. Settings:
   - **Runtime:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python main.py`
   - **Instance Type:** Free
5. **Environment Variables** (in Render dashboard, not in `.env`):
   - `DISCORD_TOKEN`
   - `GROQ_API_KEY`
   - `GROQ_MODEL` (optional)

### Keep it awake
Render free web services sleep after ~15 minutes of inactivity.

1. Get your Render URL (e.g. `https://your-bot.onrender.com`)
2. Create a free monitor at [UptimeRobot](https://uptimerobot.com)
3. Type: **HTTP(s)**, URL: `https://your-bot.onrender.com/`, Interval: **5 minutes**

## License
MIT