# RAG WhatsApp Bot (WAHA)

A personal Retrieval-Augmented Generation assistant that lives in your own
WhatsApp **"Message Yourself"** chat. You send a question to yourself, the bot
replies in the same chat using your PDF documents as context.

No Twilio, no Meta business verification, no per-message fees.

```
Your phone ── WhatsApp ──▶ WAHA (Docker, port 3000)
                              │  webhook + REST
                              ▼
                     FastAPI bot (port 8000)
                     ├─ LangChain RAG (Chroma + local embeddings)
                     ├─ OpenRouter LLM (NVIDIA Nemotron free tier)
                     └─ documents/ watched for changes
```

## Features

- 💬 **Self-chat only** – replies in "Message Yourself", ignores everyone else
- 🔁 **Loop-proof** – the bot never answers its own answer (see below)
- 📄 **PDF ingestion** – drop PDFs in `documents/`, auto re-indexed
- 🧠 **100 % free stack** – local ONNX embeddings + OpenRouter free model
- ⚡ **Async** – webhook returns instantly, answer is generated in background
- ⌨️ **Typing indicator**, long answers split into readable chunks
- 🔐 **Optional** WAHA API key + HMAC webhook signature
- 🐳 **Two containers**: `waha` + `bot`

## Quick start (Docker)

```bash
cp .env.example .env
# put your OpenRouter key in OPENROUTER_API_KEY
# set WAHA_API_KEY (openssl rand -hex 24) and a strong WAHA_DASHBOARD_PASSWORD

mkdir -p documents
cp ~/my-docs/*.pdf documents/

docker compose up -d --build
```

### 1. Pair WhatsApp

The session starts automatically; open the QR page and scan it with your phone
(WhatsApp → Settings → Linked devices):

```
http://localhost:8000/qr            # QR rendered by the bot
```

Or use the WAHA dashboard:

```
http://localhost:3000/dashboard     # login with WAHA_DASHBOARD_USERNAME / _PASSWORD
```

> ⚠️ WAHA ignores well-known credentials such as `waha`, `admin` or `123` and
> generates a random dashboard password instead (printed in
> `docker compose logs waha`). Set a strong `WAHA_DASHBOARD_PASSWORD` in `.env`
> before starting, or read the generated one from the logs.

Session state lives in `./waha_sessions`, so you only pair once.

### 2. Use it

Open WhatsApp → **Message Yourself** (Saved Messages) → send a question.

```
You:  How many vacation days do I get?
Bot:  Employees receive 25 paid vacation days per year.
```

Commands:

| Command | Action |
|---------|--------|
| `/help` | show help |
| `/stats` | chunks + files indexed |
| `/reload` | rebuild the index now |
| `/ping` | liveness check |

The bot also re-indexes automatically when PDFs are added, changed or removed.

## Running the bot without Docker

WAHA in Docker, bot on the host:

```bash
docker run -d --name waha -p 3000:3000 \
  -e WHATSAPP_DEFAULT_ENGINE=NOWEB \
  -e WHATSAPP_HOOK_URL=http://host.docker.internal:8000/webhook \
  -e WHATSAPP_HOOK_EVENTS=message,message.any \
  --add-host host.docker.internal:host-gateway \
  -v "$PWD/waha_sessions:/app/.sessions" \
  devlikeapro/waha:latest

source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `WAHA_BASE_URL` | `http://localhost:3000` | WAHA REST endpoint |
| `WAHA_SESSION` | `default` | WAHA session name |
| `WAHA_API_KEY` | – | sent as `X-Api-Key` when set |
| `WAHA_HMAC_KEY` | – | verify `X-Hmac-Signature` (SHA512) |
| `WAHA_DASHBOARD_USERNAME` / `_PASSWORD` | `admin` / – | dashboard login (use a strong password) |
| `WAHA_SELF_CHAT_ONLY` | `true` | answer only in "Message Yourself" |
| `LLM_PROVIDER` | `openrouter` | `openrouter` or `openai` |
| `LLM_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | any OpenRouter model |
| `OPENROUTER_API_KEY` | – | required for `openrouter` |
| `EMBEDDING_PROVIDER` | `local` | `local` (free) or `openai` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | splitting |
| `TOP_K_RESULTS` | `4` | chunks retrieved per question |
| `SYSTEM_PROMPT` | see `.env.example` | assistant behaviour |

### Why local embeddings?

OpenRouter serves **chat completions only** – it has no `/embeddings` endpoint.
The default backend is the ONNX MiniLM model bundled with ChromaDB
(~80 MB, downloaded once, then offline). No OpenAI key needed.

## How the echo loop is avoided

In the self-chat both your question and the bot's reply are `fromMe: true`, so a
naive bot would keep answering itself. This project:

1. fingerprints every outgoing text **before** the HTTP call to WAHA, so the
   echo cannot win a race against the API response;
2. remembers the message ids WAHA returns from `sendText`;
3. deduplicates inbound message ids (WAHA emits `message` and `message.any`,
   and retries on failures);
4. only answers messages whose chat id is your own JID.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | liveness |
| `/health` | GET | RAG + WAHA session status |
| `/webhook` | POST | WAHA webhook (HMAC verified when configured) |
| `/webhook/test?message=...` | POST | answer without WhatsApp |
| `/stats` | GET | index statistics |
| `/reload` | POST | rebuild index |
| `/send?chat_id=...&message=...` | POST | send a message via WAHA |
| `/qr` | GET | pairing QR (PNG proxy) |
| `/session`, `/session/start` | GET/POST | WAHA session state |

## Project structure

```
whatsapp_bot/
├── main.py            # FastAPI webhook + self-chat routing
├── waha.py            # WAHA client, echo/duplicate protection
├── rag.py             # Chroma retriever + LangChain chain
├── embeddings.py      # local ONNX or OpenAI embeddings
├── watcher.py         # auto re-index on file changes
├── config.py          # environment configuration
├── run.py             # preflight checks + uvicorn
├── test_rag.py        # RAG smoke test (no WhatsApp)
├── Dockerfile
├── docker-compose.yml # waha + bot
├── documents/         # your PDFs
├── chroma_db/         # vector index (generated)
└── waha_sessions/     # WhatsApp session (generated)
```

## Troubleshooting

**`docker compose up` fails with `unknown shorthand flag: 'd' in -d`** – the
Compose plugin is missing and Docker is parsing `-d` as one of its own flags.
Install it (no sudo needed):

```bash
mkdir -p ~/.docker/cli-plugins
curl -fSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
  -o ~/.docker/cli-plugins/docker-compose
chmod +x ~/.docker/cli-plugins/docker-compose
docker compose version
```

Or system-wide: `sudo apt install docker-compose-plugin` (Debian/Ubuntu),
`sudo dnf install docker-compose-plugin` (Fedora), `sudo pacman -S docker-compose` (Arch).

**QR never appears** – check `docker compose logs waha`. Try another engine:
set `WHATSAPP_DEFAULT_ENGINE=GOWS` (or `WEBJS`) in `docker-compose.yml` and
restart; the engine determines which WhatsApp protocol implementation is used.

**Session says `SCAN_QR_CODE` / `FAILED`** – `POST /session/start` (or restart
the WAHA container), then rescan from the dashboard.

**Bot never replies in the self-chat** – WAHA must send `fromMe` events, which
requires `WHATSAPP_HOOK_EVENTS` to include `message.any` (already set).
Check `docker compose logs bot` for `Question from ...` lines.

**`/session` shows `http_status: 401` (WAHA unreachable)** – the bot's
`WAHA_API_KEY` does not match WAHA's. If the key is empty WAHA generates a
random one on every start; set a fixed key in `.env`, or set
`WAHA_NO_API_KEY=true` on the `waha` service for localhost-only use. Apply it
with `docker compose up -d --force-recreate waha bot`.

**`webhook` rejects with 401** – `WAHA_HMAC_KEY` is set in one service but not
the other. Remove it or set it in both.

**Answers stop with HTTP 429** – free OpenRouter models are rate limited.
Switch `LLM_MODEL` to another `:free` model or add credits.

**Empty answers / wrong language** – the local MiniLM embedder is tuned for
English; use `EMBEDDING_PROVIDER=openai` for multilingual corpora.

## Security

- `.env` and `waha_sessions/` are git-ignored – never commit them.
- Set `WAHA_API_KEY` and `WAHA_HMAC_KEY` before exposing port 3000 publicly.
- Change the dashboard password (`WAHA_DASHBOARD_PASSWORD`).
- Keep `WAHA_SELF_CHAT_ONLY=true` so the bot never replies to other people.

## License

MIT
