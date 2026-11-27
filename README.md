# RAG WhatsApp Bot (WAHA + Qdrant)

A personal Retrieval-Augmented Generation assistant that lives in your own
WhatsApp **"Message Yourself"** chat. You send a question to yourself, the bot
replies in the same chat using your PDF documents as context.

No Twilio, no Meta business verification, no per-message fees.

```
Your phone ── WhatsApp ──▶ WAHA (Docker)  ──▶  FastAPI bot ──▶ Qdrant
                            webhook + REST      │
                                                ├─ LangChain RAG
                                                ├─ OpenRouter LLM
                                                ├─ local ONNX embeddings
                                                └─ query cache (memory + disk)
```

The stack can run **one bot** or **several independent bots** (for example a
personal and a work WhatsApp account). Each session has its own WAHA container,
its own documents folder, its own Qdrant collection and its own query cache
namespace, so the two never mix context.

## Features

- 💬 **Self-chat only** – replies in "Message Yourself", ignores everyone else
- 👥 **Multi-session** – personal and work bots side by side, fully isolated
- 🔁 **Loop-proof** – the bot never answers its own answer (see below)
- 📄 **PDF ingestion** – drop PDFs in the session's documents folder, auto re-indexed
- 🧠 **100 % free stack** – local ONNX embeddings + OpenRouter free model
- 🗄️ **Qdrant vector store** – persistent, fast, one collection per bot
- 🔎 **Hybrid retrieval** – keyword matching for names/numbers plus vector similarity
- 🎤 **Voice notes** – WhatsApp voice messages are transcribed locally (Whisper) and answered like text
- ⚡ **Two-level query cache** – in-memory LRU + on-disk TTL cache
- 📚 **Citations** – answers end with `Source: file, page N`
- 🧹 **WhatsApp-safe formatting** – Markdown is converted before sending
- 🚦 **Ordered, bounded queues** – one queue per chat, 1–2 concurrent LLM calls
- ♻️ **Retries + fallback** – jittered backoff on 429/5xx, optional second model
- 🔀 **Optional streaming** – token-by-token delivery (off by default)
- ⌨️ **Typing indicator**, long answers split into readable chunks
- 🔐 **Optional** WAHA API key + HMAC webhook signature
- 🐳 **Docker Compose** – WAHA, bot and Qdrant, all wired together

## Quick start (Docker, two sessions)

```bash
git clone https://github.com/PrateekT1987/whatsapp_ragbot.git
cd whatsapp_ragbot

# Compose-level secrets: WAHA keys + dashboard passwords per session
cp .env.example .env            # then fill in the values
# Per-bot config (LLM key, model, ports, collection name)
cp .env.personal.example .env.personal
cp .env.work.example .env.work

mkdir -p documents_personal documents_work
cp ~/personal-docs/*.pdf documents_personal/
cp ~/work-docs/*.pdf     documents_work/

docker compose up -d --build
```

Services started:

| Service | Container | Host port (loopback) | Purpose |
|---|---|---|---|
| `qdrant` | `qdrant` | 6333 | vector database (HTTP + dashboard, API key required) |
| `waha-personal` | `waha-personal` | 3001 | WhatsApp session `personal` |
| `bot-personal` | `rag-whatsapp-bot-personal` | 8001 | RAG bot for personal docs |
| `waha-work` | `waha-work` | 3002 | WhatsApp session `work` |
| `bot-work` | `rag-whatsapp-bot-work` | 8002 | RAG bot for work docs |

All published ports bind to `127.0.0.1`. To use them from another machine,
forward them instead of exposing them on the LAN:

```bash
ssh -L 8001:localhost:8001 -L 3001:localhost:3001 user@server
```

Only need one bot? Delete the `waha-work` / `bot-work` services from
`docker-compose.yml` and the `work` variables from `.env`, or run the
single-session setup described under
[Single-session deployment](#single-session-deployment).

### 1. Pair each WhatsApp session

Each WAHA container has its own QR. Open the pairing page of the bot that owns
the session:

```
http://localhost:8001/qr     # personal
http://localhost:8002/qr     # work
```

Scan with the matching phone (WhatsApp → Settings → Linked devices → Link a
device). Session state lives in `./waha_sessions/<session>`, so you only pair
once.

You can also use the WAHA dashboards:

```
http://localhost:3001/dashboard     # personal
http://localhost:3002/dashboard     # work
```

> ⚠️ WAHA ignores well-known credentials such as `waha`, `admin` or `123` and
> generates a random dashboard password instead (printed in
> `docker compose logs waha-personal`). Set a strong
> `WAHA_DASHBOARD_PASSWORD_PERSONAL` / `_WORK` in `.env` before starting, or
> read the generated one from the logs.

### 2. Use it

On the paired phone open WhatsApp → **Message Yourself** (Saved Messages) and
send a question.

```
You:  How many vacation days do I get?
Bot:  Employees receive 25 paid vacation days per year.
```

Each bot only answers its own phone's self-chat. A work question sent to the
personal number stays with the personal bot and only searches
`documents_personal/`.

## Commands

| Command | Action |
|---------|--------|
| `/help` | show help |
| `/stats` | chunks, files, vector store and cache statistics |
| `/reload` | rebuild the index now (also clears the query cache) |
| `/cache` | show cache statistics |
| `/cache clear` | clear the query cache |
| `/ping` | liveness check |

The bot also re-indexes automatically when a PDF is added, changed or removed
from its documents folder.

## How it works

### Documents → Qdrant

1. `watcher.py` notices PDF changes in `DOCUMENTS_PATH` (debounced, and it
   waits until a copied file stops changing).
2. `rag.py` loads them with `PyPDFLoader`, splits them
   (`CHUNK_SIZE` / `CHUNK_OVERLAP`) and embeds them locally. Only new or
   changed files are embedded; a changed file's old chunks are deleted first,
   and removing a PDF drops its chunks.
3. Chunks are written to the Qdrant collection named by `QDRANT_COLLECTION`
   (created automatically on first use, cosine vectors) with deterministic
   point ids, so an interrupted indexing pass resumes without duplicating
   points. The vector size is read from the embedder and checked against the
   existing collection; a mismatch fails with a clear error instead of
   corrupting the index.
4. The indexed files and their modification times are recorded in
   `cache/indexed_<collection>.json`, so a restart does not re-embed the whole
   folder (and does not duplicate points).

### Query → answer

1. WAHA's webhook is checked against `WAHA_SESSION`, then queued per chat. A
   chat's questions are answered in order, and a global semaphore
   (`MAX_CONCURRENT_QUERIES`) caps parallel model calls.
2. The question is normalized and looked up in the cache (memory, then disk).
   Identical questions already in flight share one model call.
3. On a miss, retrieval is **hybrid**: chunks containing the query's key terms
   verbatim are ranked first, then the top `TOP_K_RESULTS` vector-similarity
   chunks are appended. Keyword matching is punctuation-insensitive, which
   matters because PDF extraction glues serial numbers to names
   (`15Abhishek.J.Patil`) and dots inside names defeat ordinary token search.
4. The chunks become context for the chat prompt; OpenRouter generates the
   answer. Transient failures (429/5xx/timeout) are retried with jittered
   backoff; `LLM_FALLBACK_MODEL` is tried if the primary stays unavailable.
5. The answer is formatted for WhatsApp, given a `Source: file, page N` line,
   cached, and sent back to the self-chat. While `/reload` runs, new questions
   get a "reindexing" reply instead of an empty answer.

### Voice notes → text

WhatsApp voice messages (`ptt`/`audio`) are handled before the text path:

1. WAHA downloads the media and the webhook carries its URL
   (`/api/files/<session>/<message-id>.ogg`). Media is kept for
   `WHATSAPP_FILES_LIFETIME` seconds (600 in Compose) and stored under
   `waha_media/<session>/`.
2. `voice.py` downloads the file through the bot's WAHA connection and
   transcribes it with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   on the CPU (`WHISPER_MODEL`, default `tiny`, int8).
3. The transcript goes through exactly the same pipeline as typed text:
   commands, structured SQL queries, conversation memory, hybrid retrieval,
   caching and citations. The bot replies in text.
4. If transcription fails, the bot asks you to retry or send text. With
   `WHISPER_ENABLED=false`, voice notes are ignored silently.

The default `tiny` model is downloaded at image build time, so the first voice
note needs no network access. Setting a bigger model (`base`, `small`,
`medium`, `large`) downloads it on first use.

### Why local embeddings?

OpenRouter serves **chat completions only** – it has no `/embeddings` endpoint.
The default backend is the ONNX MiniLM model (`all-MiniLM-L6-v2`, 384-dim, ~90 MB)
run through [fastembed](https://github.com/qdrant/fastembed). It downloads once
into a shared Docker volume (`fastembed_cache`), so container rebuilds don't
download it again. No OpenAI key needed.

## Caching

Two levels, keyed by a SHA-256 of the normalized question (lower-cased,
whitespace-collapsed) **plus** the session name, model and collection, so
`How many days?` and `  how many   days? ` share an entry while changing the
model never serves an answer from the previous one.

| Level | Storage | Scope | Speed |
|---|---|---|---|
| L1 exact | in-process `OrderedDict` with LRU eviction | one container | microseconds |
| L2 disk | `diskcache` in `CACHE_DIR` | survives restarts, shared by a session | ~ms |

- A hit refreshes the entry's recency and TTL; the least recently used entry is
  evicted first.
- A disk hit is promoted back into L1 and its disk TTL is extended.
- Entries expire after `CACHE_TTL` seconds.
- Identical questions arriving concurrently share a single model call
  (single-flight), so a double-send cannot double the OpenRouter cost.
- `/reload` clears the cache, so rebuilt indexes never serve stale answers.
- `me_id` / `me_lid` for the WAHA session are also persisted in the same cache
  directory, which avoids an extra WAHA API round-trip on every restart.
- `/cache` reports hits, misses and hit rate.

Tune with:

| Variable | Default | Purpose |
|---|---|---|
| `CACHE_ENABLED` | `true` | turn caching off entirely |
| `CACHE_TTL` | `3600` | seconds before an entry expires |
| `CACHE_DIR` | `./cache` | disk-cache location |
| `EXACT_CACHE_SIZE` | `1000` | max entries in the in-memory LRU |

## Streaming

Set `LLM_STREAMING=true` to have the answer delivered as it is generated
instead of waiting for the full response. It is **off by default** because each
streamed chunk is a separate `sendText` call: tokens are coalesced into bubbles
of roughly 160 characters (or at paragraph/sentence boundaries), and a
`STREAM_CHUNK_DELAY_MS` pause (default 400 ms) separates bubbles so a long
answer cannot trip WhatsApp's flood protection.

Retrieval always happens before generation, so streaming never changes *which*
documents are used — only how the answer is delivered. Citations are sent as a
final bubble. Cached answers are sent as a single message and are not streamed.
If a stream fails before any text was sent, the bot falls back to the normal
retry path.

## Reliability

- **Session isolation** – webhooks whose `session` is not this bot's
  `WAHA_SESSION` are ignored, so a shared webhook URL cannot mix personal and
  work messages.
- **Ordered queues** – each chat has a bounded queue (`MAX_QUEUED_PER_CHAT`);
  replies stay in order and a burst cannot create dozens of parallel calls.
  When a queue is full, the chat gets a one-line "still working" notice.
- **Retries** – WAHA sends and LLM calls retry transient 429/5xx/connection
  failures with jittered exponential backoff. Sends are not retried on a read
  timeout (that could duplicate a message). A "model is busy — retrying…"
  notice is sent once per question.
- **Fallback model** – set `LLM_FALLBACK_MODEL` to a second `:free` model; it is
  used only after the primary exhausts its retries.
- **Startup resilience** – the Qdrant connection happens in the lifespan and is
  retried in the background; the HTTP server starts immediately and `/ready`
  reports 503 with the reason until it succeeds.
- **WAHA recovery** – a `FAILED` session is restarted once at startup and the
  live status is included in `/ready`.
- **Rebuild safety** – `/reload` waits for in-flight questions, tells new ones
  the bot is reindexing, and only then drops/recreates the collection.
- **Dimension check** – the embedder's vector size is verified against the
  existing collection; a mismatch stops with an actionable message instead of
  writing invalid points.
- **Shared HTTP client** – one pooled `httpx.AsyncClient` for all WAHA calls.

## Configuration

### Compose-level (`.env`)

These are consumed by `docker-compose.yml`, not by the bot:

| Variable | Purpose |
|---|---|
| `WAHA_API_KEY_PERSONAL` / `_WORK` | `X-Api-Key` for each WAHA instance |
| `WAHA_HMAC_KEY_PERSONAL` / `_WORK` | webhook HMAC secret for each instance |
| `WAHA_DASHBOARD_USERNAME_PERSONAL` / `_WORK` | dashboard login |
| `WAHA_DASHBOARD_PASSWORD_PERSONAL` / `_WORK` | dashboard password (strong!) |

### Per-bot (`.env.personal`, `.env.work`)

| Variable | Default | Purpose |
|---|---|---|
| `WAHA_BASE_URL` | `http://localhost:3000` | WAHA REST endpoint (compose overrides to the container name) |
| `WAHA_SESSION` | `default` | WAHA session name |
| `WAHA_API_KEY` | – | sent as `X-Api-Key` when set |
| `WAHA_HMAC_KEY` | – | verify `X-Webhook-Hmac` (SHA512, algorithm from `X-Webhook-Hmac-Algorithm`) |
| `WAHA_SELF_CHAT_ONLY` | `true` | answer only in "Message Yourself" |
| `WAHA_SEND_RETRIES` | `2` | retries for transient WAHA send failures |
| `WAHA_TIMEOUT` | `30` | WAHA HTTP timeout (seconds) |
| `LLM_PROVIDER` | `openrouter` | `openrouter` or `openai` |
| `LLM_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | any OpenRouter model |
| `LLM_FALLBACK_MODEL` | – | second `:free` model used after retries fail |
| `LLM_STREAMING` | `false` | deliver the answer incrementally |
| `LLM_MAX_RETRIES` | `3` | retries for transient LLM failures |
| `LLM_RETRY_BASE_SECONDS` | `2.0` | first backoff delay (doubles, capped) |
| `LLM_TIMEOUT` | `120` | LLM request timeout (seconds) |
| `SHOW_SOURCES` | `true` | append `Source: file, page N` to answers |
| `OPENROUTER_API_KEY` | – | required for `openrouter` |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API base |
| `EMBEDDING_PROVIDER` | `local` | `local` (free) or `openai` |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint (compose uses `http://qdrant:6333`) |
| `QDRANT_API_KEY` | – | Qdrant API key (required by the Compose stack) |
| `QDRANT_COLLECTION` | `rag_documents` | **must be unique per bot** |
| `FASTEMBED_CACHE_PATH` | `/model-cache` | ONNX model cache (the default model is baked into the image) |
| `CACHE_ENABLED` | `true` | enable the query cache |
| `CACHE_TTL` | `3600` | cache entry lifetime (seconds) |
| `CACHE_DIR` | `./cache` | disk-cache directory |
| `EXACT_CACHE_SIZE` | `1000` | in-memory LRU size |
| `DOCUMENTS_PATH` | `./documents` | PDF folder (compose mounts `documents_<name>`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | splitting |
| `TOP_K_RESULTS` | `4` | chunks retrieved per question |
| `MAX_CONCURRENT_QUERIES` | `2` | parallel LLM calls across all chats |
| `MAX_QUEUED_PER_CHAT` | `5` | messages buffered per chat before it gets a "busy" notice |
| `STREAM_CHUNK_DELAY_MS` | `400` | pause between streamed bubbles (flood protection) |
| `WHISPER_ENABLED` | `true` | transcribe WhatsApp voice notes locally |
| `WHISPER_MODEL` | `tiny` | faster-whisper size: `tiny` (prefetched), `base`, `small`, `medium`, `large` |
| `WHISPER_LANGUAGE` | – | force a language code, empty = auto-detect |
| `STARTUP_RETRY_SECONDS` | `2` | first Qdrant-connect retry delay (doubles to 60s) |
| `UVICORN_RELOAD` | `false` | dev-only auto-restart on `.py` edits |
| `SYSTEM_PROMPT` | see `.env.example` | assistant behaviour |

`docker-compose.yml` overrides `DOCUMENTS_PATH` via the volume mount, so inside
`bot-personal` the documents folder is always `/app/documents` – the host
folder name (`documents_personal/`) is what differs.

## Qdrant

One collection per bot:

```
rag_documents_personal   ← bot-personal
rag_documents_work       ← bot-work
```

- Web UI: `http://localhost:6333/dashboard?api_key=<QDRANT_API_KEY>`
  (the API key is in `.env`; plain HTTP requests without it return 401)
- Data lives in `./qdrant_data/` (git-ignored) and survives rebuilds.
- Collections are created automatically with cosine vectors whose size is read
  from the configured embedder (384 for the bundled MiniLM model).
- Switching embedding provider changes the vector size; the bot detects the
  mismatch at startup and asks you to delete the collection (or run `/reload`,
  which drops and recreates it) instead of writing invalid points.
- To start clean: `docker compose down && rm -rf qdrant_data && docker compose up -d`

## Single-session deployment

You can ignore `waha-work` / `bot-work` entirely. Minimum setup:

```bash
cp .env.example .env
cp .env.personal.example .env.personal   # or write your own

mkdir -p documents
cp ~/my-docs/*.pdf documents/

docker compose up -d --build waha-personal bot-personal qdrant
```

Pair at `http://localhost:8001/qr` and use the personal bot on its own. Set
`QDRANT_COLLECTION` to any name you like.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | liveness |
| `/health` | GET | RAG, cache and WAHA session status (informational) |
| `/ready` | GET | readiness: 200 when Qdrant + QA chain work, 503 otherwise |
| `/webhook` | POST | WAHA webhook (HMAC verified when configured) |
| `/webhook/test?message=...` | POST | answer without WhatsApp |
| `/stats` | GET | index + cache statistics |
| `/reload` | POST | rebuild index and clear cache |
| `/send?chat_id=...&message=...` | POST | send a message via WAHA |
| `/qr` | GET | pairing page (HTML, auto-refreshes) |
| `/qr.png` | GET | pairing QR (raw PNG) |
| `/session`, `/session/start` | GET/POST | WAHA session state |

Use the bot's port (`8001` personal, `8002` work), e.g.
`curl http://localhost:8001/health`.

## Project structure

```
whatsapp_bot/
├── main.py            # FastAPI webhook, commands, streaming switch
├── waha.py            # WAHA client, echo/duplicate protection, chunk send
├── rag.py             # Qdrant retriever + LangChain chains (cached/streaming)
├── vectorstore.py     # Qdrant manager (collection lifecycle, counts)
├── cache.py           # query cache (memory + disk) and session cache
├── formatting.py      # Markdown → WhatsApp markup conversion
├── embeddings.py      # local ONNX or OpenAI embeddings
├── watcher.py         # auto re-index on file changes
├── config.py          # environment configuration
├── run.py             # preflight checks + uvicorn
├── test_rag.py        # RAG smoke test (no WhatsApp)
├── test_p1.py         # unit tests for formatting/cache/retry/citations
├── Dockerfile
├── docker-compose.yml # qdrant + waha/bot per session
├── documents_personal/  # personal PDFs       (git-ignored)
├── documents_work/      # work PDFs           (git-ignored)
├── qdrant_data/         # vector data         (generated, git-ignored)
├── cache/               # disk cache          (generated, git-ignored)
└── waha_sessions/       # WhatsApp sessions   (generated, git-ignored)
```

## How the echo loop is avoided

In the self-chat both your question and the bot's reply are `fromMe: true`, so a
naive bot would keep answering itself. This project:

1. fingerprints every outgoing text **before** the HTTP call to WAHA, so the
   echo cannot win a race against the API response;
2. remembers the message ids WAHA returns from `sendText`;
3. deduplicates inbound message ids (WAHA emits `message` and `message.any`,
   and retries on failures);
4. only answers messages whose chat id is your own JID (checked against both
   `me.id` and `me.lid`);
5. replies to the sender JID (`from`), not the LID `chatId`, so answers land in
   the visible self-chat instead of a hidden LID thread.

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

**Image build fails resolving a package** – PyPI was unreachable or rate
limited during the build. Retry (`docker compose build bot-personal`); the
Dockerfile already wraps apt in retries and raises the pip timeout.

**Qdrant is unreachable / bot restarts** – check
`docker compose logs qdrant` and that the `QDRANT_URL` host matches where Qdrant
runs (`http://qdrant:6333` inside Compose, `http://localhost:6333` on the host).
The bot's healthcheck only covers the HTTP server, so also check
`curl http://localhost:6333/collections`.

**Collection dimension mismatch after changing embedder** – the existing
collection has the old vector size. Run `/reload` (it drops and recreates the
collection) or `docker compose down && rm -rf qdrant_data`.

**QR never appears** – check `docker compose logs waha-personal` (or `waha-work`).
Try another engine: set `WHATSAPP_DEFAULT_ENGINE=GOWS` (or `WEBJS`) in
`docker-compose.yml` and restart; the engine determines which WhatsApp protocol
implementation is used.

**`/qr` shows an error or nothing** – the page auto-restarts an expired session
and retries, but the first render can take ~10 s. If you browse from another
machine, `localhost` is *your* machine, not the server: use the server's IP
(`http://<server-ip>:8001/qr`) or forward the port with
`ssh -L 8001:localhost:8001 user@server` and open `http://localhost:8001/qr`.

**Session says `SCAN_QR_CODE` / `FAILED`** – just reload the bot's `/qr` page; it
restarts the session automatically. You can also `POST /session/start` or
restart the WAHA container.

**Bot never replies in the self-chat** – WAHA must send `fromMe` events, which
requires `WHATSAPP_HOOK_EVENTS` to include `message.any` (already set).
Check `docker compose logs bot-personal` for `Question from ...` lines.

**Answers are stale after replacing a PDF** – automatic re-indexing deletes a
changed file's old chunks before re-embedding it. If the index still looks
wrong, run `/reload` for a full rebuild.

**`/session` shows `http_status: 401` (WAHA unreachable)** – the bot's
`WAHA_API_KEY` does not match WAHA's. The WAHA container gets
`WAHA_API_KEY_PERSONAL`/`_WORK` from `.env`, while the bot gets `WAHA_API_KEY`
from `.env.personal`/`.env.work`; the two values must be identical. If the key
is empty WAHA generates a random one on every start. Fix the per-bot value (or
set `WAHA_NO_API_KEY=true` on the WAHA service for localhost-only use) and apply
with `docker compose up -d --force-recreate waha-personal bot-personal`.

**`webhook` rejects with 401** – `WAHA_HMAC_KEY` is set in one service but not
the other. Remove it or set it in both.

**Answers stop with HTTP 429** – free OpenRouter models are rate limited.
Switch `LLM_MODEL` to another `:free` model or add credits.

**Empty answers / wrong language** – the local MiniLM embedder is tuned for
English; use `EMBEDDING_PROVIDER=openai` for multilingual corpora.

## Backup & Recovery (runbook)

### Qdrant snapshots
Qdrant supports instant, consistent snapshots of a collection. Use the helper
script:

```bash
# 1) Create + download a snapshot
python backup_qdrant.py --create --collection rag_documents_personal --download --output ./backups

# 2) List all snapshots
python backup_qdrant.py --list

# 3) Restore into a fresh collection (new name avoids mismatch)
python backup_qdrant.py --restore ./backups/rag_documents_personal-<timestamp>.snapshot \
       --collection rag_documents_personal_restored
```

- Snapshots are stored inside the container at `/qdrant/storage/snapshots/`.
- The `docker compose` healthcheck hits `/readyz`; a healthy Qdrant can snapshot
  while serving queries.
- Schedule with cron (host or a sidecar container) if you want automated backups.

### Full state recovery
```bash
# Stop everything
docker compose down

# Remove old data (if corrupted)
rm -rf qdrant_data cache waha_sessions

# Restore qdrant_data from your latest host backup
# (tar/rsync/whatever you use for the ./qdrant_data directory)

# Bring stack up
docker compose up -d

# Verify
curl http://localhost:8001/ready
curl http://localhost:8001/stats
```

### Per-bot cache
Each bot now has its own cache directory (`./cache/personal`, `./cache/work`)
so the two containers never share the same `diskcache` SQLite file. If you need
to clear one bot's cache without touching the other:

```bash
rm -rf cache/personal   # or cache/work
docker compose restart bot-personal
```

## Security

- `.env`, `.env.personal`, `.env.work`, `waha_sessions/`, `qdrant_data/` and
  `cache/` are git-ignored – never commit them. `.dockerignore` also excludes
  them, so secrets and PDFs are never copied into an image.
- Every published port binds to `127.0.0.1` on the **host**. Inside the Docker
  network the containers bind to `0.0.0.0`; the loopback publish keeps the
  services off the external interface. Use SSH port forwarding for remote
  access; do not change these to `0.0.0.0` unless you add a firewall.
- Qdrant requires `QDRANT_API_KEY` (set in `.env`, injected into Qdrant and both
  bots). Use the same key in the dashboard URL.
- WAHA webhooks are HMAC-signed: `WAHA_HMAC_KEY_PERSONAL`/`_WORK` in `.env` must
  match `WAHA_HMAC_KEY` in `.env.personal`/`.env.work`.
- Set a strong WAHA API key and dashboard password per session.
- Keep `WAHA_SELF_CHAT_ONLY=true` so the bot never replies to other people.
- The bot's own HTTP API (`/send`, `/reload`, `/webhook/test`, `/metrics`) has
  no authentication; it is safe only because it is loopback-bound.

## License

MIT
