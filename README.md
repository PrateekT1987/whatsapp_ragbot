# RAG WhatsApp Bot

A Retrieval-Augmented Generation (RAG) chatbot integrated with WhatsApp via Twilio. The bot answers questions based on PDF documents you provide, with automatic re-indexing when documents change.

## Features

- 🤖 **WhatsApp Integration** via Twilio WhatsApp API
- 📄 **PDF Document Ingestion** with automatic text extraction
- 🔍 **RAG Pipeline** using OpenAI embeddings and GPT-4o-mini
- 🔄 **Auto-reindexing** - watches documents folder for changes
- 💬 **Slash Commands** - `/help`, `/stats`, `/reload`
- 🐳 **Docker Support** for easy deployment
- 📊 **Health Checks** and monitoring endpoints

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Twilio Account (free tier works for testing)
- OpenAI API Key

### 2. Setup

```bash
# Clone/navigate to project
cd whatsapp_bot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### 3. Configure `.env`

**Option A: Using OpenAI directly**
```env
# Twilio (get from https://console.twilio.com)
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886  # Twilio sandbox number

# OpenAI (get from https://platform.openai.com/api-keys)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=gpt-4o-mini

# Webhook URL (use ngrok for local development)
WEBHOOK_URL=https://your-domain.ngrok-free.app
```

**Option B: Using OpenRouter (access to 100+ models including Claude, GPT, Llama, etc.)**
```env
# Twilio (get from https://console.twilio.com)
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886  # Twilio sandbox number

# OpenRouter (get from https://openrouter.ai/keys)
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=anthropic/claude-3-haiku  # Or any model from https://openrouter.ai/models

# Webhook URL (use ngrok for local development)
WEBHOOK_URL=https://your-domain.ngrok-free.app
```

> **Note:** You only need **one** of `OPENAI_API_KEY` or `OPENROUTER_API_KEY` depending on your `LLM_PROVIDER` choice.

### 4. Add Documents

Place PDF files in the `documents/` folder:

```bash
mkdir -p documents
cp your-docs/*.pdf documents/
```

### 5. Run Locally

```bash
# Option 1: Direct Python
python run.py

# Option 2: With ngrok (for Twilio webhook)
ngrok http 8000
# Update WEBHOOK_URL in .env with the ngrok HTTPS URL
```

### 6. Configure Twilio Webhook

1. Go to [Twilio Console > Messaging > Try it out > Send a WhatsApp message](https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn)
2. Or configure your WhatsApp Business number:
   - Go to **Phone Numbers > Manage > Active Numbers**
   - Click your WhatsApp number
   - Set **A Message Comes In** webhook to: `https://your-domain.com/webhook`
   - Method: **POST**

## Usage

### WhatsApp Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/stats` | Show document index statistics |
| `/reload` | Manually rebuild the document index |

### Ask Questions

Just send any question! The bot will search your documents and answer based on their content.

## Project Structure

```
whatsapp_bot/
├── main.py              # FastAPI application & webhook
├── config.py            # Configuration management
├── rag.py               # RAG system (documents, embeddings, QA)
├── watcher.py           # File system watcher for auto-reindex
├── whatsapp.py          # Twilio WhatsApp integration
├── run.py               # Entry point with validation
├── test_rag.py          # Test script for RAG system
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Docker Compose for deployment
├── documents/           # Put your PDF files here
└── chroma_db/           # Vector database (auto-created)
```

## Deployment

### Docker (Recommended)

```bash
# Build and run
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Production Considerations

1. **Use a reverse proxy** (nginx, Traefik) with SSL
2. **Set `WEBHOOK_URL`** to your public HTTPS endpoint
3. **Monitor logs** for errors
4. **Backup `chroma_db/`** periodically
5. **Consider using a managed vector DB** (Pinecone, Weaviate) for scale

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Detailed health status |
| `/webhook` | POST | Twilio WhatsApp webhook |
| `/stats` | GET | RAG index statistics |
| `/reload` | POST | Rebuild document index |
| `/send` | POST | Send a WhatsApp message (testing) |

## Customization

### Change Model

**Via `.env` (recommended):**
```env
# For OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# For OpenRouter - choose from 100+ models at https://openrouter.ai/models
LLM_PROVIDER=openrouter
LLM_MODEL=anthropic/claude-3-haiku
# Other examples:
# LLM_MODEL=openai/gpt-4o
# LLM_MODEL=meta-llama/llama-3.1-8b-instruct
# LLM_MODEL=google/gemini-flash-1.5
```

**Or edit `rag.py` directly:**
```python
self.llm = ChatOpenAI(
    model="gpt-4o",  # or any model supported by your provider
    temperature=0,
    openai_api_key=config.OPENAI_API_KEY
)
```

### Adjust Chunking

Edit `.env` or `config.py`:
```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=4
```

### Custom System Prompt

Edit `.env`:
```env
SYSTEM_PROMPT=You are a specialized assistant for [domain]. Answer based only on provided context.
```

## Troubleshooting

### "RAG system not initialized"
- Check your API key is valid (OpenAI or OpenRouter based on `LLM_PROVIDER`)
- Check logs for initialization errors
- For OpenRouter: verify model name exists at https://openrouter.ai/models

### Twilio webhook not receiving messages
- Ensure `WEBHOOK_URL` is publicly accessible (use ngrok locally)
- Verify Twilio webhook URL matches exactly: `https://your-domain.com/webhook`
- Check Twilio debugger for errors

### Documents not indexing
- Ensure PDFs are in `documents/` folder
- Check file permissions
- Run `python test_rag.py` to diagnose

### "Missing configuration" error
- Copy `.env.example` to `.env`
- Fill in all required values

## License

MIT License - feel free to use and modify!