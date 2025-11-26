# Memlayer Server Examples

This directory contains examples for running Memlayer as an **OpenAI-compatible reverse proxy server**.

## What is Memlayer Server?

Memlayer Server is a FastAPI-based reverse proxy that adds persistent memory capabilities to llama-server, while maintaining full OpenAI API compatibility. This allows you to use any OpenAI-compatible client (SDKs, tools, frameworks) with your local models **and** get automatic memory features.

## Key Features

- **100% Offline**: Uses local sentence-transformers for embeddings (no API calls)
- **OpenAI-compatible**: Drop-in replacement for OpenAI API
- **Multi-user support**: Per-user memory isolation via `X-User-ID` header
- **Tool calling**: Native function calling support via llama-server
- **Streaming**: Full SSE streaming support
- **Performance**: Shared embedding model across users for efficiency

## Prerequisites

1. **Install llama.cpp and build llama-server**:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make llama-server
   ```

2. **Download a GGUF model** (e.g., from Hugging Face)

3. **Start llama-server with function calling support**:
   ```bash
   ./llama-server -m /path/to/model.gguf --port 8080 -ngl 99 --chat-template llama3
   ```

4. **Install Memlayer with server dependencies**:
   ```bash
   pip install memlayer[server]
   # or for development:
   python3.12 -m pip install -e .[server]
   ```

## Quick Start

### Start the Server

```bash
# Using defaults (llama-server at localhost:8080, proxy at 0.0.0.0:8000)
python3.12 -m memlayer.server

# With custom settings
python3.12 -m memlayer.server \
    --llama-host http://localhost:8080 \
    --proxy-port 8000 \
    --storage-path ./my_memories

# Enable debug mode
python3.12 -m memlayer.server --debug
```

### Use with OpenAI SDK

```python
from openai import OpenAI

# Point to Memlayer proxy instead of OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # No API key required
)

# Use exactly like OpenAI API
response = client.chat.completions.create(
    model="qwen2.5:7b",  # Your llama-server model
    messages=[
        {"role": "user", "content": "My name is Alice"}
    ]
)

print(response.choices[0].message.content)
```

## Examples

### 1. run_server.py

Simple script to start the server with custom configuration.

```bash
python3.12 examples/07_server/run_server.py
```

### 2. test_client.py

Demonstrates using the OpenAI SDK with Memlayer proxy, including:
- Storing memories
- Retrieving memories
- Multi-turn conversations

```bash
python3.12 examples/07_server/test_client.py
```

### 3. multi_user_example.py

Shows how to use per-user memory isolation with the `X-User-ID` header.

```bash
python3.12 examples/07_server/multi_user_example.py
```

## Architecture

```
┌─────────────────┐
│  Your Client    │
│  (OpenAI SDK)   │
└────────┬────────┘
         │ HTTP POST /v1/chat/completions
         ▼
┌─────────────────────────────────────┐
│   Memlayer Proxy (FastAPI)          │
│   - Parse OpenAI request             │
│   - Extract user_id from X-User-ID   │
│   - Route to LlamaServer wrapper     │
│   - Add memory via SearchService     │
│   - Return OpenAI response           │
└────────┬────────────────────────────┘
         │
         ├──► LocalEmbeddingModel (shared, singleton)
         │    └─ sentence-transformers (384d)
         │
         ├──► ChromaDB (per user_id)
         │    └─ Vector storage for facts
         │
         ├──► NetworkX (per user_id)
         │    └─ Graph storage for relationships
         │
         ▼
┌─────────────────┐
│  llama-server   │
│  (llama.cpp)    │
└─────────────────┘
```

## Configuration

### Environment Variables

All settings can be configured via environment variables:

```bash
export MEMLAYER_LLAMA_SERVER_HOST=http://localhost:8080
export MEMLAYER_PROXY_PORT=8000
export MEMLAYER_STORAGE_PATH=./memlayer_data
export MEMLAYER_DEFAULT_USER_ID=default_user
export MEMLAYER_LOG_LEVEL=INFO
export MEMLAYER_DEBUG_MODE=false
```

### CLI Options

```bash
python3.12 -m memlayer.server --help
```

Options:
- `--llama-host`: llama-server URL
- `--llama-port`: llama-server port
- `--proxy-host`: Proxy bind address
- `--proxy-port`: Proxy port
- `--storage-path`: Memory storage location
- `--no-curation`: Disable memory curation
- `--curation-interval`: Curation interval (seconds)
- `--scheduler-interval`: Task scheduler interval (seconds)
- `--debug`: Enable debug logging
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--reload`: Enable auto-reload for development
- `--workers`: Number of worker processes

## API Endpoints

### POST /v1/chat/completions

OpenAI-compatible chat completions endpoint with memory.

**Headers:**
- `Content-Type: application/json`
- `X-User-ID: <user_id>` (optional, defaults to "default_user")

**Request Body:**
```json
{
  "model": "qwen2.5:7b",
  "messages": [
    {"role": "user", "content": "What's my name?"}
  ],
  "temperature": 0.7,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "qwen2.5:7b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Your name is Alice."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

### GET /

Health check endpoint.

**Response:**
```json
{
  "service": "memlayer-server",
  "status": "ready",
  "llama_server": "http://localhost:8080",
  "mode": "offline (local embeddings)"
}
```

## Multi-User Support

Use the `X-User-ID` header to isolate memories per user:

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "X-User-ID": "alice"
    },
    json={
        "model": "qwen2.5:7b",
        "messages": [
            {"role": "user", "content": "My favorite color is blue"}
        ]
    }
)
```

Each user gets their own isolated storage at `{storage_path}/{user_id}/`.

## Streaming

Streaming is supported via Server-Sent Events (SSE):

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

stream = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Performance Tips

1. **Shared Embedding Model**: The server uses a singleton embedding model shared across all users, reducing memory usage and initialization time.

2. **Per-User Client Caching**: LlamaServer wrapper instances are cached per `user_id`, avoiding redundant initialization.

3. **Background Processing**: Memory consolidation runs asynchronously, so responses are fast.

4. **GPU Acceleration**: Use `-ngl 99` flag with llama-server to offload layers to GPU.

5. **Model Selection**: Use smaller GGUF models (e.g., qwen2.5:3b) for faster responses, or larger models (e.g., llama3.2:8b) for better quality.

## Troubleshooting

### Server won't start

- **Check llama-server is running**: `curl http://localhost:8080/health`
- **Check port is available**: `lsof -i :8000`
- **Check Python version**: Must use `python3.12` (homebrew installed)

### Client can't connect

- **Verify proxy URL**: `curl http://localhost:8000/`
- **Check firewall settings**
- **Try localhost instead of 0.0.0.0**: `--proxy-host 127.0.0.1`

### Memory not working

- **Wait for consolidation**: Memory extraction runs in background thread (2-3 seconds)
- **Check storage path exists and is writable**
- **Enable debug mode**: `--debug` to see detailed logs

### Tool calling not working

- **Ensure llama-server supports function calling**: Use `--chat-template` flag
- **Check model supports function calling**: Not all models do
- **Try explicit system prompts** if model doesn't have native support

## Next Steps

- Read the [main documentation](../../docs/)
- Try the [provider examples](../05_providers/)
- Explore [search tiers](../02_search_tiers/)
- Learn about [memory lifecycle](../03_features/memory_lifecycle.py)

## Support

For issues, questions, or feature requests:
- GitHub Issues: https://github.com/divagr18/memlayer/issues
- Documentation: https://divagr18.github.io/memlayer/
