# Memlayer Quickstart

Get started with Memlayer in under 5 minutes. This guide shows you how to add persistent memory to any LLM.

## Installation

```bash
pip install memlayer
```

### Provider-Specific Dependencies

Install the SDK for your chosen provider:

```bash
# OpenAI
pip install openai

# Anthropic Claude
pip install anthropic

# Google Gemini
pip install google-generativeai

# Ollama (local models)
pip install ollama
```

## Quick Start Examples

### OpenAI

```python
from memlayer.wrappers.openai import OpenAI

# Initialize with memory
client = OpenAI(
    api_key="your-openai-api-key",
    model="gpt-4.1-mini",
    user_id="alice"
)

# First conversation - teach it something
response = client.chat([
    {"role": "user", "content": "My name is Alice and I work on Project Phoenix"}
])
print(response)

# Later conversation - it remembers!
response = client.chat([
    {"role": "user", "content": "What project do I work on?"}
])
print(response)
# Output: "You work on Project Phoenix."
```

### Anthropic Claude

```python
from memlayer.wrappers.claude import Claude

client = Claude(
    api_key="your-anthropic-api-key",
    model="claude-3-5-sonnet-20241022",
    user_id="alice"
)

# Use exactly like OpenAI wrapper
response = client.chat([
    {"role": "user", "content": "Remember: my favorite color is blue"}
])
```

### Google Gemini

```python
from memlayer.wrappers.gemini import Gemini

client = Gemini(
    api_key="your-gemini-api-key",
    model="gemini-2.5-flash",
    user_id="alice"
)

response = client.chat([
    {"role": "user", "content": "I live in San Francisco"}
])
```

### Ollama (Local Models)

```python
from memlayer.wrappers.ollama import Ollama

# Make sure Ollama server is running: ollama serve
client = Ollama(
    model="llama3.2",
    host="http://localhost:11434",
    user_id="alice",
    operation_mode="local"  # Use local embeddings too
)

response = client.chat([
    {"role": "user", "content": "My dog's name is Max"}
])
```

## Environment Variables

Instead of passing API keys in code, use environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-key"

# Anthropic Claude
export ANTHROPIC_API_KEY="your-key"

# Google Gemini
export GOOGLE_API_KEY="your-key"
```

Then initialize without the `api_key` parameter:

```python
from memlayer.wrappers.openai import OpenAI

client = OpenAI(
    model="gpt-4.1-mini",
    user_id="alice"
)
```

## Basic Usage Patterns

### 1. Regular Chat (Non-Streaming)

```python
# Single turn
response = client.chat([
    {"role": "user", "content": "Hello!"}
])

# Multi-turn conversation
messages = [
    {"role": "user", "content": "My birthday is May 15th"},
    {"role": "assistant", "content": "I'll remember that!"},
    {"role": "user", "content": "When is my birthday?"}
]
response = client.chat(messages)
```

### 2. Streaming Chat

```python
# Stream response chunks as they arrive
for chunk in client.chat(
    [{"role": "user", "content": "Tell me a story"}],
    stream=True
):
    print(chunk, end="", flush=True)
print()  # Newline after stream completes
```

### 3. Direct Knowledge Import

```python
# Import knowledge from documents/emails/notes
client.update_from_text("""
Meeting Notes - Nov 15, 2025:
- Q4 deadline is December 20th
- Budget increased by 15%
- New team member: Bob (joins Monday)
""")

# Now the LLM can answer questions about this
response = client.chat([
    {"role": "user", "content": "When is the Q4 deadline?"}
])
# Output: "The Q4 deadline is December 20th."
```

### 4. Memory-Grounded Q&A

```python
# Get a synthesized answer with sources
answer_obj = client.synthesize_answer(
    "What do we know about Project Phoenix?",
    return_object=True
)

print(f"Answer: {answer_obj.answer}")
print(f"Sources: {answer_obj.sources}")
print(f"Confidence: {answer_obj.confidence}")
```

## Configuration Basics

### User Isolation

Each `user_id` gets an isolated memory space:

```python
alice_client = OpenAI(model="gpt-4.1-mini", user_id="alice")
bob_client = OpenAI(model="gpt-4.1-mini", user_id="bob")

# Alice's memories don't leak to Bob
alice_client.chat([{"role": "user", "content": "My secret is XYZ"}])
bob_response = bob_client.chat([{"role": "user", "content": "What's Alice's secret?"}])
# Bob won't know - different memory spaces
```

### Storage Paths

Customize where memories are stored:

```python
client = OpenAI(
    model="gpt-4.1-mini",
    user_id="alice",
    chroma_dir="./memories/vector_db",      # Vector embeddings
    networkx_path="./memories/graph.pkl"    # Knowledge graph
)
```

### Operation Modes

Choose how embeddings are computed:

```python
# Online mode (default) - uses OpenAI API for embeddings
client = OpenAI(model="gpt-4.1-mini", operation_mode="online")

# Local mode - uses local sentence-transformer (no API calls)
client = OpenAI(model="gpt-4.1-mini", operation_mode="local")

# Lightweight mode - no embeddings, graph-only (fastest startup)
client = OpenAI(model="gpt-4.1-mini", operation_mode="lightweight")
```

## Common Patterns

### Persistent Sessions

```python
# Initialize once, reuse across application lifetime
client = OpenAI(model="gpt-4.1-mini", user_id="alice")

# All conversations automatically build on previous memories
client.chat([{"role": "user", "content": "I like pizza"}])
# ... later ...
client.chat([{"role": "user", "content": "What food do I like?"}])
# Remembers: "You like pizza"
```

### Conversation History Management

```python
# Memlayer handles memory automatically, but you control conversation history
conversation = []

# Turn 1
conversation.append({"role": "user", "content": "My name is Alice"})
response = client.chat(conversation)
conversation.append({"role": "assistant", "content": response})

# Turn 2
conversation.append({"role": "user", "content": "What's my name?"})
response = client.chat(conversation)
# LLM can answer from:
# 1. Conversation history (conversation list)
# 2. Long-term memory (knowledge graph)
```

### Time-Sensitive Facts

```python
# System automatically extracts expiration dates
client.chat([{
    "role": "user", 
    "content": "The temporary password is 1234, valid for 24 hours"
}])

# After 24 hours, this fact is automatically removed by curation service
```

## Next Steps

- **[Streaming Mode Guide](./streaming.md)**: Learn about streaming responses
- **[Operation Modes](../tuning/operation_mode.md)**: Architecture implications
- **[Search Tiers](../tuning/intervals.md)**: Optimize search performance
- **[Ollama Setup](../providers/ollama.md)**: Run completely offline with local models
- **[Examples](../../examples/README.md)**: Browse complete working examples

## Troubleshooting

### "No module named 'memlayer'"
```bash
pip install memlayer
```

### "API key not found"
Set your environment variable or pass `api_key` parameter:
```python
client = OpenAI(api_key="your-key", ...)
```

### "Ollama connection refused"
Start the Ollama server:
```bash
ollama serve
```

### Slow first response
First call initializes salience gate (~1-2s). Subsequent calls are fast. Use `operation_mode="lightweight"` for instant startup in demos.

### Memory not persisting
Check that `chroma_dir` and `networkx_path` are writable directories. By default, they're created in the current working directory.
