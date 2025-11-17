# Operation Modes: Architecture & Performance Implications

## Overview

`operation_mode` is a fundamental architectural choice that determines how Memlayer computes embeddings for semantic search. This affects startup time, runtime performance, cost, resource usage, and deployment constraints.

```python
from memlayer.wrappers.openai import OpenAI

client = OpenAI(
    model="gpt-4.1-mini",
    operation_mode="online"  # or "local" or "lightweight"
)
```

## The Three Modes

### 1. Online Mode (Default)

**Architecture:** Uses OpenAI's API for computing text embeddings.

```
User Query → Vector Search (OpenAI API) → ChromaDB Lookup → Graph Traversal → Results
             └─ API call (~50-100ms)
```

**When to use:**
- ✅ Serverless/cloud deployments (AWS Lambda, Cloud Functions)
- ✅ Production applications with internet access
- ✅ When you want minimal local resource usage
- ✅ Rapid development without model downloads

**Tradeoffs:**
- **Startup time**: Fast (~200ms, no model loading)
- **Search latency**: 100-300ms (includes API call)
- **Memory usage**: Low (~50MB base)
- **Cost**: ~$0.0001 per search (API call)
- **Privacy**: Embeddings computed by OpenAI
- **Reliability**: Requires internet connection

**Architecture components:**
```
┌─────────────────────────────────────────┐
│  Memlayer (online mode)                 │
├─────────────────────────────────────────┤
│  • ChromaStorage (vector DB)            │
│  • NetworkX/Memgraph (graph DB)         │
│  • OpenAI Embeddings API                │
│  • Salience Gate (API embeddings)       │
└─────────────────────────────────────────┘
```

**Configuration:**
```python
client = OpenAI(
    operation_mode="online",
    embedding_model="text-embedding-3-small",  # OpenAI model
    salience_threshold=0.5
)
```

---

### 2. Local Mode

**Architecture:** Uses a local sentence-transformer model for embeddings.

```
User Query → Vector Search (Local Model) → ChromaDB Lookup → Graph Traversal → Results
             └─ GPU/CPU inference (~20-50ms)
```

**When to use:**
- ✅ Privacy-sensitive deployments (no external API calls)
- ✅ Offline/air-gapped environments
- ✅ High-volume applications (no per-query API cost)
- ✅ GPU-accelerated servers

**Tradeoffs:**
- **Startup time**: Slow first call (~5-10s model load, cached after)
- **Search latency**: 50-200ms (no API dependency)
- **Memory usage**: High (~500MB-2GB depending on model)
- **Cost**: Zero API costs, higher compute costs
- **Privacy**: All computation on-premises
- **Reliability**: No internet required

**Architecture components:**
```
┌─────────────────────────────────────────┐
│  Memlayer (local mode)                  │
├─────────────────────────────────────────┤
│  • ChromaStorage (vector DB)            │
│  • NetworkX/Memgraph (graph DB)         │
│  • sentence-transformers (local model)  │
│  • Salience Gate (local embeddings)     │
│  • GPU/CPU inference pipeline           │
└─────────────────────────────────────────┘
```

**Configuration:**
```python
client = OpenAI(
    operation_mode="local",
    embedding_model="all-MiniLM-L6-v2",  # Local model
    salience_threshold=0.5
)
```

**First-time setup:**
```python
# First call downloads model (~80MB) and loads to memory
client.chat([{"role": "user", "content": "test"}])  # ~5-10s

# Subsequent calls are fast (model cached in memory)
client.chat([{"role": "user", "content": "hello"}])  # ~50ms search
```

---

### 3. Lightweight Mode

**Architecture:** No embeddings, pure graph-based memory with keyword matching.

```
User Query → Graph Keyword Search → NetworkX Traversal → Results
             └─ O(n) text scan (~10-50ms)
```

**When to use:**
- ✅ Demos and rapid prototyping
- ✅ Minimal resource constraints
- ✅ When semantic search isn't critical
- ✅ Testing and development

**Tradeoffs:**
- **Startup time**: Instant (~10ms)
- **Search latency**: 10-100ms (no embeddings)
- **Memory usage**: Minimal (~20MB base)
- **Cost**: Zero (no API, no models)
- **Search quality**: Lower recall, keyword-based only
- **Reliability**: No dependencies

**Architecture components:**
```
┌─────────────────────────────────────────┐
│  Memlayer (lightweight mode)            │
├─────────────────────────────────────────┤
│  • NetworkX/Memgraph (graph DB only)    │
│  • Keyword-based search                 │
│  • No vector embeddings                 │
│  • No salience gate                     │
└─────────────────────────────────────────┘
```

**Configuration:**
```python
client = OpenAI(
    operation_mode="lightweight",
    # No embedding_model needed
)
```

**Search behavior:**
```python
# Finds exact matches and entity relationships only
client.chat([{"role": "user", "content": "What's my name?"}])
# ✅ Works: "My name is Alice" stored → "Alice" found

client.chat([{"role": "user", "content": "Tell me about my identity"}])
# ❌ May not work: "identity" doesn't match "name" keyword
```

---

## Comparison Table

| Feature | Online | Local | Lightweight |
|---------|--------|-------|-------------|
| **Startup time** | ~200ms | ~5-10s (first call) | ~10ms |
| **Search latency** | 100-300ms | 50-200ms | 10-100ms |
| **Memory usage** | ~50MB | ~500MB-2GB | ~20MB |
| **API costs** | ~$0.0001/search | $0 | $0 |
| **Privacy** | API call to OpenAI | Fully local | Fully local |
| **Search quality** | High (semantic) | High (semantic) | Medium (keywords) |
| **GPU benefit** | None | Yes (faster inference) | None |
| **Internet required** | Yes | No | No |
| **Best for** | Production | Privacy/offline | Demos/testing |

---

## Architecture Deep Dive

### Storage Backend Impact

```python
# Online mode: Full stack
storage_stack = {
    "chroma": ChromaStorage(),      # Vector embeddings
    "graph": NetworkXStorage(),     # Entity relationships
    "embedder": OpenAIEmbeddings()  # API-based
}

# Local mode: Full stack
storage_stack = {
    "chroma": ChromaStorage(),           # Vector embeddings
    "graph": NetworkXStorage(),          # Entity relationships
    "embedder": SentenceTransformer()    # Local model
}

# Lightweight mode: Graph only
storage_stack = {
    "chroma": None,                 # Disabled
    "graph": NetworkXStorage(),     # Only component
    "embedder": None                # Disabled
}
```

### Salience Gate Behavior

The salience gate filters trivial content before storing it. Behavior varies by mode:

**Online mode:**
```python
# Uses OpenAI embeddings API to compute similarity
# Cache: ~/.memlayer_cache/salience_prototypes_online.pkl
# First call: ~1-2s (compute 100 prototype embeddings)
# Cached calls: ~0.01s (load from disk)
```

**Local mode:**
```python
# Uses local sentence-transformer model
# Cache: ~/.memlayer_cache/salience_prototypes_local.pkl
# First call: ~5-10s (load model + compute prototypes)
# Cached calls: ~0.05s (local inference)
```

**Lightweight mode:**
```python
# Salience gate disabled entirely
# All content stored (no filtering)
# Zero overhead
```

### Memory Search Flow

**Online mode search:**
```
1. User query: "What's my name?"
2. Embed query → OpenAI API call (~50ms)
3. ChromaDB vector search (~30ms)
4. NetworkX graph traversal (~20ms)
5. Return results (~100ms total)
```

**Local mode search:**
```
1. User query: "What's my name?"
2. Embed query → Local inference (~20ms)
3. ChromaDB vector search (~30ms)
4. NetworkX graph traversal (~20ms)
5. Return results (~70ms total)
```

**Lightweight mode search:**
```
1. User query: "What's my name?"
2. Keyword extraction (~5ms)
3. NetworkX keyword search (~30ms)
4. Graph traversal (~20ms)
5. Return results (~55ms total)
```

---

## Choosing the Right Mode

### Decision Tree

```
Need privacy/offline?
├─ Yes → LOCAL mode
└─ No → Need semantic search?
         ├─ Yes → ONLINE mode (recommended)
         └─ No → LIGHTWEIGHT mode
```

### By Use Case

**Serverless Functions (Lambda, Cloud Functions)**
```python
# Use online mode - fast cold starts
client = OpenAI(operation_mode="online")
```

**Long-running Servers**
```python
# Use local mode - absorb startup cost once
client = OpenAI(operation_mode="local")
# First request pays model load cost
# All subsequent requests are fast
```

**Demos and Prototypes**
```python
# Use lightweight - instant startup
client = OpenAI(operation_mode="lightweight")
```

**Privacy-Sensitive Applications**
```python
# Use local mode - no external API calls
client = OpenAI(operation_mode="local")
```

**High-Volume Production**
```python
# Use local mode on GPU server - no API costs
client = OpenAI(
    operation_mode="local",
    embedding_model="all-mpnet-base-v2"  # Better quality
)
```

---

## Performance Tuning

### Optimizing Online Mode

```python
client = OpenAI(
    operation_mode="online",
    embedding_model="text-embedding-3-small",  # Faster than 3-large
    max_search_results=5,                      # Fewer results = faster
    search_tier="fast"                          # Vector-only search
)
```

### Optimizing Local Mode

```python
client = OpenAI(
    operation_mode="local",
    embedding_model="all-MiniLM-L6-v2",  # Smaller, faster model
    # Use GPU if available (detected automatically)
)

# Warm up model on startup
client.chat([{"role": "user", "content": "warmup"}])
```

### Optimizing Lightweight Mode

```python
client = OpenAI(
    operation_mode="lightweight",
    max_search_results=3,      # Faster graph traversal
    search_tier="fast"          # Minimal graph hops
)
```

---

## Migration Between Modes

Switching modes requires re-embedding existing memories:

```python
# Start with online mode
client_online = OpenAI(
    user_id="alice",
    operation_mode="online",
    chroma_dir="./chroma_online"
)

# Migrate to local mode
client_local = OpenAI(
    user_id="alice", 
    operation_mode="local",
    chroma_dir="./chroma_local"  # Different directory
)

# Graph data (networkx_path) can be reused
# Vector embeddings must be recomputed
```

**Note:** Graph storage (NetworkX/Memgraph) is mode-independent and can be reused.

---

## Examples

See complete examples:
- **Online mode**: `examples/05_providers/openai_example.py`
- **Local mode**: `examples/05_providers/ollama_example.py`
- **Lightweight mode**: `examples/01_basics/getting_started.py`

---

## Related Documentation

- **[Basics Overview](../basics/overview.md)**: Architecture fundamentals
- **[Quickstart](../basics/quickstart.md)**: Getting started guide
- **[Salience Threshold](./salience_threshold.md)**: Content filtering
- **[ChromaDB Storage](../storage/chroma.md)**: Vector storage details
- **[Ollama Provider](../providers/ollama.md)**: Local model setup