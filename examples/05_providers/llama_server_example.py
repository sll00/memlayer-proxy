"""
llama-server (Local) Provider Example

Demonstrates using Memlayer with llama-server for local LLM inference.
100% offline operation with local sentence-transformers embeddings.

Prerequisites:
1. Install llama.cpp and build llama-server:
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp && make llama-server

2. Download a GGUF model (e.g., from Hugging Face)

3. Start llama-server with function calling support:
   ./llama-server -m model.gguf --port 8080 -ngl 99 --chat-template llama3

Note: llama-server is OpenAI-compatible and runs completely offline!
"""

from memlayer.wrappers.llama_server import LlamaServer

print("="*70)
print("Memlayer - LLAMA-SERVER (LOCAL) EXAMPLE")
print("="*70)

# Initialize the memory-enhanced llama-server client
client = LlamaServer(
    # llama-server URL (default: http://localhost:8080)
    host="http://localhost:8080",

    # Model name (can be anything, just for identification)
    model="qwen2.5:7b",  # Or "llama3.2:3b", "mistral:7b", etc.

    # Standard LLM parameters
    temperature=0.7,

    # Memlayer settings (100% offline)
    storage_path="./llama_server_memories",
    user_id="demo_user",

    # Operation mode: Always "local" for offline operation
    # Uses sentence-transformers for embeddings (no API calls)
    operation_mode="local",
)

print("\n📝 Conversation 1: Teaching the local LLM about yourself")
print("-" * 70)

response = client.chat(messages=[
    {"role": "user", "content": "Hello! My name is Alex and I'm a machine learning engineer."}
])
print(f"Assistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "I specialize in NLP and work with transformers, PyTorch, and Hugging Face."}
])
print(f"\nAssistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "My current project is building a local RAG system for document search."}
])
print(f"\nAssistant: {response}")

# Wait for background consolidation
print("\n⏳ Waiting for memory consolidation...")
import time
time.sleep(3)

print("\n🔍 Conversation 2: Testing memory recall with tool calling")
print("-" * 70)

response = client.chat(messages=[
    {"role": "user", "content": "What do you know about my work and specialization?"}
])
print(f"Assistant: {response}")

response = client.chat(messages=[
    {"role": "user", "content": "What project am I currently working on?"}
])
print(f"\nAssistant: {response}")

print("\n🔧 Conversation 3: Testing task scheduling")
print("-" * 70)

response = client.chat(messages=[
    {"role": "user", "content": "Remind me tomorrow at 3pm to review the RAG system performance metrics."}
])
print(f"Assistant: {response}")

print("\n📊 Observability: Inspecting the last search")
print("-" * 70)

if client.last_trace:
    print(f"\nSearch Trace:")
    for event in client.last_trace.events:
        metadata_str = f" ({event.metadata})" if event.metadata else ""
        print(f"  • {event.event_type}: {event.duration_ms:.1f}ms{metadata_str}")

    print(f"\nTotal search time: {client.last_trace.total_duration_ms:.1f}ms")
else:
    print("No search trace available (no memory search was triggered)")

print("\n🧪 Conversation 4: Testing deep search with graph traversal")
print("-" * 70)

response = client.chat(messages=[
    {"role": "user", "content": "Tell me everything you know about me using deep search."}
])
print(f"Assistant: {response}")

if client.last_trace:
    print(f"\nDeep Search Stats:")
    for event in client.last_trace.events:
        if event.event_type == "graph_search" and event.metadata:
            print(f"  • Entities matched: {event.metadata.get('matched_entities', [])}")
            print(f"  • Relationships found: {event.metadata.get('relationships_found', 0)}")
            print(f"  • Nodes traversed: {event.metadata.get('nodes_traversed', 0)}")

print("\n✅ Example complete!")
print("\n💡 Tips for llama-server:")
print("  - 100% offline - no API calls whatsoever!")
print("  - OpenAI-compatible API (native function calling)")
print("  - Uses local sentence-transformers for embeddings")
print("  - GPU acceleration via -ngl flag")
print("  - Supports jinja chat templates for tool calling")
print("  - Perfect for privacy-sensitive applications")
print("\n📦 Quick llama-server Setup:")
print("  1. Clone: git clone https://github.com/ggerganov/llama.cpp")
print("  2. Build: cd llama.cpp && make llama-server")
print("  3. Download GGUF model from Hugging Face")
print("  4. Start: ./llama-server -m model.gguf --port 8080 --chat-template llama3")
print("  5. Run this script with python3.12!")

# Cleanup
client.close()
