"""
Test client for Memlayer Server using OpenAI SDK.

This demonstrates how to use any OpenAI-compatible client with Memlayer proxy,
getting automatic memory capabilities without changing your code.

Prerequisites:
1. Start llama-server: ./llama-server -m model.gguf --port 8080 --chat-template llama3
2. Start Memlayer proxy: python3.12 -m memlayer.server
3. Run this script: python3.12 examples/07_server/test_client.py
"""

from openai import OpenAI
import time

print("=" * 70)
print("Memlayer Server - OpenAI SDK Test Client")
print("=" * 70)

# Initialize OpenAI client pointing to Memlayer proxy
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # Memlayer doesn't require authentication
)

print("\n✅ Connected to Memlayer proxy at http://localhost:8000")
print("\n" + "=" * 70)
print("📝 Phase 1: Teaching the LLM about yourself")
print("=" * 70)

# First conversation - storing information
print("\n💬 Message 1: Introducing yourself...")
response = client.chat.completions.create(
    model="qwen2.5:7b",  # Your llama-server model name
    messages=[
        {"role": "user", "content": "Hello! My name is Jordan and I'm a software architect."}
    ],
    temperature=0.7,
)
print(f"Assistant: {response.choices[0].message.content}")

print("\n💬 Message 2: Sharing your work...")
response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "user", "content": "I work on designing microservices architectures using Kubernetes and gRPC."}
    ],
    temperature=0.7,
)
print(f"Assistant: {response.choices[0].message.content}")

print("\n💬 Message 3: Mentioning your tech stack...")
response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "user", "content": "My favorite technologies are Go, Python, and PostgreSQL."}
    ],
    temperature=0.7,
)
print(f"Assistant: {response.choices[0].message.content}")

# Wait for background consolidation
print("\n⏳ Waiting for memory consolidation (3 seconds)...")
time.sleep(3)

print("\n" + "=" * 70)
print("🔍 Phase 2: Testing memory recall")
print("=" * 70)

print("\n💬 Query 1: Asking about your name...")
response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "user", "content": "What's my name?"}
    ],
    temperature=0.7,
)
print(f"Assistant: {response.choices[0].message.content}")

print("\n💬 Query 2: Asking about your profession...")
response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "user", "content": "What do I do for work?"}
    ],
    temperature=0.7,
)
print(f"Assistant: {response.choices[0].message.content}")

print("\n💬 Query 3: Asking about your tech stack...")
response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "user", "content": "What technologies do I like to use?"}
    ],
    temperature=0.7,
)
print(f"Assistant: {response.choices[0].message.content}")

print("\n" + "=" * 70)
print("🚀 Phase 3: Testing streaming")
print("=" * 70)

print("\n💬 Streaming query: Tell me about myself...")
print("Assistant: ", end="", flush=True)

stream = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "user", "content": "Tell me everything you remember about me."}
    ],
    temperature=0.7,
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print("\n")

print("\n" + "=" * 70)
print("✅ Test Complete!")
print("=" * 70)
print("\n💡 Key Takeaways:")
print("  ✓ Drop-in OpenAI API replacement")
print("  ✓ Automatic memory storage and retrieval")
print("  ✓ 100% offline operation (no API keys needed)")
print("  ✓ Works with any OpenAI-compatible client")
print("  ✓ Streaming support")
print("\n📊 Behind the scenes:")
print("  • Memories stored in ./memlayer_server_data/default_user/")
print("  • Local sentence-transformers for embeddings")
print("  • ChromaDB for vector search")
print("  • NetworkX for knowledge graph")
print("  • llama-server for LLM inference")
print("\n🔧 Next steps:")
print("  • Try multi-user support with X-User-ID header")
print("  • Explore tool calling (function calling)")
print("  • Test with different models")
print("  • Integrate into your existing OpenAI workflows")
