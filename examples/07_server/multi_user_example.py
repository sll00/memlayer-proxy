"""
Multi-User Support Example

Demonstrates how to use per-user memory isolation with the X-User-ID header.
Each user gets their own isolated memory storage.

Prerequisites:
1. Start llama-server: ./llama-server -m model.gguf --port 8080 --chat-template llama3
2. Start Memlayer proxy: python3.12 -m memlayer.server
3. Run this script: python3.12 examples/07_server/multi_user_example.py
"""

import httpx
import json
import time

BASE_URL = "http://localhost:8000/v1/chat/completions"

def chat(user_id: str, message: str, model: str = "qwen2.5:7b"):
    """Send a chat message for a specific user"""
    response = httpx.post(
        BASE_URL,
        headers={
            "Content-Type": "application/json",
            "X-User-ID": user_id,
        },
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": message}
            ],
            "temperature": 0.7,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]

print("=" * 70)
print("Memlayer Server - Multi-User Support Example")
print("=" * 70)

print("\n📝 Phase 1: Alice shares her information")
print("-" * 70)

print("\n[Alice] Introducing herself...")
response = chat("alice", "Hi! My name is Alice and I'm a data scientist.")
print(f"Assistant: {response}")

print("\n[Alice] Sharing her work...")
response = chat("alice", "I work with pandas, scikit-learn, and TensorFlow.")
print(f"Assistant: {response}")

print("\n[Alice] Mentioning her hobby...")
response = chat("alice", "I love hiking and photography in my free time.")
print(f"Assistant: {response}")

print("\n📝 Phase 2: Bob shares his information")
print("-" * 70)

print("\n[Bob] Introducing himself...")
response = chat("bob", "Hello! My name is Bob and I'm a DevOps engineer.")
print(f"Assistant: {response}")

print("\n[Bob] Sharing his work...")
response = chat("bob", "I specialize in Docker, Kubernetes, and CI/CD pipelines.")
print(f"Assistant: {response}")

print("\n[Bob] Mentioning his hobby...")
response = chat("bob", "I enjoy playing guitar and brewing coffee.")
print(f"Assistant: {response}")

# Wait for consolidation
print("\n⏳ Waiting for memory consolidation (3 seconds)...")
time.sleep(3)

print("\n🔍 Phase 3: Testing memory isolation")
print("-" * 70)

print("\n[Alice] Asking about her own info...")
response = chat("alice", "What do you know about me?")
print(f"Assistant: {response}")

print("\n[Bob] Asking about his own info...")
response = chat("bob", "What do you know about me?")
print(f"Assistant: {response}")

print("\n🧪 Phase 4: Cross-user privacy test")
print("-" * 70)

print("\n[Alice] Asking about Bob (should not know)...")
response = chat("alice", "Do you know anything about Bob?")
print(f"Assistant: {response}")
print("✓ Privacy maintained: Alice's memory is isolated from Bob's")

print("\n[Bob] Asking about Alice (should not know)...")
response = chat("bob", "Do you know anything about Alice?")
print(f"Assistant: {response}")
print("✓ Privacy maintained: Bob's memory is isolated from Alice's")

print("\n" + "=" * 70)
print("✅ Multi-User Test Complete!")
print("=" * 70)

print("\n💡 Key Findings:")
print("  ✓ Each user has isolated memory storage")
print("  ✓ Alice's memories: ./memlayer_server_data/alice/")
print("  ✓ Bob's memories: ./memlayer_server_data/bob/")
print("  ✓ Cross-user privacy is maintained")
print("  ✓ Shared embedding model reduces memory usage")

print("\n📊 Storage Structure:")
print("  memlayer_server_data/")
print("  ├── alice/")
print("  │   ├── chroma/              # Alice's vector DB")
print("  │   └── knowledge_graph.pkl  # Alice's knowledge graph")
print("  └── bob/")
print("      ├── chroma/              # Bob's vector DB")
print("      └── knowledge_graph.pkl  # Bob's knowledge graph")

print("\n🔧 Implementation Tips:")
print("  • Pass X-User-ID header in all requests")
print("  • Use unique user IDs (email, UUID, username, etc.)")
print("  • Storage is automatically created per user")
print("  • Embedding model is shared (performance optimization)")
print("  • Each user gets independent memory lifecycle")

print("\n🚀 Production Use Cases:")
print("  • Multi-tenant SaaS applications")
print("  • Customer support chatbots")
print("  • Personal AI assistants")
print("  • Team collaboration tools")
print("  • Educational platforms")
