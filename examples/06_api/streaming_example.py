"""
Streaming Response Example
===========================

This example demonstrates how to use streaming mode for real-time responses
from MemLayer-enhanced LLMs. Streaming is useful for:

1. Immediate user feedback (see responses as they're generated)
2. Better UX in interactive applications
3. Lower perceived latency

All MemLayer features (memory search, consolidation, etc.) work seamlessly
with streaming mode.
"""

import os
import sys
import time

from memlayer.wrappers.openai import OpenAI


def example_basic_streaming():
    """
    Example 1: Basic streaming without memory operations
    Shows how to enable streaming and process chunks in real-time.
    """
    print("="*70)
    print("Example 1: Basic Streaming")
    print("="*70)
    
    client = OpenAI(
        model="gpt-4.1-mini",
        storage_path="./example_data/streaming",
        user_id="demo_user"
    )
    
    print("\nAsking: 'Tell me a short joke about programming'\n")
    print("Response (streaming): ", end="", flush=True)
    
    # Enable streaming with stream=True
    stream = client.chat(
        messages=[{"role": "user", "content": "Tell me a short joke about programming"}],
        stream=True
    )
    
    # Process chunks as they arrive
    for chunk in stream:
        print(chunk, end="", flush=True)
    
    print("\n")
    client.close()


def example_streaming_with_memory():
    """
    Example 2: Streaming with memory operations
    Shows that memory search and consolidation work seamlessly with streaming.
    """
    print("="*70)
    print("Example 2: Streaming + Memory")
    print("="*70)
    
    client = OpenAI(
        model="gpt-4.1-mini",
        storage_path="./example_data/streaming",
        user_id="demo_user"
    )
    
    # Add some information to memory
    print("\nAdding information to memory...")
    client.chat([
        {"role": "user", "content": "My name is Alex and I'm learning Python and JavaScript."}
    ])
    
    print("Waiting for memory consolidation...")
    time.sleep(2)  # Give consolidation time to complete
    
    # Query with streaming - memory will be searched automatically
    print("\nAsking: 'What programming languages am I learning?'\n")
    print("Response (streaming): ", end="", flush=True)
    
    stream = client.chat(
        messages=[{"role": "user", "content": "What programming languages am I learning?"}],
        stream=True
    )
    
    for chunk in stream:
        print(chunk, end="", flush=True)
    
    print("\n")
    client.close()


def example_streaming_with_timing():
    """
    Example 3: Measure streaming performance
    Shows the latency benefits of streaming - first chunk arrives quickly.
    """
    print("="*70)
    print("Example 3: Streaming Performance")
    print("="*70)
    
    client = OpenAI(
        model="gpt-4.1-mini",
        storage_path="./example_data/streaming",
        user_id="demo_user"
    )
    
    print("\nComparing streaming vs non-streaming latency...\n")
    
    # Non-streaming (wait for complete response)
    print("Non-streaming mode:")
    start = time.time()
    response = client.chat([
        {"role": "user", "content": "List 3 benefits of Python"}
    ])
    total_time = time.time() - start
    print(f"  Time to complete response: {total_time:.2f}s")
    print(f"  Response: {response[:50]}...\n")
    
    # Streaming (first chunk arrives quickly)
    print("Streaming mode:")
    start = time.time()
    stream = client.chat(
        messages=[{"role": "user", "content": "List 3 benefits of JavaScript"}],
        stream=True
    )
    
    first_chunk = None
    first_chunk_time = None
    chunks = []
    
    for i, chunk in enumerate(stream):
        chunks.append(chunk)
        if i == 0:
            first_chunk = chunk
            first_chunk_time = time.time() - start
    
    total_time = time.time() - start
    
    print(f"  Time to FIRST chunk: {first_chunk_time:.2f}s ⚡")
    print(f"  Time to COMPLETE response: {total_time:.2f}s")
    print(f"  Total chunks received: {len(chunks)}")
    print(f"  Full response: {''.join(chunks)[:50]}...\n")
    
    print(f"💡 Streaming advantage: User sees response {first_chunk_time:.2f}s earlier!")
    
    client.close()


def example_all_providers():
    """
    Example 4: Streaming works with all providers
    Shows streaming with OpenAI, Claude, Gemini, and Ollama.
    """
    print("="*70)
    print("Example 4: Streaming Across All Providers")
    print("="*70)
    
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        print("\n🤖 OpenAI (gpt-4.1-mini) - streaming...", end=" ", flush=True)
        from memlayer.wrappers.openai import OpenAI
        client = OpenAI(model="gpt-4.1-mini", storage_path="./example_data/streaming", user_id="demo")
        stream = client.chat([{"role": "user", "content": "Say hi in 3 words"}], stream=True)
        response = "".join(stream)
        print(f"✓ '{response}'")
        client.close()
    
    # Claude
    if os.getenv("ANTHROPIC_API_KEY"):
        print("🤖 Claude (claude-3-5-sonnet) - streaming...", end=" ", flush=True)
        from memlayer.wrappers.claude import Claude
        import anthropic
        claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        client = Claude(
            client=claude_client,
            model="claude-3-5-sonnet-20241022",
            storage_path="./example_data/streaming",
            user_id="demo"
        )
        stream = client.chat([{"role": "user", "content": "Say hi in 3 words"}], stream=True)
        response = "".join(stream)
        print(f"✓ '{response}'")
        client.close()
    
    # Gemini
    if os.getenv("GOOGLE_API_KEY"):
        print("🤖 Gemini (gemini-2.5-flash) - streaming...", end=" ", flush=True)
        from memlayer.wrappers.gemini import Gemini
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        gemini_client = genai.GenerativeModel('gemini-2.5-flash')
        client = Gemini(
            client=gemini_client,
            storage_path="./example_data/streaming",
            user_id="demo"
        )
        stream = client.chat([{"role": "user", "content": "Say hi in 3 words"}], stream=True)
        response = "".join(stream)
        print(f"✓ '{response}'")
        client.close()
    
    # Ollama (local)
    try:
        print("🤖 Ollama (llama3.2) - streaming...", end=" ", flush=True)
        from memlayer.wrappers.ollama import Ollama
        client = Ollama(
            client_config={"provider": "ollama", "model": "llama3.2", "base_url": "http://localhost:11434"},
            storage_path="./example_data/streaming",
            user_id="demo"
        )
        stream = client.chat([{"role": "user", "content": "Say hi in 3 words"}], stream=True)
        response = "".join(stream)
        print(f"✓ '{response}'")
        client.close()
    except Exception as e:
        print(f"✗ Ollama not available ({str(e)[:30]}...)")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY environment variable to run these examples")
        sys.exit(1)
    
    print("\n" + "="*70)
    print(" MemLayer Streaming Examples")
    print("="*70 + "\n")
    
    # Run all examples
    example_basic_streaming()
    print("\n")
    
    example_streaming_with_memory()
    print("\n")
    
    example_streaming_with_timing()
    print("\n")
    
    example_all_providers()
    
    print("\n" + "="*70)
    print("✓ All streaming examples completed!")
    print("="*70)
    print("\n💡 Key Takeaways:")
    print("   • Use stream=True to enable streaming mode")
    print("   • Iterate over the generator to get chunks in real-time")
    print("   • Memory features work seamlessly with streaming")
    print("   • Streaming reduces perceived latency significantly")
    print("   • All providers (OpenAI, Claude, Gemini, Ollama) support streaming")
