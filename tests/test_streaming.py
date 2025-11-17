"""
Tests for streaming functionality across all providers.
"""
import os
import sys
import time
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memlayer.wrappers.openai import OpenAI


@pytest.fixture
def openai_client():
    """Create a test OpenAI client."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    client = OpenAI(
        model="gpt-4.1-mini",
        storage_path="./test_data/streaming",
        user_id="test_user"
    )
    yield client
    client.close()


def test_streaming_basic(openai_client):
    """Test basic streaming without memory operations."""
    stream = openai_client.chat(
        messages=[{"role": "user", "content": "Count from 1 to 3."}],
        stream=True
    )
    
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
    
    # Should receive multiple small chunks
    assert len(chunks) > 1, "Should receive multiple chunks"
    
    # Each chunk should be relatively small (typical token chunks are 1-10 chars)
    avg_chunk_size = sum(len(c) for c in chunks) / len(chunks)
    assert avg_chunk_size < 20, f"Average chunk size too large: {avg_chunk_size}"
    
    # Full response should contain the expected content
    full_response = "".join(chunks)
    assert "1" in full_response
    assert "2" in full_response
    assert "3" in full_response


def test_streaming_with_memory(openai_client):
    """Test streaming with memory operations."""
    # Add information to memory
    openai_client.chat([
        {"role": "user", "content": "My favorite programming language is Python."}
    ])
    
    # Wait for consolidation
    time.sleep(2)
    
    # Query with streaming
    stream = openai_client.chat(
        messages=[{"role": "user", "content": "What do you know about my preferences?"}],
        stream=True
    )
    
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
    
    # Should receive multiple chunks
    assert len(chunks) > 5, "Should receive multiple chunks even with memory lookup"
    
    # Response should mention Python
    full_response = "".join(chunks)
    assert "Python" in full_response or "python" in full_response.lower()


def test_streaming_performance(openai_client):
    """Test that streaming starts quickly."""
    start_time = time.time()
    
    stream = openai_client.chat(
        messages=[{"role": "user", "content": "Say hello"}],
        stream=True
    )
    
    # Get first chunk
    first_chunk = next(stream)
    first_chunk_time = time.time() - start_time
    
    # First chunk should arrive within reasonable time (< 5 seconds)
    assert first_chunk_time < 5.0, f"First chunk took too long: {first_chunk_time:.2f}s"
    
    # Consume rest of stream
    for _ in stream:
        pass


def test_non_streaming_still_works(openai_client):
    """Test that non-streaming mode still works."""
    response = openai_client.chat([
        {"role": "user", "content": "Say hi"}
    ])
    
    assert isinstance(response, str)
    assert len(response) > 0


if __name__ == "__main__":
    # Run tests manually if executed directly
    print("Testing OpenAI streaming...")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    client = OpenAI(
        model="gpt-4.1-mini",
        storage_path="./test_data/streaming",
        user_id="test_user"
    )
    
    print("\n1. Testing basic streaming...")
    test_streaming_basic(client)
    print("✓ Passed")
    
    print("\n2. Testing streaming with memory...")
    test_streaming_with_memory(client)
    print("✓ Passed")
    
    print("\n3. Testing streaming performance...")
    test_streaming_performance(client)
    print("✓ Passed")
    
    print("\n4. Testing non-streaming mode...")
    test_non_streaming_still_works(client)
    print("✓ Passed")
    
    client.close()
    
    print("\n" + "="*60)
    print("All streaming tests passed!")
    print("="*60)
