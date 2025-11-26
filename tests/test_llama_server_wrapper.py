"""
Tests for LlamaServer wrapper.

These tests verify that the LlamaServer wrapper correctly:
- Initializes with proper configuration
- Uses local embeddings by default
- Handles chat completions
- Integrates with memory services

Note: These tests mock llama-server responses to avoid requiring a running server.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from memlayer.wrappers.llama_server import LlamaServer


class TestLlamaServerInitialization:
    """Test LlamaServer initialization and configuration"""

    def test_default_initialization(self):
        """Test initialization with default parameters"""
        with patch('memlayer.wrappers.llama_server.openai.OpenAI'):
            client = LlamaServer()

            assert client.host == "http://localhost:8080"
            assert client.model == "model"
            assert client.temperature == 0.7
            assert client.storage_path == "./memlayer_data"
            assert client.user_id == "default_user"
            assert client.operation_mode == "local"  # Always local

    def test_custom_initialization(self):
        """Test initialization with custom parameters"""
        with patch('memlayer.wrappers.llama_server.openai.OpenAI'):
            client = LlamaServer(
                host="http://192.168.1.100:9000",
                model="qwen2.5:7b",
                temperature=0.5,
                storage_path="/tmp/test_memories",
                user_id="test_user",
            )

            assert "192.168.1.100" in client.base_url
            assert client.model == "qwen2.5:7b"
            assert client.temperature == 0.5
            assert client.storage_path == "/tmp/test_memories"
            assert client.user_id == "test_user"

    def test_base_url_construction(self):
        """Test that base_url is correctly constructed"""
        with patch('memlayer.wrappers.llama_server.openai.OpenAI'):
            # Test with port parameter
            client1 = LlamaServer(host="http://localhost", port=8080)
            assert client1.base_url == "http://localhost:8080/v1"

            # Test with port in host
            client2 = LlamaServer(host="http://localhost:8080")
            assert "localhost:8080" in client2.base_url

            # Test with /v1 already in host
            client3 = LlamaServer(host="http://localhost:8080/v1")
            assert client3.base_url == "http://localhost:8080/v1"

    def test_forces_local_mode(self):
        """Test that operation_mode is always forced to 'local' for offline operation"""
        with patch('memlayer.wrappers.llama_server.openai.OpenAI'):
            # Try to set online mode
            client = LlamaServer(operation_mode="online")

            # Should be forced to local
            assert client.operation_mode == "local"


class TestLlamaServerLazyLoading:
    """Test lazy loading of services and storage"""

    @patch('memlayer.wrappers.llama_server.openai.OpenAI')
    def test_embedding_model_lazy_load(self, mock_openai):
        """Test that embedding model is only loaded when accessed"""
        client = LlamaServer()

        # Should not be loaded yet
        assert client._embedding_model is None

        # Access it
        with patch('memlayer.wrappers.llama_server.LocalEmbeddingModel') as mock_embed:
            mock_embed.return_value = Mock(dimension=384)
            _ = client.embedding_model

            # Should now be loaded
            mock_embed.assert_called_once()

    @patch('memlayer.wrappers.llama_server.openai.OpenAI')
    def test_storage_lazy_load(self, mock_openai):
        """Test that storage is only loaded when accessed"""
        client = LlamaServer()

        # Should not be loaded yet
        assert client._vector_storage is None
        assert client._graph_storage is None

        # Access vector storage
        with patch('memlayer.wrappers.llama_server.ChromaStorage') as mock_chroma:
            with patch('memlayer.wrappers.llama_server.LocalEmbeddingModel') as mock_embed:
                mock_embed.return_value = Mock(dimension=384)
                _ = client.vector_storage
                mock_chroma.assert_called_once()

        # Access graph storage
        with patch('memlayer.wrappers.llama_server.NetworkXStorage') as mock_nx:
            _ = client.graph_storage
            mock_nx.assert_called_once()


class TestLlamaServerToolSchema:
    """Test that tool schema is properly defined"""

    @patch('memlayer.wrappers.llama_server.openai.OpenAI')
    def test_tool_schema_structure(self, mock_openai):
        """Test that tool schema has correct structure"""
        client = LlamaServer()

        assert len(client.tool_schema) == 2

        # Check search_memory tool
        search_tool = client.tool_schema[0]
        assert search_tool["type"] == "function"
        assert search_tool["function"]["name"] == "search_memory"
        assert "query" in search_tool["function"]["parameters"]["properties"]
        assert "search_tier" in search_tool["function"]["parameters"]["properties"]

        # Check schedule_task tool
        task_tool = client.tool_schema[1]
        assert task_tool["type"] == "function"
        assert task_tool["function"]["name"] == "schedule_task"
        assert "task_description" in task_tool["function"]["parameters"]["properties"]
        assert "due_date" in task_tool["function"]["parameters"]["properties"]


class TestLlamaServerChat:
    """Test chat functionality with mocked responses"""

    @patch('memlayer.wrappers.llama_server.openai.OpenAI')
    def test_simple_chat_no_tools(self, mock_openai):
        """Test simple chat without tool calls"""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock(
            content="Hello! How can I help you?",
            tool_calls=None
        )

        mock_client.chat.completions.create.return_value = mock_response

        # Create client and chat
        client = LlamaServer()

        # Mock services to avoid initialization
        client._search_service = Mock()
        client._search_service.get_triggered_tasks_context.return_value = ""
        client._consolidation_service = Mock()
        client._curation_service = Mock()
        client._scheduler_service = Mock()

        response = client.chat(messages=[
            {"role": "user", "content": "Hello"}
        ])

        assert response == "Hello! How can I help you?"
        assert mock_client.chat.completions.create.called

    @patch('memlayer.wrappers.llama_server.openai.OpenAI')
    def test_chat_with_consolidation(self, mock_openai):
        """Test that consolidation service is called"""
        mock_client = MagickMock()
        mock_openai.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock(
            content="Got it!",
            tool_calls=None
        )

        mock_client.chat.completions.create.return_value = mock_response

        client = LlamaServer()

        # Mock services
        client._search_service = Mock()
        client._search_service.get_triggered_tasks_context.return_value = ""
        client._consolidation_service = Mock()
        client._curation_service = Mock()
        client._scheduler_service = Mock()

        client.chat(messages=[
            {"role": "user", "content": "My name is Alice"}
        ])

        # Consolidation should be called
        client._consolidation_service.consolidate.assert_called_once()


class TestLlamaServerKnowledgeExtraction:
    """Test knowledge extraction methods"""

    @patch('memlayer.wrappers.llama_server.openai.OpenAI')
    def test_analyze_and_extract_knowledge(self, mock_openai):
        """Test knowledge extraction from text"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock LLM response with knowledge graph
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock(
            content=json.dumps({
                "facts": [{"fact": "Alice is a data scientist", "importance_score": 0.8, "expiration_date": None}],
                "entities": [{"name": "Alice", "type": "Person"}],
                "relationships": []
            })
        )

        mock_client.chat.completions.create.return_value = mock_response

        client = LlamaServer()
        result = client.analyze_and_extract_knowledge("Alice is a data scientist")

        assert "facts" in result
        assert "entities" in result
        assert "relationships" in result
        assert len(result["facts"]) == 1
        assert len(result["entities"]) == 1

    @patch('memlayer.wrappers.llama_server.openai.OpenAI')
    def test_extract_query_entities(self, mock_openai):
        """Test entity extraction from query"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock LLM response with entities
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock(
            content='["Alice", "TechCorp"]'
        )

        mock_client.chat.completions.create.return_value = mock_response

        client = LlamaServer()
        entities = client.extract_query_entities("Tell me about Alice at TechCorp")

        assert isinstance(entities, list)
        assert "Alice" in entities
        assert "TechCorp" in entities


class TestLlamaServerErrorHandling:
    """Test error handling"""

    @patch('memlayer.wrappers.llama_server.openai.OpenAI')
    def test_connection_error_handling(self, mock_openai):
        """Test handling of connection errors"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Simulate connection error
        mock_client.chat.completions.create.side_effect = Exception("Connection refused")

        client = LlamaServer()

        # Mock services
        client._search_service = Mock()
        client._search_service.get_triggered_tasks_context.return_value = ""
        client._consolidation_service = Mock()
        client._curation_service = Mock()
        client._scheduler_service = Mock()

        response = client.chat(messages=[
            {"role": "user", "content": "Hello"}
        ])

        assert "Error" in response
        assert "llama-server" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
