"""
Memlayer Server - OpenAI-compatible reverse proxy with memory capabilities.

This module provides a FastAPI-based server that acts as a drop-in replacement
for the OpenAI API, adding persistent memory to any llama-server instance.

Usage:
    python3.12 -m memlayer.server --llama-host http://localhost:8080

Or programmatically:
    from memlayer.server import MemlayerProxy

    proxy = MemlayerProxy(
        llama_server_host="http://localhost:8080",
        storage_path="./memlayer_data"
    )
    app = proxy.create_app()
"""

from .proxy import MemlayerProxy

__all__ = ["MemlayerProxy"]
