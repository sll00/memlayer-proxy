"""
Simple script to start Memlayer Server with custom configuration.

This script demonstrates how to programmatically start the server instead of
using the CLI.

Usage:
    python3.12 examples/07_server/run_server.py
"""

import uvicorn
from memlayer.server import MemlayerProxy
from memlayer.server.config import ServerConfig

# Configure server settings
config = ServerConfig(
    llama_server_host="http://localhost:8080",
    proxy_host="0.0.0.0",
    proxy_port=8000,
    storage_path="./memlayer_server_data",
    operation_mode="local",  # 100% offline
    salience_threshold=0.0,
    enable_curation=True,
    curation_interval=3600,  # 1 hour
    scheduler_interval=60,  # 1 minute
    default_user_id="default_user",
    log_level="INFO",
    debug_mode=False,
)

print("=" * 70)
print("🚀 Starting Memlayer Server")
print("=" * 70)
print(f"  llama-server: {config.llama_server_host}")
print(f"  Proxy: http://{config.proxy_host}:{config.proxy_port}")
print(f"  Storage: {config.storage_path}")
print(f"  Mode: {config.operation_mode} (100% offline)")
print("=" * 70)
print("\n💡 Connect with OpenAI SDK:")
print("  from openai import OpenAI")
print(f"  client = OpenAI(base_url='http://localhost:{config.proxy_port}/v1', api_key='not-needed')")
print("\n📝 Example request:")
print(f"  curl http://localhost:{config.proxy_port}/v1/chat/completions \\")
print("    -H 'Content-Type: application/json' \\")
print("    -d '{\"model\": \"model\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'")
print("\n🏥 Health check:")
print(f"  curl http://localhost:{config.proxy_port}/")
print("=" * 70)
print("\n⏳ Initializing server...\n")

# Create proxy instance
proxy = MemlayerProxy(
    llama_server_host=config.llama_server_host,
    llama_server_port=config.llama_server_port,
    storage_path=config.storage_path,
    config=config,
)

# Create FastAPI app
app = proxy.create_app()

# Run server
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.proxy_host,
        port=config.proxy_port,
        log_level=config.log_level.lower(),
    )
