"""
CLI entry point for Memlayer Server.

Usage:
    python3.12 -m memlayer.server [OPTIONS]

Examples:
    # Start with defaults
    python3.12 -m memlayer.server

    # Custom llama-server URL
    python3.12 -m memlayer.server --llama-host http://192.168.1.100:8080

    # Custom proxy port
    python3.12 -m memlayer.server --proxy-port 9000

    # Custom storage path
    python3.12 -m memlayer.server --storage-path /data/memlayer

    # Debug mode
    python3.12 -m memlayer.server --debug
"""

import argparse
import sys
import logging
import os
import uvicorn

# Suppress warnings BEFORE importing anything else
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY_DISABLED'] = '1'

from .proxy import MemlayerProxy
from .config import ServerConfig, set_config
from ..config import set_debug_mode


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Memlayer Server - OpenAI-compatible API with persistent memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with defaults (llama-server at localhost:8080, proxy at 0.0.0.0:8000)
  python3.12 -m memlayer.server

  # Custom llama-server URL
  python3.12 -m memlayer.server --llama-host http://192.168.1.100:8080

  # Custom proxy settings
  python3.12 -m memlayer.server --proxy-host 127.0.0.1 --proxy-port 9000

  # Enable debug mode
  python3.12 -m memlayer.server --debug

Environment Variables:
  All settings can be configured via environment variables with MEMLAYER_ prefix.
  See memlayer/server/config.py for full list.
        """,
    )

    # llama-server connection
    parser.add_argument(
        "--llama-host",
        type=str,
        help="llama-server URL (default: http://localhost:8080)",
    )

    parser.add_argument(
        "--llama-port",
        type=int,
        help="llama-server port (optional if included in host URL)",
    )

    # Proxy settings
    parser.add_argument(
        "--proxy-host",
        type=str,
        help="Proxy server host address (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--proxy-port",
        type=int,
        help="Proxy server port (default: 8000)",
    )

    # Storage
    parser.add_argument(
        "--storage-path",
        type=str,
        help="Path for memory storage (default: ./memlayer_data)",
    )

    # Performance
    parser.add_argument(
        "--no-curation",
        action="store_true",
        help="Disable memory curation service",
    )

    parser.add_argument(
        "--curation-interval",
        type=int,
        help="Memory curation interval in seconds (default: 3600)",
    )

    parser.add_argument(
        "--scheduler-interval",
        type=int,
        help="Task scheduler interval in seconds (default: 60)",
    )

    parser.add_argument(
        "--max-consolidations",
        type=int,
        help="Maximum concurrent memory consolidations to avoid saturating llama-server (default: 2)",
    )

    # Logging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose logging",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    # Server options
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Create config from args and environment
    config = ServerConfig()

    # Override with CLI args
    if args.llama_host:
        config.llama_server_host = args.llama_host
    if args.llama_port:
        config.llama_server_port = args.llama_port
    if args.proxy_host:
        config.proxy_host = args.proxy_host
    if args.proxy_port:
        config.proxy_port = args.proxy_port
    if args.storage_path:
        config.storage_path = args.storage_path
    if args.no_curation:
        config.enable_curation = False
    if args.curation_interval:
        config.curation_interval = args.curation_interval
    if args.scheduler_interval:
        config.scheduler_interval = args.scheduler_interval
    if args.max_consolidations:
        config.max_concurrent_consolidations = args.max_consolidations
    if args.debug:
        config.debug_mode = True
        config.log_level = "DEBUG"
    if args.log_level:
        config.log_level = args.log_level

    # Set global config
    set_config(config)

    # Set debug mode globally
    if config.debug_mode:
        set_debug_mode(True)

    # Configure logging
    log_level = getattr(logging, config.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    # Print startup info
    print("=" * 70)
    print("🚀 Starting Memlayer Server")
    print("=" * 70)
    print(f"  llama-server: {config.llama_server_host}")
    print(f"  Proxy: http://{config.proxy_host}:{config.proxy_port}")
    print(f"  Storage: {config.storage_path}")
    print(f"  Mode: {config.operation_mode} (100% offline)")
    print(f"  Curation: {'enabled' if config.enable_curation else 'disabled'}")
    print(f"  Debug: {'enabled' if config.debug_mode else 'disabled'}")
    print("=" * 70)
    print("\n💡 Usage:")
    print("  OpenAI SDK:")
    print("    from openai import OpenAI")
    print(f"    client = OpenAI(base_url='http://localhost:{config.proxy_port}/v1', api_key='not-needed')")
    print("    response = client.chat.completions.create(...)")
    print("\n  Curl:")
    print(f"    curl http://localhost:{config.proxy_port}/v1/chat/completions \\")
    print("      -H 'Content-Type: application/json' \\")
    print("      -H 'X-User-ID: user123' \\")
    print("      -d '{\"model\": \"model\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'")
    print("\n  Health check:")
    print(f"    curl http://localhost:{config.proxy_port}/")
    print("=" * 70)
    print("")

    try:
        # Create proxy
        proxy = MemlayerProxy(
            llama_server_host=config.llama_server_host,
            llama_server_port=config.llama_server_port,
            storage_path=config.storage_path,
            config=config,
        )

        # Create FastAPI app
        app = proxy.create_app()

        # Run server
        uvicorn.run(
            app,
            host=config.proxy_host,
            port=config.proxy_port,
            reload=args.reload,
            workers=args.workers,
            log_level=config.log_level.lower(),
        )

    except KeyboardInterrupt:
        logger.info("\n\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
