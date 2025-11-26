"""
Configuration management for Memlayer Server.

Supports configuration via environment variables for deployment flexibility.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """
    Configuration for Memlayer proxy server.

    All settings can be overridden via environment variables with MEMLAYER_ prefix.
    Example: MEMLAYER_LLAMA_SERVER_HOST=http://localhost:8080
    """

    # ========================================================================
    # llama-server Connection
    # ========================================================================
    llama_server_host: str = Field(
        default=os.getenv("MEMLAYER_LLAMA_SERVER_HOST", "http://localhost:8080"),
        description="Base URL of llama-server instance"
    )

    llama_server_port: Optional[int] = Field(
        default=int(os.getenv("MEMLAYER_LLAMA_SERVER_PORT", "0")) or None,
        description="Port number (optional if included in host URL)"
    )

    # ========================================================================
    # Memlayer Settings (100% Offline)
    # ========================================================================
    storage_path: str = Field(
        default=os.getenv("MEMLAYER_STORAGE_PATH", "./memlayer_data"),
        description="Path where memories will be stored"
    )

    operation_mode: str = Field(
        default=os.getenv("MEMLAYER_OPERATION_MODE", "local"),
        description="Always 'local' for offline operation with sentence-transformers"
    )

    salience_threshold: float = Field(
        default=float(os.getenv("MEMLAYER_SALIENCE_THRESHOLD", "0.0")),
        description="Memory filtering threshold (-0.1 to 0.2)"
    )

    # ========================================================================
    # Proxy Server Settings
    # ========================================================================
    proxy_host: str = Field(
        default=os.getenv("MEMLAYER_PROXY_HOST", "0.0.0.0"),
        description="Host address for proxy server"
    )

    proxy_port: int = Field(
        default=int(os.getenv("MEMLAYER_PROXY_PORT", "8000")),
        description="Port for proxy server"
    )

    # ========================================================================
    # Performance & Lifecycle
    # ========================================================================
    enable_curation: bool = Field(
        default=os.getenv("MEMLAYER_ENABLE_CURATION", "true").lower() == "true",
        description="Enable memory curation service"
    )

    curation_interval: int = Field(
        default=int(os.getenv("MEMLAYER_CURATION_INTERVAL", "3600")),
        description="Memory curation interval in seconds (default: 1 hour)"
    )

    scheduler_interval: int = Field(
        default=int(os.getenv("MEMLAYER_SCHEDULER_INTERVAL", "60")),
        description="Task scheduler check interval in seconds (default: 1 minute)"
    )

    max_concurrent_consolidations: int = Field(
        default=int(os.getenv("MEMLAYER_MAX_CONCURRENT_CONSOLIDATIONS", "2")),
        description="Maximum concurrent consolidation operations to avoid saturating llama-server (default: 2)"
    )

    # ========================================================================
    # User Management
    # ========================================================================
    default_user_id: str = Field(
        default=os.getenv("MEMLAYER_DEFAULT_USER_ID", "default_user"),
        description="Default user ID when X-User-ID header not provided"
    )

    # ========================================================================
    # Logging & Debug
    # ========================================================================
    log_level: str = Field(
        default=os.getenv("MEMLAYER_LOG_LEVEL", "INFO"),
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )

    debug_mode: bool = Field(
        default=os.getenv("MEMLAYER_DEBUG_MODE", "false").lower() == "true",
        description="Enable debug mode for verbose logging"
    )

    # ========================================================================
    # Model Settings
    # ========================================================================
    default_model: str = Field(
        default=os.getenv("MEMLAYER_DEFAULT_MODEL", "model"),
        description="Default model name for llama-server"
    )

    default_temperature: float = Field(
        default=float(os.getenv("MEMLAYER_DEFAULT_TEMPERATURE", "0.7")),
        description="Default temperature for completions"
    )

    class Config:
        env_prefix = "MEMLAYER_"
        case_sensitive = False


# Global config instance
_config: Optional[ServerConfig] = None


def get_config() -> ServerConfig:
    """Get or create the global server configuration"""
    global _config
    if _config is None:
        _config = ServerConfig()
    return _config


def set_config(config: ServerConfig):
    """Set the global server configuration"""
    global _config
    _config = config
