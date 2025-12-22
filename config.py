"""Configuration management for Autorouter service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Set


def _env_bool(name: str, default: bool) -> bool:
    """Get boolean environment variable with fallback."""
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    """Get float environment variable with fallback."""
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Get integer environment variable with fallback."""
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v.strip())
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    """Get string environment variable with fallback."""
    v = os.getenv(name)
    if v is None:
        return default
    return v


def _csv_set(name: str) -> Set[str]:
    """Parse comma-separated environment variable into a set."""
    v = os.getenv(name, "")
    items = [x.strip() for x in v.split(",") if x.strip() and x.strip().lower() != "empty"]
    return set(items)


@dataclass(frozen=True)
class AppConfig:
    """Application configuration."""

    # OpenRouter settings
    openrouter_base_url: str
    openrouter_api_key: str
    openrouter_http_referer: str
    openrouter_x_title: str

    # Virtual model settings
    virtual_model_id: str
    virtual_model_name: str

    # Model filtering
    min_context_length: int
    max_price: float
    priority_models: Set[str]
    banned_models: Set[str]

    # Timeouts and limits
    request_timeout_s: float
    stream_idle_timeout_s: float
    refresh_models_s: int

    # Buffered-stream settings (client stream=True, upstream stream=True, but we only forward after [DONE])
    buffer_stream_keepalive_s: float
    max_buffered_sse_bytes: int

    # Debug traffic logging (VERY VERBOSE)
    debug_sse_traffic: bool
    debug_sse_traffic_log_path: str
    debug_sse_traffic_truncate_bytes: int
    debug_sse_traffic_max_bytes: int
    debug_sse_traffic_backup_count: int
    # Mirror traffic log to main autorouter logger (helps when file handler doesn't write)
    debug_sse_traffic_mirror_main: bool
    debug_sse_traffic_mirror_main_max_chunks: int

    # Server settings
    port: int
    log_level: str
    max_request_bytes: int
    log_path: str
    user_agent: str

    @classmethod
    def from_env(cls) -> AppConfig:
        """Load configuration from environment variables."""
        return cls(
            openrouter_base_url="https://openrouter.ai/api/v1",
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openrouter_http_referer=os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
            openrouter_x_title=os.getenv("OPENROUTER_X_TITLE", "copilot-autorouter"),
            virtual_model_id="copilot-autorouter",
            virtual_model_name="Copilot Autorouter",
            min_context_length=_env_int("MIN_CTX", 131072),
            max_price=_env_float("MAX_PRICE", 0.0),
            priority_models=_csv_set("PRIORITY_MODELS"),
            banned_models=_csv_set("BAN_MODELS"),
            request_timeout_s=_env_float("REQUEST_TIMEOUT_S", 60.0),
            stream_idle_timeout_s=_env_float("STREAM_IDLE_TIMEOUT_S", 20.0),
            refresh_models_s=_env_int("REFRESH_MODELS_S", 300),
            buffer_stream_keepalive_s=_env_float("BUFFER_STREAM_KEEPALIVE_S", 5.0),
            max_buffered_sse_bytes=_env_int("MAX_BUFFERED_SSE_BYTES", 20_000_000),
            debug_sse_traffic=_env_bool("DEBUG_SSE_TRAFFIC", False),
            debug_sse_traffic_log_path=_env_str("DEBUG_SSE_TRAFFIC_LOG_PATH", "traffic_sse.log"),
            debug_sse_traffic_truncate_bytes=_env_int("DEBUG_SSE_TRAFFIC_TRUNCATE_BYTES", 0),  # 0 = no truncate
            debug_sse_traffic_max_bytes=_env_int("DEBUG_SSE_TRAFFIC_MAX_BYTES", 100_000_000),
            debug_sse_traffic_backup_count=_env_int("DEBUG_SSE_TRAFFIC_BACKUP_COUNT", 3),
            debug_sse_traffic_mirror_main=_env_bool("DEBUG_SSE_TRAFFIC_MIRROR_MAIN", True),
            debug_sse_traffic_mirror_main_max_chunks=_env_int("DEBUG_SSE_TRAFFIC_MIRROR_MAIN_MAX_CHUNKS", 5),
            port=int(os.getenv("PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper().strip(),
            max_request_bytes=_env_int("MAX_REQUEST_BYTES", 2_000_000),  # ~2MB
            log_path=_env_str("LOG_PATH", "/var/log/autorouter/autorouter.log"),
            user_agent=_env_str("USER_AGENT", "copilot-autorouter/0.8.0 (+https://openrouter.ai)"),
        )

    def validate(self, require_api_key: bool = True) -> None:
        """Validate configuration."""
        if require_api_key and not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        if self.min_context_length <= 0:
            raise ValueError("MIN_CTX must be > 0")
        if self.max_price < 0:
            raise ValueError("MAX_PRICE must be >= 0")
        if self.request_timeout_s <= 0:
            raise ValueError("REQUEST_TIMEOUT_S must be > 0")
        if self.stream_idle_timeout_s <= 0:
            raise ValueError("STREAM_IDLE_TIMEOUT_S must be > 0")
        if self.buffer_stream_keepalive_s <= 0:
            raise ValueError("BUFFER_STREAM_KEEPALIVE_S must be > 0")
        if self.max_buffered_sse_bytes <= 0:
            raise ValueError("MAX_BUFFERED_SSE_BYTES must be > 0")
        if self.refresh_models_s <= 0:
            raise ValueError("REFRESH_MODELS_S must be > 0")
        if self.debug_sse_traffic_truncate_bytes < 0:
            raise ValueError("DEBUG_SSE_TRAFFIC_TRUNCATE_BYTES must be >= 0")
        if self.debug_sse_traffic_max_bytes <= 0:
            raise ValueError("DEBUG_SSE_TRAFFIC_MAX_BYTES must be > 0")
        if self.debug_sse_traffic_backup_count < 0:
            raise ValueError("DEBUG_SSE_TRAFFIC_BACKUP_COUNT must be >= 0")
        if self.debug_sse_traffic_mirror_main_max_chunks < 0:
            raise ValueError("DEBUG_SSE_TRAFFIC_MIRROR_MAIN_MAX_CHUNKS must be >= 0")
        if self.max_request_bytes <= 0:
            raise ValueError("MAX_REQUEST_BYTES must be > 0")
        if not self.log_path:
            raise ValueError("LOG_PATH must be non-empty")
        if not self.user_agent:
            raise ValueError("USER_AGENT must be non-empty")


def load_config() -> AppConfig:
    """Load configuration from environment."""
    return AppConfig.from_env()
