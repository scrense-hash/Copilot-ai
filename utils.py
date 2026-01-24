"""Utility functions for Copilot AI."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("copilot_ai")


def load_env_files() -> None:
    """Load .env files from program and current directory."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        log.warning("python-dotenv not installed; .env will NOT be loaded automatically.")
        return

    this_dir = Path(__file__).resolve().parent
    p1 = this_dir / ".env"
    p2 = Path.cwd() / ".env"

    loaded_any = False
    if p1.exists():
        loaded_any = load_dotenv(dotenv_path=str(p1), override=True) or loaded_any
        log.info("Loaded .env from %s", str(p1))
    else:
        log.info("No .env in program directory: %s", str(p1))

    if p2.exists() and p2 != p1:
        loaded_any = load_dotenv(dotenv_path=str(p2), override=True) or loaded_any
        log.info("Loaded .env from %s", str(p2))
    elif p2 != p1:
        log.info("No .env in current directory: %s", str(p2))

    if not loaded_any:
        log.info(".env not loaded (not found or no variables applied).")


def dump_config(config) -> None:
    """Log effective configuration at startup."""
    from logger import mask_secret

    log.info("=== Copilot AI startup config ===")
    log.info("OPENROUTER_BASE_URL=%s", config.openrouter_base_url)
    log.info("VIRTUAL_MODEL_ID=%s", config.virtual_model_id)
    log.info("VIRTUAL_MODEL_NAME=%s", config.virtual_model_name)
    log.info("MIN_CTX=%s", config.min_context_length)
    log.info(
        "OPENROUTER_API_KEY_set=%s value=%s len=%s",
        bool(config.openrouter_api_key),
        mask_secret(config.openrouter_api_key),
        len(config.openrouter_api_key or ""),
    )
    log.info("OPENROUTER_HTTP_REFERER=%s", config.openrouter_http_referer)
    log.info("OPENROUTER_X_TITLE=%s", config.openrouter_x_title)
    log.info("MAX_PRICE=%s", config.max_price)
    if float(config.max_price) == 0.0:
        log.info("MAX_PRICE=0.0 means only free models will pass price filter.")
    log.info("PRIORITY_MODELS=%s", sorted(config.priority_models))
    log.info("BAN_MODELS=%s", sorted(config.banned_models))
    log.info("REQUEST_TIMEOUT_S=%s", config.request_timeout_s)
    log.info("STREAM_IDLE_TIMEOUT_S=%s", config.stream_idle_timeout_s)
    log.info("BUFFER_STREAM_KEEPALIVE_S=%s", getattr(config, "buffer_stream_keepalive_s", None))
    log.info("MAX_BUFFERED_SSE_BYTES=%s", getattr(config, "max_buffered_sse_bytes", None))
    log.info("REFRESH_MODELS_S=%s", config.refresh_models_s)
    log.info("LOG_LEVEL=%s", config.log_level)
    log.info("DEBUG_SSE_TRAFFIC=%s", getattr(config, "debug_sse_traffic", None))
    log.info("DEBUG_SSE_TRAFFIC_LOG_PATH=%s", getattr(config, "debug_sse_traffic_log_path", None))
    log.info("DEBUG_SSE_TRAFFIC_TRUNCATE_BYTES=%s", getattr(config, "debug_sse_traffic_truncate_bytes", None))
    log.info("DEBUG_SSE_TRAFFIC_MAX_BYTES=%s", getattr(config, "debug_sse_traffic_max_bytes", None))
    log.info("DEBUG_SSE_TRAFFIC_BACKUP_COUNT=%s", getattr(config, "debug_sse_traffic_backup_count", None))
    log.info("DEBUG_SSE_TRAFFIC_MIRROR_MAIN=%s", getattr(config, "debug_sse_traffic_mirror_main", None))
    log.info(
        "DEBUG_SSE_TRAFFIC_MIRROR_MAIN_MAX_CHUNKS=%s",
        getattr(config, "debug_sse_traffic_mirror_main_max_chunks", None),
    )
    log.info("MAX_REQUEST_BYTES=%s", getattr(config, "max_request_bytes", None))
    log.info("LOG_PATH=%s", getattr(config, "log_path", None))
    log.info("USER_AGENT=%s", getattr(config, "user_agent", None))
    log.info("WorkingDir=%s", str(Path.cwd()))
    log.info("ProgramDir=%s", str(Path(__file__).resolve().parent))
    log.info("===============================")
