"""Model management and selection for Copilot AI."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

from config import AppConfig

log = logging.getLogger("copilot_ai")


@dataclass
class ModelInfo:
    """Information about an OpenRouter model."""

    id: str
    name: str
    context_length: int
    prompt_price: float
    completion_price: float
    supported_parameters: List[str]

    @property
    def max_price(self) -> float:
        """Get the maximum price (prompt or completion)."""
        return max(self.prompt_price, self.completion_price)

    def has_tools_support(self) -> bool:
        """Check if model supports tools/function calling."""
        supported_params = set(self.supported_parameters or [])
        return "tools" in supported_params or "tool_choice" in supported_params

    def to_virtual_model_dict(self, virtual_id: str, virtual_name: str) -> Dict[str, Any]:
        """Convert to virtual model dictionary for API response."""
        return {
            "id": virtual_id,
            "object": "model",
            "owned_by": "copilot-ai",
            "name": virtual_name,
            "context_length": self.context_length,
            "supported_parameters": self.supported_parameters,
            "pricing": {"prompt": self.prompt_price, "completion": self.completion_price},
            "upstream_model_id": self.id,
            "upstream_model_name": self.name,
        }


class ModelCache:
    """Cache for OpenRouter models with automatic refresh."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._models: List[ModelInfo] = []
        self._last_fetch = 0.0

    async def get_models(
        self, client: httpx.AsyncClient, config: AppConfig
    ) -> List[ModelInfo]:
        """Get cached models or fetch fresh ones if expired."""
        now = time.time()
        if self._models and (now - self._last_fetch) < config.refresh_models_s:
            return self._models

        async with self._lock:
            now = time.time()
            if self._models and (now - self._last_fetch) < config.refresh_models_s:
                return self._models
            models = await self._fetch_models(client, config)
            self._models = models
            self._last_fetch = time.time()
            return self._models

    async def _fetch_models(
        self, client: httpx.AsyncClient, config: AppConfig
    ) -> List[ModelInfo]:
        """Fetch models from OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {config.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": config.openrouter_http_referer,
            "X-Title": config.openrouter_x_title,
            "User-Agent": config.user_agent,
        }

        t0 = time.time()
        r = await client.get(f"{config.openrouter_base_url}/models", headers=headers)
        dt = (time.time() - t0) * 1000

        if r.status_code != 200:
            log.error(
                "Upstream /models failed status=%s ms=%.1f body=%s",
                r.status_code,
                dt,
                r.text[:500],
            )
            raise Exception(f"Upstream /models error: {r.status_code} {r.text[:2000]}")

        data = r.json()
        items = data.get("data", [])
        out: List[ModelInfo] = []

        for it in items:
            model_info = self._parse_model(it)
            if model_info:
                out.append(model_info)

        log.info("Fetched models: count=%d ms=%.1f", len(out), dt)
        return out

    def _parse_model(self, data: Dict[str, Any]) -> Optional[ModelInfo]:
        """Parse a single model from API response."""
        mid = data.get("id") or ""
        if not mid:
            return None

        pricing = data.get("pricing") or {}
        prompt = self._parse_price(pricing.get("prompt"))
        completion = self._parse_price(pricing.get("completion"))

        if prompt is None or completion is None:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "Model skipped due to invalid price: id=%s prompt=%r completion=%r",
                    mid,
                    pricing.get("prompt"),
                    pricing.get("completion"),
                )
            return None

        ctx = data.get("context_length") or data.get("context") or 0
        try:
            ctx = int(ctx)
        except Exception:
            ctx = 0

        supported = data.get("supported_parameters") or []
        if not isinstance(supported, list):
            supported = []

        return ModelInfo(
            id=mid,
            name=data.get("name") or mid,
            context_length=ctx,
            prompt_price=float(prompt),
            completion_price=float(completion),
            supported_parameters=[str(x) for x in supported],
        )

    @staticmethod
    def _parse_price(value: Any) -> Optional[float]:
        """Parse and validate a price value."""
        if value is None or value == "":
            return None
        try:
            v = float(value)
        except Exception:
            return None
        if v < 0:
            return None
        return v


class ModelSelector:
    """Select appropriate models based on criteria."""

    def choose_candidates(
        self,
        models: List[ModelInfo],
        config: AppConfig,
    ) -> List[ModelInfo]:
        """
        Filter and sort models based on configuration.

        Returns models sorted by priority, then price, then context length.
        """
        filtered = self._filter_models(models, config)
        return self._sort_models(filtered, config.priority_models)

    def _filter_models(
        self, models: List[ModelInfo], config: AppConfig
    ) -> List[ModelInfo]:
        """Filter models based on requirements and bans."""
        filtered: List[ModelInfo] = []

        for model in models:
            # Skip banned models
            if model.id in config.banned_models:
                continue

            # Require minimum context length
            if model.context_length <= 0 or model.context_length < config.min_context_length:
                continue

            # Require tools support
            if not model.has_tools_support():
                continue

            # Check price limit
            if model.max_price > config.max_price:
                continue

            filtered.append(model)

        return filtered

    def _sort_models(
        self, models: List[ModelInfo], priority_set: Set[str]
    ) -> List[ModelInfo]:
        """Sort models by priority, then price, then context length."""
        pri = [m for m in models if m.id in priority_set]
        rest = [m for m in models if m.id not in priority_set]

        def sort_key(m: ModelInfo) -> Tuple[float, int]:
            return (m.max_price, -m.context_length)

        pri.sort(key=sort_key)
        rest.sort(key=sort_key)
        return pri + rest
