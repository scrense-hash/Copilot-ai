"""Upstream OpenRouter API communication."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import string
import time
from typing import Any, Dict, Tuple

import httpx

from config import AppConfig

log = logging.getLogger("autorouter")

# Base62 alphabet for strict tool_call IDs (Mistral compatibility)
_BASE62_ALPHABET = string.ascii_letters + string.digits  # A-Za-z0-9


def _base62_from_sha1(s: str, length: int = 9) -> str:
    """Generate a base62 encoded string from SHA1 hash."""
    hash_bytes = hashlib.sha1(s.encode("utf-8", errors="ignore")).digest()
    num = int.from_bytes(hash_bytes, "big")
    result = []
    for _ in range(length):
        num, remainder = divmod(num, len(_BASE62_ALPHABET))
        result.append(_BASE62_ALPHABET[remainder])
    return "".join(result)


def build_tool_id_maps_from_messages(messages: Any) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build bidirectional maps for tool_call IDs: original <-> short (9-char base62).

    Needed for providers requiring strict ID format (e.g. Mistral).
    Returns: (orig_to_short, short_to_orig) dictionaries
    """
    orig_to_short: Dict[str, str] = {}
    short_to_orig: Dict[str, str] = {}

    if not isinstance(messages, list):
        return orig_to_short, short_to_orig

    def _register_tool_id(original_id: str) -> None:
        """Register a tool call ID with collision handling."""
        if original_id in orig_to_short:
            return

        # Generate short ID with collision resolution
        short_id = _base62_from_sha1(original_id, 9)
        salt = 0
        while short_id in short_to_orig and short_to_orig[short_id] != original_id:
            salt += 1
            short_id = _base62_from_sha1(f"{original_id}:{salt}", 9)

        orig_to_short[original_id] = short_id
        short_to_orig[short_id] = original_id

    # Extract all tool_call IDs from messages
    for message in messages:
        if not isinstance(message, dict):
            continue

        # Check tool_call_id field
        tool_call_id = message.get("tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            _register_tool_id(tool_call_id)

        # Check tool_calls array
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    tc_id = tool_call.get("id")
                    if isinstance(tc_id, str) and tc_id:
                        _register_tool_id(tc_id)

    return orig_to_short, short_to_orig


def translate_tool_ids_in_messages(messages: Any, orig_to_short: Dict[str, str]) -> Any:
    """
    Replace tool_call_id and tool_calls[].id with short ids for strict upstreams.
    """
    if not isinstance(messages, list) or not orig_to_short:
        return messages
    out = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        mm = dict(m)
        if isinstance(mm.get("tool_call_id"), str):
            mm["tool_call_id"] = orig_to_short.get(mm["tool_call_id"], mm["tool_call_id"])
        if isinstance(mm.get("tool_calls"), list):
            new_tcs = []
            for t in mm["tool_calls"]:
                if not isinstance(t, dict):
                    new_tcs.append(t)
                    continue
                tt = dict(t)
                if isinstance(tt.get("id"), str):
                    tt["id"] = orig_to_short.get(tt["id"], tt["id"])
                new_tcs.append(tt)
            mm["tool_calls"] = new_tcs
        out.append(mm)
    return out


class UpstreamClient:
    """Handle communication with OpenRouter API."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def get_headers(self) -> Dict[str, str]:
        """Get default headers for OpenRouter API."""
        return {
            "Authorization": f"Bearer {self._config.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._config.openrouter_http_referer,
            "X-Title": self._config.openrouter_x_title,
            "User-Agent": self._config.user_agent,
        }

    def get_proxy_url(self) -> str | None:
        """
        Get proxy URL for httpx AsyncClient.

        Returns HTTPS proxy if set (preferred for HTTPS API calls),
        otherwise HTTP proxy if set, or None if no proxy configured.
        """
        if self._config.https_proxy:
            log.info("HTTPS proxy configured: %s", self._config.https_proxy)
            return self._config.https_proxy
        if self._config.http_proxy:
            log.info("HTTP proxy configured: %s", self._config.http_proxy)
            return self._config.http_proxy
        return None

    async def chat_completion(
        self,
        client: httpx.AsyncClient,
        body: Dict[str, Any],
        model_id: str,
    ) -> httpx.Response:
        """
        Send chat completion request to OpenRouter.

        For streaming requests, the response must be streamed to avoid buffering.
        """
        headers = self.get_headers()

        messages = body.get("messages")
        orig_to_short, short_to_orig = build_tool_id_maps_from_messages(messages)

        body["_tool_id_map_short_to_orig"] = short_to_orig

        payload = dict(body)
        if orig_to_short:
            try:
                preview = list(orig_to_short.items())[:10]
                log.debug(
                    "tool_call_id map built model=%s entries=%d preview=%r",
                    model_id,
                    len(orig_to_short),
                    preview,
                )
            except Exception:
                pass
            payload["messages"] = translate_tool_ids_in_messages(messages, orig_to_short)

        payload["model"] = model_id

        stream = bool(payload.get("stream", False))

        t0 = time.time()

        req = client.build_request(
            "POST",
            f"{self._config.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp = await client.send(req, stream=stream)

        dt = (time.time() - t0) * 1000
        log.info("Upstream chat model=%s status=%s ms=%.1f", model_id, resp.status_code, dt)

        if resp.status_code != 200:
            log.warning(
                "Upstream chat error model=%s status=%s content-type=%s",
                model_id,
                resp.status_code,
                resp.headers.get("content-type", ""),
            )

        return resp

    @staticmethod
    async def read_error_snippet(
        resp: httpx.Response, limit: int = 2000, timeout_s: float = 2.0
    ) -> str:
        """Best-effort: read small error body without risking a hang."""
        try:
            raw = await asyncio.wait_for(resp.aread(), timeout=timeout_s)
        except Exception:
            return ""
        try:
            txt = raw.decode("utf-8", errors="replace")
        except Exception:
            return ""
        return txt[:limit]

    @staticmethod
    def payload_has_choices(payload: Any) -> bool:
        """Check if response payload has choices field."""
        return isinstance(payload, dict) and bool(payload.get("choices"))
