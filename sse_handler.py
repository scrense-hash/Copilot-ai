"""Server-Sent Events (SSE) handling for streaming responses."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
import uuid
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, List, Optional, Tuple

import httpx

log = logging.getLogger("copilot_ai")

SSEEventLines = List[str]


def sse_event_to_bytes(lines: SSEEventLines) -> bytes:
    """Serialize SSE event lines (without the terminating blank line) to bytes."""
    if not lines:
        # Keepalive / empty event separator
        return b"\n\n"
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def sse_event_data_text(lines: SSEEventLines) -> str:
    """
    Join all `data:` lines in an SSE event into a single payload.

    SSE spec concatenates multiple data lines with '\n'.
    """
    parts: List[str] = []
    for ln in lines:
        if ln.startswith("data:"):
            parts.append(ln[len("data:"):].lstrip())
    return "\n".join(parts)


def sse_event_has_non_activity_lines(lines: SSEEventLines) -> bool:
    """Return True if any event line violates the SSE field/comment/continuation format."""
    return any(ln and not is_sse_activity_line(ln) for ln in lines)


async def read_next_sse_event(
    aiter: AsyncIterator[str],
    *,
    timeout_s: float | None = None,
    on_line: Callable[[], None] | None = None,
) -> SSEEventLines | None:
    """
    Read one SSE event (blank-line delimited) from an async line iterator.

    Returns:
    - list[str]: event lines excluding the terminating blank line (may be empty for keepalive)
    - None: EOF (no more data)
    """
    lines: SSEEventLines = []
    while True:
        try:
            if timeout_s is None:
                raw = await aiter.__anext__()  # type: ignore[attr-defined]
            else:
                raw = await asyncio.wait_for(
                    aiter.__anext__(), timeout=timeout_s  # type: ignore[attr-defined]
                )
        except StopAsyncIteration:
            if lines:
                return lines
            return None

        if on_line is not None:
            with contextlib.suppress(Exception):
                on_line()

        line = raw.rstrip("\r\n")
        if line == "":
            return lines
        lines.append(line)


def is_sse_activity_line(line: str) -> bool:
    """
    Treat any SSE field/comment/continuation as activity to avoid watchdog/prebuffer false-positives.
    Fields: data, event, id, retry; comments ":"; and (rare) continuation lines that start with space.
    """
    return (
        line.startswith("data:")
        or line.startswith("event:")
        or line.startswith("id:")
        or line.startswith("retry:")
        or line.startswith(":")
        or line.startswith(" ")
    )


def is_done_data_line(line: str) -> bool:
    """
    Accept: "data:[DONE]" / "data: [DONE]" / "data:    [DONE]" (tolerate whitespace)
    """
    if not line.startswith("data:"):
        return False
    return line[len("data:"):].strip() == "[DONE]"


def chunk_has_done_data_line(b: bytes) -> bool:
    """Strictly detect SSE DONE event inside already-buffered bytes."""
    try:
        txt = b.decode("utf-8", errors="replace")
    except Exception:
        return False
    for ln in txt.splitlines():
        if ln and is_done_data_line(ln):
            return True
    return False


def looks_like_inline_tool_call(text: str) -> bool:
    """
    Check if text contains inline tool call markup.

    Requires exact signature to avoid false positives:
    - tool_call open+close tags
    - function and parameter markers
    """
    if not text:
        return False
    t = text.lower()
    return (
        "<tool_call" in t
        and "</tool_call>" in t
        and ("<function=" in t or "function=" in t)
        and ("<parameter=" in t or "parameter=" in t)
    )


def extract_content_fragments(obj: Any) -> List[str]:
    """Extract content fragments from SSE data object."""
    out: List[str] = []
    if not isinstance(obj, dict):
        return out

    for ch in (obj.get("choices") or []):
        if not isinstance(ch, dict):
            continue
        d = ch.get("delta") or ch.get("message") or {}
        if isinstance(d, dict):
            c = d.get("content")
            if isinstance(c, str) and c:
                out.append(c)
    return out


def normalize_upstream_chunk_to_openai(
    obj: Dict[str, Any],
    *,
    out_id: str,
    out_model: str,
    tool_id_map_short_to_orig: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Normalize upstream chunk to strict OpenAI chat.completion.chunk:
      {id, object, created, model, choices:[{index, delta:{role?, content?, tool_calls?}, finish_reason?}]}
    Provider-specific fields are dropped.
    If tool_id_map_short_to_orig is provided, tool call ids are mapped back (short->original).
    """
    if not isinstance(obj, dict):
        return None
    choices = obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    # --- Copilot compatibility ---
    # Некоторые провайдеры/модели вместо "tool_calls" JSON возвращают текстовые теги:
    #   <tool_call>
    #   <function name="glob_search">
    #   <parameter name="pattern">...</parameter>
    #   </function>
    #   </tool_call>
    #
    # Copilot ожидает структурированный OpenAI tool_calls (choices[].delta.tool_calls),
    # поэтому конвертируем такие текстовые блоки в delta.tool_calls и чистим content.
    _FUNC_RE = re.compile(r"<function\s+name=['\"]([^'\"]+)['\"]\s*>", re.I)
    _PARAM_RE = re.compile(r"<parameter\s+name=['\"]([^'\"]+)['\"]\s*>(.*?)</parameter>", re.I | re.S)
    _TOOLCALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.I | re.S)

    def _try_parse_toolcall_text(s: str) -> Optional[List[Dict[str, Any]]]:
        """
        Parse <tool_call>...</tool_call> blocks from plain text into structured tool_calls.

        Some providers emit tool calls as tagged text instead of native delta.tool_calls.
        We extract all parseable blocks. Returns None if tags exist but none are valid.
        """
        if not s or "<tool_call" not in s:
            return None

        tool_calls: List[Dict[str, Any]] = []

        for match in _TOOLCALL_BLOCK_RE.finditer(s):
            inner = (match.group(1) or "").strip()
            if not inner:
                continue

            # Extract function name
            func_match = _FUNC_RE.search(inner)
            if not func_match:
                continue
            function_name = func_match.group(1)

            # Extract parameters
            params: Dict[str, Any] = {}
            for param_match in _PARAM_RE.finditer(inner):
                param_name = (param_match.group(1) or "").strip()
                param_value = (param_match.group(2) or "").strip()
                if param_name:
                    params[param_name] = param_value

            # Build tool call structure
            tool_call_id = f"call_{uuid.uuid4().hex}"
            tool_calls.append({
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(params, ensure_ascii=False)
                },
            })

        return tool_calls if tool_calls else None

    out_choices: List[Dict[str, Any]] = []
    for ch in choices:
        if not isinstance(ch, dict):
            continue
        idx = ch.get("index", 0)
        finish = ch.get("finish_reason")
        d = ch.get("delta") or {}
        if not isinstance(d, dict):
            d = {}

        out_delta: Dict[str, Any] = {}
        if isinstance(d.get("role"), str):
            out_delta["role"] = d["role"]

        if "content" in d:
            if isinstance(d["content"], str):
                # Convert "<tool_call>...</tool_call>" text blocks into proper tool_calls for Copilot
                parsed_tc = _try_parse_toolcall_text(d["content"])
                if parsed_tc:
                    out_delta["tool_calls"] = parsed_tc
                    out_delta["content"] = ""  # important: tool_calls must not be embedded in content
                    # finish_reason may be "stop" upstream even though it is actually a tool call.
                    # For Copilot correctness, override to "tool_calls".
                    if finish in (None, "stop"):
                        finish = "tool_calls"
                else:
                    # If parsing failed, keep original content as-is.
                    # Copilot AI will detect unparseable/incomplete tool markup and failover to next model.
                    # We do NOT attempt to fix or strip partial tags - that would mask broken models.
                    out_delta["content"] = d["content"]
            elif d["content"] is None:
                out_delta["content"] = ""

        if isinstance(d.get("tool_calls"), list):
            norm_tc: List[Dict[str, Any]] = []
            for t in d["tool_calls"]:
                if not isinstance(t, dict):
                    continue
                tt = dict(t)
                if tool_id_map_short_to_orig:
                    if isinstance(tt.get("id"), str):
                        tt["id"] = tool_id_map_short_to_orig.get(tt["id"], tt["id"])
                    if isinstance(tt.get("tool_call_id"), str):
                        tt["tool_call_id"] = tool_id_map_short_to_orig.get(
                            tt["tool_call_id"], tt["tool_call_id"]
                        )
                norm_tc.append(tt)
            if norm_tc:
                existing = out_delta.get("tool_calls")
                if isinstance(existing, list):
                    existing.extend(norm_tc)
                else:
                    out_delta["tool_calls"] = norm_tc

        out_choices.append({"index": idx, "delta": out_delta, "finish_reason": finish})

    if not out_choices:
        return None

    created = obj.get("created")
    if not isinstance(created, int):
        created = int(time.time())

    return {
        "id": out_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": out_model,
        "choices": out_choices,
    }


class ToolCallTextRepairer:
    """Stateful repair for providers that emit tool calls as tagged text in delta.content.

    Many upstreams stream the <tool_call>...</tool_call> markup split across chunks.
    This helper buffers partial markup until it is complete, then converts it into native
    OpenAI-style delta.tool_calls while stripping the markup from content.

    Designed for buffered/atomic delivery: it may defer emitting content that contains a
    partial opening tag, to avoid leaking markup to the client.
    """

    _FUNC_RE = re.compile(r"<function\s+name=['\"]([^'\"]+)['\"]\s*>", re.I)
    _PARAM_RE = re.compile(r"<parameter\s+name=['\"]([^'\"]+)['\"]\s*>(.*?)</parameter>", re.I | re.S)
    _TOOLCALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.I | re.S)

    def __init__(self) -> None:
        self._pending: str = ""

    def reset(self) -> None:
        self._pending = ""

    @staticmethod
    def _parse_blocks(text: str) -> list[dict]:
        tool_calls: list[dict] = []
        for match in ToolCallTextRepairer._TOOLCALL_BLOCK_RE.finditer(text):
            inner = (match.group(1) or "").strip()
            if not inner:
                continue
            func_match = ToolCallTextRepairer._FUNC_RE.search(inner)
            if not func_match:
                continue
            function_name = func_match.group(1)

            params: dict[str, object] = {}
            for pm in ToolCallTextRepairer._PARAM_RE.finditer(inner):
                pn = (pm.group(1) or "").strip()
                pv = (pm.group(2) or "").strip()
                if not pn:
                    continue
                if pv and pv[0] in "[{" and pv[-1] in "]}":
                    try:
                        params[pn] = json.loads(pv)
                    except Exception:
                        params[pn] = pv
                else:
                    params[pn] = pv

            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(params, ensure_ascii=False),
                    },
                }
            )
        return tool_calls

    def consume(self, content: str) -> tuple[str, list[dict] | None, bool]:
        """Consume a content fragment.

        Returns: (safe_text, tool_calls_or_none, deferred)
          - safe_text: content with any *completed* tool_call markup removed.
                       If a partial opening tag is seen, text from the tag onward is deferred.
          - tool_calls_or_none: parsed tool_calls when at least one complete block is available.
          - deferred: True when we are buffering a partial markup and should avoid emitting it.
        """
        content = content or ""

        if not self._pending and "<tool_call" not in content:
            return content, None, False

        if self._pending:
            self._pending += content
            text = self._pending
        else:
            idx = content.lower().find("<tool_call")
            if idx >= 0:
                prefix = content[:idx]
                self._pending = content[idx:]
                return prefix, None, True
            return content, None, False

        blocks = self._parse_blocks(text)
        if not blocks:
            if "<tool_call" not in text:
                self._pending = ""
                return text, None, False
            return "", None, True

        remainder = self._TOOLCALL_BLOCK_RE.sub("", text)
        low = remainder.lower()
        if "<tool_call" in low and "</tool_call>" not in low:
            idx = low.find("<tool_call")
            emit = remainder[:idx]
            self._pending = remainder[idx:]
            return emit, blocks, True

        self._pending = ""
        return remainder, blocks, False


class SSEValidator:
    """Validate and prebuffer SSE streams before forwarding to client."""

    def __init__(self) -> None:
        # Stateful repair of "<tool_call>...</tool_call>" markup split across chunks.
        self._tool_repair = ToolCallTextRepairer()

    def reset_tool_repair(self) -> None:
        self._tool_repair.reset()

    def repair_tool_markup_inplace(self, obj: Dict[str, Any]) -> bool:
        """Best-effort repair of tagged tool_call markup in-place.

        Returns True if the chunk is safe to emit/normalize now.
        Returns False only when this chunk carries *only* deferred markup (no safe text/tool_calls),
        so it should be skipped until the markup closes.
        """
        try:
            choices = obj.get("choices") or []
            if not isinstance(choices, list) or not choices:
                return True
            ch0 = choices[0]
            if not isinstance(ch0, dict):
                return True
            delta = (ch0.get("delta") or {})
            if not isinstance(delta, dict):
                return True
            content = delta.get("content")
            if not isinstance(content, str) or not content:
                return True

            safe_text, tool_calls, deferred = self._tool_repair.consume(content)

            if tool_calls:
                existing = delta.get("tool_calls")
                if isinstance(existing, list):
                    existing.extend(tool_calls)
                else:
                    delta["tool_calls"] = tool_calls

                # If upstream claims stop/None, this is actually tool_calls; align finish_reason.
                if ch0.get("finish_reason") in (None, "stop"):
                    ch0["finish_reason"] = "tool_calls"

            delta["content"] = safe_text

            # Emit if we produced any safe content or any tool_calls; otherwise defer.
            if safe_text or tool_calls:
                return True
            return not deferred
        except Exception:
            return True

    async def peek_first_sse_choices(
        self,
        aiter: AsyncIterator[str],
    ) -> Tuple[bool, SSEEventLines | None, AsyncIterator[str] | None, str]:
        """
        Peek the first SSE event with JSON `data:` and validate that it is usable.

        Treat as MODEL FAILURE (trigger failover) when:
        - choices[].error exists
        - choices[].finish_reason == "error"
        - choices[].native_finish_reason == "error"
        """
        try:
            while True:
                ev = await read_next_sse_event(aiter)
                if ev is None:
                    return False, None, aiter, "early-eof: no SSE events"

                json_payload: str | None = None
                for line in ev:
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data or data == "[DONE]":
                        continue
                    if data.startswith("{") and data.endswith("}"):
                        json_payload = data
                        break

                if json_payload is None:
                    continue  # keepalive / processing / etc.

                try:
                    obj = json.loads(json_payload)
                except json.JSONDecodeError:
                    return False, None, aiter, "bad-json: first data line is not valid JSON"

                choices = obj.get("choices")
                if not isinstance(choices, list) or not choices:
                    return False, None, aiter, "no-choices: first JSON chunk has no choices[]"

                for ch in choices:
                    if not isinstance(ch, dict):
                        continue
                    if ch.get("error") is not None:
                        return False, None, aiter, "choice-error: choices[].error present"
                    if ch.get("finish_reason") == "error":
                        return False, None, aiter, "choice-error: finish_reason == error"
                    if ch.get("native_finish_reason") == "error":
                        return False, None, aiter, "choice-error: native_finish_reason == error"

                return True, ev, aiter, ""
        except Exception as e:
            return False, None, aiter, f"peek-exception: {type(e).__name__}: {e}"

    async def prebuffer_before_commit(
        self,
        aiter: AsyncIterator[str],
        first_bytes: bytes,
        min_data_events: int,
        window_s: float,
        idle_timeout_s: float,
    ) -> Tuple[bool, List[bytes], str | None]:
        """
        Buffer initial SSE window before sending to client.

        If stream dies/stalls here, we can switch to next model without
        breaking client UX.

        Returns: (ok, initial_bytes_list, reason)
        """
        initial: List[bytes] = [first_bytes]

        # Sniff first event for inline tool markup
        if not await self._check_first_event(first_bytes):
            return False, initial, "inline-tool-markup"

        data_count = 1  # first_bytes is valid data: with choices
        start = time.monotonic()
        last_activity = time.monotonic()

        def mark_activity() -> None:
            nonlocal last_activity
            last_activity = time.monotonic()

        while True:
            now = time.monotonic()

            # Commit as soon as we have enough data events
            if data_count >= max(1, min_data_events):
                return True, initial, None

            # Max window reached, commit anyway
            if window_s > 0 and (now - start) >= window_s:
                return True, initial, None

            # Idle timeout
            if (now - last_activity) > idle_timeout_s:
                return False, initial, "prebuffer-idle-timeout"

            try:
                event_lines = await read_next_sse_event(
                    aiter,
                    timeout_s=min(1.0, idle_timeout_s),
                    on_line=mark_activity,
                )
            except StopAsyncIteration:
                # Upstream closes before DONE and before commit
                if data_count < max(1, min_data_events):
                    return False, initial, "early-eof-before-commit"
                return True, initial, None
            except Exception as e:
                return False, initial, f"prebuffer-read-error:{type(e).__name__}"

            if event_lines is None:
                # EOF
                if data_count < max(1, min_data_events):
                    return False, initial, "early-eof-before-commit"
                return True, initial, None

            if not event_lines:
                # keepalive / blank event separator lines still indicate liveness
                initial.append(sse_event_to_bytes(event_lines))
                continue

            # Guard against upstream injecting non-SSE (e.g. JSON error bodies) into a stream.
            if sse_event_has_non_activity_lines(event_lines):
                return False, initial, "non-sse-line-before-commit"

            data = sse_event_data_text(event_lines)
            if data.strip() == "[DONE]":
                initial.append(sse_event_to_bytes(event_lines))
                return True, initial, None

            if data:
                # Check for inline tool markup
                try:
                    obj = json.loads(data)
                    for frag in extract_content_fragments(obj):
                        if looks_like_inline_tool_call(frag):
                            return False, initial, "inline-tool-markup"
                except Exception:
                    pass
                data_count += 1

            initial.append(sse_event_to_bytes(event_lines))

    async def _check_first_event(self, first_bytes: bytes) -> bool:
        """Check first event for inline tool markup."""
        try:
            txt = first_bytes.decode("utf-8", errors="replace")
            lines = [ln for ln in txt.splitlines() if ln.strip() != ""]
            data = sse_event_data_text(lines)
            if data and data.strip() != "[DONE]":
                _obj = json.loads(data)
                for frag in extract_content_fragments(_obj):
                    if looks_like_inline_tool_call(frag):
                        return False
        except Exception:
            pass
        return True


class SSEStreamer:
    """Stream SSE events with watchdog for stalled connections."""

    @staticmethod
    async def stream_with_watchdog(
        client: httpx.AsyncClient,
        resp: httpx.Response,
        aiter: AsyncIterator[str],
        initial_bytes: List[bytes],
        idle_timeout_s: float,
        req_id: str,
        model_id: str,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream SSE with watchdog that closes connection if idle too long.

        Does not throw exception to client - just ends stream.
        """
        last_ts = time.monotonic()
        stalled = False
        seen_done = False
        cancelled = False

        async def watchdog() -> None:
            nonlocal stalled
            try:
                while True:
                    await asyncio.sleep(1.0)
                    if time.monotonic() - last_ts > idle_timeout_s:
                        stalled = True
                        log.warning(
                            "Upstream SSE stalled (>%.1fs). Closing upstream connection. req_id=%s model=%s",
                            idle_timeout_s,
                            req_id,
                            model_id,
                        )
                        await resp.aclose()
                        return
            except asyncio.CancelledError:
                return

        wd = asyncio.create_task(watchdog())
        try:
            for b in initial_bytes:
                # Track DONE in initial bytes too (prebuffer may already contain it)
                # Strict matching avoids false positives inside JSON strings.
                if chunk_has_done_data_line(b):
                    seen_done = True
                yield b

            def mark_activity() -> None:
                nonlocal last_ts
                last_ts = time.monotonic()

            while True:
                event_lines = await read_next_sse_event(aiter, on_line=mark_activity)
                if event_lines is None:
                    break

                if not event_lines:
                    # keepalive / blank event separator
                    yield sse_event_to_bytes(event_lines)
                    continue

                if sse_event_has_non_activity_lines(event_lines):
                    log.warning(
                        "Non-SSE line from upstream SSE; closing connection. req_id=%s model=%s line=%r",
                        req_id,
                        model_id,
                        (event_lines[0] if event_lines else "")[:200],
                    )
                    stalled = True  # ensures we emit a clean DONE in finally (unless already seen)
                    await resp.aclose()
                    break

                if sse_event_data_text(event_lines).strip() == "[DONE]":
                    seen_done = True

                yield sse_event_to_bytes(event_lines)
        except asyncio.CancelledError:
            cancelled = True
            raise
        except Exception as e:
            if not stalled:
                log.warning(
                    "SSE passthrough ended with error req_id=%s model=%s err=%r",
                    req_id,
                    model_id,
                    e,
                )
        finally:
            # If watchdog closed upstream due to stall, try to end the SSE cleanly.
            # Only send DONE if upstream didn't already provide it.
            if not cancelled and stalled and not seen_done:
                yield b"data: [DONE]\n\n"
            wd.cancel()
            with contextlib.suppress(Exception):
                await resp.aclose()
            with contextlib.suppress(Exception):
                await client.aclose()

    @staticmethod
    async def error_response(message: str, model_id: str) -> AsyncGenerator[bytes, None]:
        """
        Emit single SSE event with error message, then [DONE].

        This prevents Copilot "Sorry, no response was returned." message.
        """
        now = int(time.time())
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": now,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": message},
                    "finish_reason": "stop",
                }
            ],
        }
        yield ("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode("utf-8")
        yield b"data: [DONE]\n\n"
