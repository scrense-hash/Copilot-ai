"""
Copilot AI service (OpenAI-compatible) -> OpenRouter as upstream.

This refactored version provides:
- Better code organization with separate modules
- Improved maintainability and testability
- Clear separation of concerns
- Type hints throughout

Virtual model:
  id="copilot-ai"
  name="Copilot AI"

Hard filters:
  tools support + context_length >= 131072
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from config import load_config
from logger import setup_logging
from models import ModelCache, ModelInfo, ModelSelector
from sse_handler import (
    SSEStreamer,
    SSEValidator,
    normalize_upstream_chunk_to_openai,
    read_next_sse_event,
    sse_event_data_text,
    sse_event_to_bytes,
)
from upstream import UpstreamClient
from utils import dump_config, load_env_files

# "DONE roulette" mode: accept native tool_calls (Copilot needs them!),
# but reject text-based tool markup leaked into content.
# We keep only two safety checks elsewhere: idle-timeout + max buffered bytes.
# Note: markers are checked case-insensitively in _contains_tool_garbage_text
TOOL_GARBAGE_MARKERS = (
    "<tool_call",
    "</tool_call>",
    "<function=",
    "</function>",
    "<function",  # Also catch incomplete/malformed function tags
)


def _contains_tool_garbage_text(s: str) -> bool:
    """Check if string contains any tool garbage markers (case-insensitive)."""
    if not s:
        return False
    low = s.lower()
    return any(m in low for m in TOOL_GARBAGE_MARKERS)


def _get_tool_garbage_marker(s: str) -> str | None:
    """Return the first tool garbage marker found in string, or None."""
    if not s:
        return None
    low = s.lower()
    for marker in TOOL_GARBAGE_MARKERS:
        if marker in low:
            return marker
    return None


def _chunk_has_tool_garbage_text(norm: dict) -> bool:
    """Check if normalized chunk contains tool garbage text markers in content."""
    try:
        for ch in norm.get("choices") or []:
            delta = ch.get("delta") or {}
            content = delta.get("content") or ""
            if _contains_tool_garbage_text(content):
                return True
    except Exception:
        return False
    return False


def _check_and_log_tool_garbage(
    norm: dict,
    req_id: str,
    model_id: str,
    phase: str = "unknown"
) -> bool:
    """
    Check if chunk has tool garbage and log details if found.

    Returns True if garbage detected, False otherwise.
    """
    if not _chunk_has_tool_garbage_text(norm):
        return False

    # Log details about what triggered the rejection
    details = []
    try:
        for ch in norm.get("choices") or []:
            delta = ch.get("delta") or {}
            content = delta.get("content") or ""
            marker = _get_tool_garbage_marker(content)
            if marker:
                details.append(f"text_marker={marker!r}")
                break
    except Exception:
        details.append("text_marker=unknown")

    _traffic_dump_text(
        "IN_TOOL_GARBAGE",
        req_id,
        model_id,
        f"phase={phase} unparseable_markup {' '.join(details)}"
    )
    return True


def _split_content_and_tool_calls(chunk: dict) -> List[dict]:
    """Ensure we never emit a chunk that mixes delta.content and delta.tool_calls."""
    try:
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return [chunk]

        need_split = False
        for ch in choices:
            if not isinstance(ch, dict):
                continue
            delta = ch.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            tool_calls = delta.get("tool_calls")
            if isinstance(content, str) and content and isinstance(tool_calls, list) and tool_calls:
                need_split = True
                break

        if not need_split:
            return [chunk]

        import copy as _copy
        chunk_content = _copy.deepcopy(chunk)
        chunk_tools = _copy.deepcopy(chunk)

        for ch_c, ch_t in zip(chunk_content.get("choices", []), chunk_tools.get("choices", [])):
            if not isinstance(ch_c, dict) or not isinstance(ch_t, dict):
                continue
            d_c = ch_c.get("delta")
            d_t = ch_t.get("delta")
            if not isinstance(d_c, dict) or not isinstance(d_t, dict):
                continue

            # Content-only
            d_c.pop("tool_calls", None)
            if ch_c.get("finish_reason") == "tool_calls":
                ch_c["finish_reason"] = None

            # Tools-only
            d_t["content"] = ""
            if not (isinstance(d_t.get("tool_calls"), list) and d_t["tool_calls"]):
                return [chunk_content]

        return [chunk_content, chunk_tools]
    except Exception:
        return [chunk]


traffic_log = logging.getLogger("copilot_ai.traffic")
_traffic_mirror_budget: dict[str, int] = {}  # req_id -> remaining mirrored chunks to main log
_traffic_mirror_budget_max_size = 10000  # Limit the size of the budget dict to prevent memory leaks

# Load environment
load_env_files()

# Load configuration
config = load_config()
config.validate(require_api_key=False)

# Initialize logging
log = setup_logging(config.log_path)
dump_config(config)

# Attach dedicated traffic logger (VERY verbose). Kept separate from main logs.
if getattr(config, "debug_sse_traffic", False):
    try:
        traffic_log.setLevel(logging.DEBUG)
        traffic_log.propagate = False
        handler = RotatingFileHandler(
            filename=config.debug_sse_traffic_log_path,
            maxBytes=int(config.debug_sse_traffic_max_bytes),
            backupCount=int(config.debug_sse_traffic_backup_count),
            encoding="utf-8",
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s - %(message)s"))
        traffic_log.addHandler(handler)
        traffic_log.debug("Traffic logger enabled path=%s", config.debug_sse_traffic_log_path)
        # Diagnostic dump: handlers + levels
        try:
            traffic_log.debug(
                "Traffic logger state name=%s level=%s propagate=%s handlers=%d",
                traffic_log.name,
                traffic_log.level,
                traffic_log.propagate,
                len(traffic_log.handlers),
            )
            for idx, h in enumerate(list(traffic_log.handlers)):
                traffic_log.debug(
                    "Traffic handler[%d]=%s level=%s",
                    idx,
                    type(h).__name__,
                    getattr(h, "level", None),
                )
        except Exception:
            pass
    except Exception:
        log.exception("Failed to enable traffic logger")


def inject_system_prompt(body: Dict[str, Any]) -> None:
    """Prepend active system prompt as a system message if enabled.

    - Does nothing if body has no valid 'messages' list.
    - Avoids inserting duplicates (exact same content already present in a system message).
    """
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return

    def _has_system_prompt(content: str) -> bool:
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "system" and m.get("content") == content:
                return True
        return False

    if _active_prompt is None:
        return

    active_title, active_content = _active_prompt
    if not active_content:
        return

    if _has_system_prompt(active_content):
        return

    msgs.insert(0, {"role": "system", "content": active_content})
    log.info("ACTIVE_SYSTEM_PROMPT injected title=%r", active_title)


PROMPT_DB_PATH = Path(__file__).parent / "prompt.db"
_PROMPT_HEADER_RE = re.compile(r"^(?:#+\s*)?-{4,}\s*(.*?)\s*-{4,}(?:\s*#+)?$")
_USER_REQUEST_RE = re.compile(r"<userRequest>(.*?)</userRequest>", re.I | re.S)
_USER_REQUEST_TAG_RE = re.compile(r"</?userRequest>", re.I)
_ATTACHMENTS_RE = re.compile(r"<attachments>.*?</attachments>", re.I | re.S)
_CTX_RE = re.compile(r"<context>.*?</context>", re.I | re.S)
_EDITOR_CTX_RE = re.compile(r"<editorContext>.*?</editorContext>", re.I | re.S)
_REMINDER_RE = re.compile(r"<reminderInstructions>.*?</reminderInstructions>", re.I | re.S)


@dataclass
class _PromptCommandState:
    mode: str
    title: str | None = None


_prompt_command_states: dict[str, _PromptCommandState] = {}
_prompt_command_lock = asyncio.Lock()
_prompt_db_lock = asyncio.Lock()
_active_prompt: tuple[str, str] | None = None  # (title, content)
_LOCAL_COMMAND_PREFIXES = (":",)


def _parse_prompt_header(line: str) -> str | None:
    m = _PROMPT_HEADER_RE.match(line.strip())
    if not m:
        return None
    title = (m.group(1) or "").strip()
    if not title:
        return None
    return title


def _parse_prompt_db_text(text: str) -> list[tuple[str, list[str]]]:
    entries: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        header = _parse_prompt_header(line)
        if header is not None:
            if current_title is not None:
                entries.append((current_title, current_lines))
            current_title = header
            current_lines = []
            continue

        if current_title is None:
            if not line.strip():
                continue
            # Ignore stray lines before first header.
            continue

        if line.startswith("- "):
            current_lines.append(line[2:])
        elif line.startswith("-\t"):
            current_lines.append(line[2:])
        else:
            current_lines.append(line)

    if current_title is not None:
        entries.append((current_title, current_lines))

    return entries


def _serialize_prompt_db(entries: list[tuple[str, list[str]]]) -> str:
    lines: list[str] = []
    for title, content_lines in entries:
        lines.append(f"### ---- {title} ---- ###")
        for line in content_lines:
            line = "" if line is None else line
            lines.append(f"- {line}")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _normalize_prompt_content(text: str) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return []
    normalized: list[str] = []
    for line in lines:
        if line.startswith("- "):
            normalized.append(line[2:])
        elif line.startswith("-\t"):
            normalized.append(line[2:])
        else:
            normalized.append(line)
    return normalized


def _extract_user_request(text: str) -> str:
    if not text:
        return text
    matches = _USER_REQUEST_RE.findall(text)
    if matches:
        for match in reversed(matches):
            candidate = (match or "").strip()
            if not candidate:
                continue
            if candidate == "(.*?)":
                continue
            return candidate
        text = _USER_REQUEST_RE.sub("", text)
    cleaned = _ATTACHMENTS_RE.sub("", text)
    cleaned = _CTX_RE.sub("", cleaned)
    cleaned = _EDITOR_CTX_RE.sub("", cleaned)
    cleaned = _REMINDER_RE.sub("", cleaned)
    cleaned = _USER_REQUEST_TAG_RE.sub("", cleaned)
    return cleaned.strip()


async def _load_prompt_db_entries() -> list[tuple[str, list[str]]]:
    async with _prompt_db_lock:
        if not PROMPT_DB_PATH.exists():
            return []
        try:
            text = PROMPT_DB_PATH.read_text(encoding="utf-8")
        except Exception:
            log.exception("Failed to read prompt.db at %s", PROMPT_DB_PATH)
            return []
    return _parse_prompt_db_text(text)


async def _write_prompt_db_entries(entries: list[tuple[str, list[str]]]) -> bool:
    data = _serialize_prompt_db(entries)
    async with _prompt_db_lock:
        try:
            PROMPT_DB_PATH.write_text(data, encoding="utf-8")
            return True
        except Exception:
            log.exception("Failed to write prompt.db at %s", PROMPT_DB_PATH)
            return False


def _get_last_user_content(messages: List[Dict[str, Any]]) -> Optional[tuple[str, str]]:
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            cleaned = _extract_user_request(c)
            if cleaned != c:
                log.debug("LOCAL_COMMAND user_content cleaned len=%d raw_preview=%r", len(cleaned), cleaned[:200])
            return c, cleaned
        if isinstance(c, list):
            parts: list[str] = []
            for item in c:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                joined = "\n".join(parts)
                cleaned = _extract_user_request(joined)
                if cleaned != joined:
                    log.debug(
                        "LOCAL_COMMAND user_content cleaned len=%d raw_preview=%r",
                        len(cleaned),
                        cleaned[:200],
                    )
                return joined, cleaned
    return None


def _get_prompt_session_key(request: Request, authorization: Optional[str]) -> str:
    for header in ("x-session-id", "x-client-id", "x-user-id"):
        val = (request.headers.get(header) or "").strip()
        if val:
            return f"{header}:{val}"

    xff = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if xff:
        return f"xff:{xff}"

    if authorization:
        h = hashlib.sha256(authorization.strip().encode("utf-8")).hexdigest()[:16]
        return f"auth:{h}"

    if request.client:
        return f"ip:{request.client.host}"

    return "unknown"


def _format_prompt_titles(titles: list[str]) -> str:
    lines = ["Оглавления системных промптов:"]
    for i, t in enumerate(titles, start=1):
        lines.append(f"{i}. {t}")
    return "\n".join(lines)


def _get_help_text() -> str:
    return (
        "Доступные команды:\n"
        ":help - показывает это сообщение\n"
        ":prompt_list - выводит список всех оглавлений промпта\n"
        ":prompt_select - активирует системный промпт\n"
        ":prompt_add - добавляет в базу системный промпт\n"
        ":prompt_edit - обновляет содержимое системного промпта с выбранным заголовком\n"
        ":prompt_delete - удаляет из базы системный промпт\n"
    )


async def _local_sse_text_response(text: str, req_id: str) -> Response:
    async def gen() -> AsyncGenerator[bytes, None]:
        yield _sse_data(_first_choice_chunk(req_id=req_id, model_id=config.virtual_model_id))
        yield _sse_data(
            _create_chunk_dict(
                req_id=req_id,
                model_id=config.virtual_model_id,
                delta={"content": text},
                finish_reason=None,
            )
        )
        yield _sse_data(
            _create_chunk_dict(
                req_id=req_id,
                model_id=config.virtual_model_id,
                delta={},
                finish_reason="stop",
            )
        )
        yield _sse_done()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _handle_prompt_commands(
    request: Request,
    authorization: Optional[str],
    messages: List[Dict[str, Any]],
    req_id: str,
) -> Optional[Response]:
    global _active_prompt
    content_pair = _get_last_user_content(messages)
    if content_pair is None:
        return None

    raw_content, cleaned_content = content_pair
    text = cleaned_content.strip()
    command = None

    # Normalize and search for a command line (ignore attachments block if present).
    def _strip_known_blocks(src: str) -> str:
        if not src:
            return src
        src = _ATTACHMENTS_RE.sub("", src)
        src = _CTX_RE.sub("", src)
        src = _EDITOR_CTX_RE.sub("", src)
        src = _REMINDER_RE.sub("", src)
        src = _USER_REQUEST_TAG_RE.sub("", src)
        return src

    def _find_command_line(src: str) -> tuple[str | None, bool]:
        if not src:
            return None, False
        command_src = src.lstrip()
        command_src = command_src.lstrip("\ufeff\u200b\u200c\u200d")
        in_attachments = False
        first_nonempty: str | None = None
        found_line: str | None = None
        saw_prefix = False
        for ln in command_src.splitlines():
            stripped = ln.lstrip()
            if stripped == "<attachments>":
                in_attachments = True
                continue
            if stripped == "</attachments>":
                in_attachments = False
                continue
            if in_attachments:
                continue
            if not stripped:
                continue
            if first_nonempty is None:
                first_nonempty = stripped
            candidate = stripped.lstrip(">").lstrip()
            if candidate and candidate[0] in _LOCAL_COMMAND_PREFIXES:
                saw_prefix = True
                found_line = candidate
                break
        if found_line is None and first_nonempty is not None:
            candidate = first_nonempty.lstrip(">").lstrip()
            if candidate and candidate[0] in _LOCAL_COMMAND_PREFIXES:
                saw_prefix = True
                found_line = candidate
        return found_line, saw_prefix

    command_line: str | None = None
    saw_prefixed_line = False
    command_candidates = [cleaned_content]
    if raw_content != cleaned_content:
        command_candidates.append(_strip_known_blocks(raw_content))

    for candidate_src in command_candidates:
        command_line, saw_prefixed_line = _find_command_line(candidate_src)
        if command_line or saw_prefixed_line:
            break

    if command_line:
        head = command_line.split()[0]
        command = (head[1:] or "").strip().lower()

    session_key = _get_prompt_session_key(request, authorization)
    if command is not None:
        log.info("LOCAL_COMMAND received cmd=%r session=%s line=%r", command, session_key, command_line)
    elif saw_prefixed_line:
        log.info("LOCAL_COMMAND not matched session=%s raw=%r", session_key, raw_content[:200])

    async with _prompt_command_lock:
        pending = _prompt_command_states.get(session_key)

        known_commands = {
            "help",
            "prompt_list",
            "prompt_select",
            "prompt_add",
            "prompt_edit",
            "prompt_delete",
        }

        if command is not None and command not in known_commands and pending is None:
            return await _local_sse_text_response(_get_help_text(), req_id)

        if command in known_commands:
            _prompt_command_states.pop(session_key, None)

            if command == "help":
                return await _local_sse_text_response(_get_help_text(), req_id)

            if command == "prompt_list":
                entries = await _load_prompt_db_entries()
                titles = [t for t, _ in entries]
                if not titles:
                    return await _local_sse_text_response("База prompt.db пуста.", req_id)
                return await _local_sse_text_response(_format_prompt_titles(titles), req_id)

            if command == "prompt_select":
                entries = await _load_prompt_db_entries()
                titles = [t for t, _ in entries]
                if not titles:
                    return await _local_sse_text_response("База prompt.db пуста.", req_id)
                _prompt_command_states[session_key] = _PromptCommandState(mode="select_title")
                prompt = _format_prompt_titles(titles) + "\n\nВыберите номер оглавления и отправьте его сообщением."
                return await _local_sse_text_response(prompt, req_id)

            if command == "prompt_add":
                _prompt_command_states[session_key] = _PromptCommandState(mode="add_title")
                return await _local_sse_text_response(
                    "Введите оглавление для нового системного промпта.", req_id
                )

            if command == "prompt_edit":
                entries = await _load_prompt_db_entries()
                titles = [t for t, _ in entries]
                if not titles:
                    return await _local_sse_text_response("База prompt.db пуста.", req_id)
                _prompt_command_states[session_key] = _PromptCommandState(mode="edit_title")
                prompt = _format_prompt_titles(titles) + "\n\nВведите номер оглавления для редактирования."
                return await _local_sse_text_response(prompt, req_id)

            if command == "prompt_delete":
                entries = await _load_prompt_db_entries()
                titles = [t for t, _ in entries]
                if not titles:
                    return await _local_sse_text_response("База prompt.db пуста.", req_id)
                _prompt_command_states[session_key] = _PromptCommandState(mode="delete_title")
                prompt = _format_prompt_titles(titles) + "\n\nВведите номер оглавления для удаления."
                return await _local_sse_text_response(prompt, req_id)

        if pending is None:
            return None

        if pending.mode == "select_title":
            title = text.strip()
            if not title:
                return await _local_sse_text_response("Оглавление не может быть пустым. Повторите ввод.", req_id)
            entries = await _load_prompt_db_entries()
            if title.isdigit():
                idx = int(title)
                if 1 <= idx <= len(entries):
                    t, lines = entries[idx - 1]
                    content = "\n".join(lines)
                    _active_prompt = (t, content)
                    _prompt_command_states.pop(session_key, None)
                    log.info(
                        "ACTIVE_SYSTEM_PROMPT selected title=%r len=%d",
                        t,
                        len(content),
                    )
                    return await _local_sse_text_response(
                        f"Активирован системный промпт: {t}", req_id
                    )
                return await _local_sse_text_response(
                    f"Номер вне диапазона. Используйте 1..{len(entries)} или :prompt_list.", req_id
                )
            for t, lines in entries:
                if t == title:
                    content = "\n".join(lines)
                    _active_prompt = (t, content)
                    _prompt_command_states.pop(session_key, None)
                    log.info(
                        "ACTIVE_SYSTEM_PROMPT selected title=%r len=%d",
                        t,
                        len(content),
                    )
                    return await _local_sse_text_response(
                        f"Активирован системный промпт: {t}", req_id
                    )
            return await _local_sse_text_response(
                "Оглавление не найдено. Повторите ввод или используйте :prompt_list.", req_id
            )

        if pending.mode == "delete_title":
            title = text.strip()
            if not title:
                return await _local_sse_text_response("Оглавление не может быть пустым. Повторите ввод.", req_id)
            entries = await _load_prompt_db_entries()
            if title.isdigit():
                idx = int(title)
                if 1 <= idx <= len(entries):
                    title = entries[idx - 1][0]
                else:
                    return await _local_sse_text_response(
                        f"Номер вне диапазона. Используйте 1..{len(entries)} или :prompt_list.", req_id
                    )
            new_entries = [(t, lines) for t, lines in entries if t != title]
            if len(new_entries) == len(entries):
                return await _local_sse_text_response(
                    "Оглавление не найдено. Повторите ввод или используйте :prompt_list.", req_id
                )
            ok = await _write_prompt_db_entries(new_entries)
            if not ok:
                return await _local_sse_text_response("Не удалось записать prompt.db.", req_id)

            if _active_prompt is not None and _active_prompt[0] == title:
                _active_prompt = None
                _prompt_command_states.pop(session_key, None)
                return await _local_sse_text_response(
                    f"Промпт удалён и деактивирован: {title}", req_id
                )
            _prompt_command_states.pop(session_key, None)
            return await _local_sse_text_response(f"Промпт удалён: {title}", req_id)

        if pending.mode == "add_title":
            title = text.strip()
            if not title:
                return await _local_sse_text_response("Оглавление не может быть пустым. Повторите ввод.", req_id)
            entries = await _load_prompt_db_entries()
            if any(t == title for t, _ in entries):
                return await _local_sse_text_response(
                    "Оглавление уже существует. Используйте другое имя или :prompt_delete.", req_id
                )
            _prompt_command_states[session_key] = _PromptCommandState(mode="add_content", title=title)
            return await _local_sse_text_response(
                "Введите содержимое системного промпта (можно в несколько строк).", req_id
            )

        if pending.mode == "edit_title":
            title = text.strip()
            if not title:
                return await _local_sse_text_response("Оглавление не может быть пустым. Повторите ввод.", req_id)
            entries = await _load_prompt_db_entries()
            if title.isdigit():
                idx = int(title)
                if 1 <= idx <= len(entries):
                    title = entries[idx - 1][0]
                else:
                    return await _local_sse_text_response(
                        f"Номер вне диапазона. Используйте 1..{len(entries)} или :prompt_list.", req_id
                    )
            if not any(t == title for t, _ in entries):
                return await _local_sse_text_response(
                    "Оглавление не найдено. Повторите ввод или используйте :prompt_list.", req_id
                )
            _prompt_command_states[session_key] = _PromptCommandState(mode="edit_content", title=title)
            return await _local_sse_text_response(
                "Введите новое содержимое системного промпта (можно в несколько строк).", req_id
            )

        if pending.mode == "add_content":
            title = pending.title or ""
            if not title:
                _prompt_command_states.pop(session_key, None)
                return await _local_sse_text_response(
                    "Сбой состояния добавления. Повторите :prompt_add.", req_id
                )
            if not cleaned_content.strip():
                return await _local_sse_text_response(
                    "Содержимое не может быть пустым. Повторите ввод.", req_id
                )
            lines = _normalize_prompt_content(cleaned_content)
            entries = await _load_prompt_db_entries()
            entries.append((title, lines))
            ok = await _write_prompt_db_entries(entries)
            _prompt_command_states.pop(session_key, None)
            if not ok:
                return await _local_sse_text_response("Не удалось записать prompt.db.", req_id)
            return await _local_sse_text_response(
                f"Промпт сохранён: {title}\nДля активации используйте :prompt_select.", req_id
            )

        if pending.mode == "edit_content":
            title = pending.title or ""
            if not title:
                _prompt_command_states.pop(session_key, None)
                return await _local_sse_text_response(
                    "Сбой состояния редактирования. Повторите :prompt_edit.", req_id
                )
            if not cleaned_content.strip():
                return await _local_sse_text_response(
                    "Содержимое не может быть пустым. Повторите ввод.", req_id
                )
            entries = await _load_prompt_db_entries()
            updated = False
            new_lines = _normalize_prompt_content(cleaned_content)
            new_entries: list[tuple[str, list[str]]] = []
            for t, lines in entries:
                if t == title:
                    new_entries.append((t, new_lines))
                    updated = True
                else:
                    new_entries.append((t, lines))
            if not updated:
                _prompt_command_states.pop(session_key, None)
                return await _local_sse_text_response(
                    "Оглавление не найдено. Повторите :prompt_edit.", req_id
                )
            ok = await _write_prompt_db_entries(new_entries)
            _prompt_command_states.pop(session_key, None)
            if not ok:
                return await _local_sse_text_response("Не удалось записать prompt.db.", req_id)
            return await _local_sse_text_response(
                f"Промпт обновлён: {title}\nДля активации используйте :prompt_select.", req_id
            )

    return None


# Initialize components
model_cache = ModelCache()
model_selector = ModelSelector()
sse_validator = SSEValidator()
sse_streamer = SSEStreamer()
upstream_client = UpstreamClient(config)

# Track last selected model
_last_selected_lock = asyncio.Lock()


@dataclass
class _LastSelectedState:
    selected: Optional[ModelInfo] = None
    selected_at: Optional[float] = None


_last_selected_state = _LastSelectedState()


def _read_last_lines(path: Path, n: int, max_bytes: int = 256_000) -> list[str]:
    """
    Best-effort: read last N lines without loading the whole file.
    Caps reading to max_bytes to avoid large IO.
    """
    if n <= 0:
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # Read up to max_bytes from end
            start = max(0, size - max_bytes)
            f.seek(start, os.SEEK_SET)
            data = f.read(size - start)
        # Decode with replacement to avoid crashes on partial UTF-8 at boundary
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return lines[-n:]
    except Exception:
        return []


async def _tail_f_sse(
    path: Path,
    *,
    poll_interval_s: float = 0.4,
) -> AsyncIterator[bytes]:
    """
    SSE 'tail -F' for a log file.
    - follows new lines
    - tolerates RotatingFileHandler rotation (inode change / truncation)
    """

    def _stat_key(p: Path) -> tuple[int, int]:
        st = p.stat()
        # (inode, size) for rotation/truncation detection
        return (getattr(st, "st_ino", 0), st.st_size)

    # Wait until file exists
    while not path.exists():
        yield b": waiting for log file\n\n"
        await asyncio.sleep(min(1.0, poll_interval_s))

    try:
        ino, _ = _stat_key(path)
    except Exception:
        ino = 0

    try:
        f = path.open("r", encoding="utf-8", errors="replace")
    except Exception as e:
        # surface error to client as SSE comments
        yield f": failed to open log file: {type(e).__name__}\n\n".encode("utf-8")
        return

    # Start at EOF (true tail -f)
    try:
        f.seek(0, os.SEEK_END)
    except Exception:
        pass

    try:
        while True:
            line = f.readline()
            if line:
                # strip trailing newline; SSE event is delimited by blank line
                msg = line.rstrip("\r\n")
                yield f"data: {msg}\n\n".encode("utf-8")
                continue

            # No new data: check rotation/truncation occasionally
            await asyncio.sleep(poll_interval_s)

            # Detect rotation/truncation: inode change or file size < current position
            try:
                new_ino, new_sz = _stat_key(path)
                cur_pos = f.tell()
                rotated = (new_ino != ino and new_ino != 0)
                truncated = (new_sz < cur_pos)
                if rotated or truncated:
                    with contextlib.suppress(Exception):
                        f.close()
                    ino = new_ino
                    f = path.open("r", encoding="utf-8", errors="replace")
                    # after rotation: start from beginning of new file
                    # after truncation: start from beginning too
                    with contextlib.suppress(Exception):
                        f.seek(0, os.SEEK_SET)
                    yield b": log rotated/truncated\n\n"
            except Exception:
                # If stat fails transiently, ignore and continue polling
                pass
    finally:
        with contextlib.suppress(Exception):
            f.close()


async def log_filtered_models(*, timeout_s: Optional[float] = None) -> None:
    """Log filtered models at startup."""
    if not config.openrouter_api_key:
        log.warning("OPENROUTER_API_KEY not set, cannot fetch models at startup")
        return

    try:
        timeout = config.request_timeout_s if timeout_s is None else timeout_s
        proxy_url = upstream_client.get_proxy_url()
        async with httpx.AsyncClient(timeout=timeout, proxy=proxy_url) as client:
            models = await model_cache.get_models(client, config)
        candidates = model_selector.choose_candidates(models, config)
    except Exception as e:
        log.exception("Failed to fetch/filter models at startup: %s", e)
        return

    log.info("=== FILTERED MODELS LIST ===")
    log.info(
        "Total candidates=%d (min_ctx=%s max_price=%s)",
        len(candidates),
        config.min_context_length,
        config.max_price,
    )
    for i, m in enumerate(candidates, start=1):
        log.info("%d. id=%s name=%s ctx=%s price=%s", i, m.id, m.name, m.context_length, m.max_price)
    log.info("===========================")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Lifespan context manager for FastAPI application.

    Handles startup and shutdown events in a modern way.
    """
    # Startup: keep it lightweight (do not block readiness).
    startup_task: Optional[asyncio.Task[None]] = None
    with contextlib.suppress(Exception):
        if config.openrouter_api_key:
            startup_task = asyncio.create_task(
                log_filtered_models(timeout_s=min(5.0, config.request_timeout_s)),
                name="copilot_ai.log_filtered_models",
            )

    yield  # Application is running

    # Shutdown: cleanup if needed
    if startup_task is not None:
        startup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await startup_task


# FastAPI app with lifespan
app = FastAPI(
    title="copilot-ai-service",
    version="0.8.0",
    lifespan=lifespan
)

# Enable CORS for logs.html and other web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for web viewer
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/debug/traffic")
async def debug_traffic() -> Dict[str, Any]:
    """
    Write a test line into traffic logger and return its current handler state.
    Useful to verify that the running worker can write into DEBUG traffic log.
    """
    try:
        traffic_log.debug("DEBUG_TRAFFIC_PING ts=%s pid=%s", time.time(), os.getpid())
    except Exception:
        log.exception("debug_traffic: failed to write traffic_log")
    return {
        "debug_sse_traffic": bool(getattr(config, "debug_sse_traffic", False)),
        "traffic_logger_name": traffic_log.name,
        "traffic_logger_level": traffic_log.level,
        "traffic_logger_propagate": traffic_log.propagate,
        "traffic_handlers": [type(h).__name__ for h in traffic_log.handlers],
        "pid": os.getpid(),
        "traffic_path": getattr(config, "debug_sse_traffic_log_path", None),
    }


async def set_last_selected(m: ModelInfo) -> None:
    """Set the last selected model."""
    async with _last_selected_lock:
        _last_selected_state.selected = m
        _last_selected_state.selected_at = time.monotonic()
    log.info(
        "Selected model upstream_id=%s upstream_name=%s ctx=%s price=%s/%s",
        m.id,
        m.name,
        m.context_length,
        m.prompt_price,
        m.completion_price,
    )


async def get_last_selected() -> Optional[ModelInfo]:
    """Get the last selected model."""
    ttl_s = int(getattr(config, "last_selected_ttl_s", 0))
    async with _last_selected_lock:
        if _last_selected_state.selected is None:
            return None
        if ttl_s <= 0:
            return None
        if _last_selected_state.selected_at is None:
            return None
        age = time.monotonic() - _last_selected_state.selected_at
        if age <= ttl_s:
            return _last_selected_state.selected
        # Expired: clear cache
        _last_selected_state.selected = None
        _last_selected_state.selected_at = None
        return None


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


def _get_logs_html() -> str:
    """Return embedded HTML for logs viewer with ANSI color support."""
    template_path = Path(__file__).parent / "templates" / "logs_viewer.html"
    try:
        return template_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.error(f"Logs viewer template not found at {template_path}")
        return "<html><body><h1>Error: Logs viewer template not found</h1></body></html>"


@app.get("/logs")
async def logs() -> Response:
    """Serve embedded web UI for viewing logs."""
    return HTMLResponse(content=_get_logs_html())


@app.get("/logs/sse")
async def logs_stream(
    request: Request,
    lines: int = Query(default=200, ge=0, le=5000),
    follow: bool = Query(default=True),
    poll_interval_s: float = Query(default=0.4, ge=0.05, le=5.0),
    keepalive_s: float = Query(default=30.0, ge=0.0, le=300.0),
    x_logs_token: Optional[str] = Header(default=None),
) -> Response:
    """
    Stream service logs as SSE (tail -F) for web UI.

    Security:
      - If env LOGS_TOKEN is set: require header X-Logs-Token: <token>
      - If not set: endpoint is open (consider restricting at reverse proxy / firewall).
    """
    required = (os.getenv("LOGS_TOKEN") or "").strip()
    if required:
        if (x_logs_token or "").strip() != required:
            raise HTTPException(status_code=403, detail="Forbidden")

    log_path = Path(config.log_path)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"Log file not found: {log_path}")

    async def _stream() -> AsyncIterator[bytes]:
        # Send last N lines first (as individual SSE events)
        if lines > 0:
            for ln in _read_last_lines(log_path, lines):
                yield f"data: {ln}\n\n".encode("utf-8")

        if not follow:
            # End stream cleanly
            yield b": end\n\n"
            return

        # Follow file (tail -F)
        async for b in _tail_f_sse(
            log_path, poll_interval_s=poll_interval_s
        ):
            # Client disconnect handling
            if await request.is_disconnected():
                return
            yield b

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/logs/download")
async def logs_download(
    request: Request,
    x_logs_token: Optional[str] = Header(default=None),
) -> Response:
    """
    Save logs to a downloadable file.

    Security:
      - If env LOGS_TOKEN is set: require header X-Logs-Token: <token>
      - If not set: endpoint is open (consider restricting at reverse proxy / firewall).
    """
    required = (os.getenv("LOGS_TOKEN") or "").strip()
    if required:
        if (x_logs_token or "").strip() != required:
            raise HTTPException(status_code=403, detail="Forbidden")

    log_path = Path(config.log_path)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"Log file not found: {log_path}")

    try:
        body = await request.json()
        lines = body.get("lines", 200)
        if not isinstance(lines, int) or lines < 0 or lines > 10000:
            raise HTTPException(status_code=400, detail="Invalid lines parameter (0-10000)")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Read last N lines
    log_lines = _read_last_lines(log_path, lines, max_bytes=1_000_000)
    content = "\n".join(log_lines)

    return Response(
        content=content,
        media_type="text/plain",
        headers={
            "Content-Disposition": "attachment; filename=copilot-ai-logs.log",
        },
    )


@app.get("/v1/models")
async def v1_models() -> Dict[str, Any]:
    """List available models (returns virtual model)."""
    if not config.openrouter_api_key:
        raise HTTPException(
            status_code=500, detail="OPENROUTER_API_KEY environment variable required"
        )

    selected = await get_last_selected()
    if selected is None:
        log.info("/v1/models no last_selected; choosing best candidate now")
        proxy_url = upstream_client.get_proxy_url()
        async with httpx.AsyncClient(timeout=config.request_timeout_s, proxy=proxy_url) as client:
            models = await model_cache.get_models(client, config)
        candidates = model_selector.choose_candidates(models, config)
        log.info("/v1/models candidates count=%d", len(candidates))

        if not candidates:
            return {
                "object": "list",
                "data": [
                    {
                        "id": config.virtual_model_id,
                        "object": "model",
                        "owned_by": "copilot-ai",
                        "name": config.virtual_model_name,
                        "unavailable_reason": "No candidates match tools + >=MIN_CTX (and price/bans).",
                    }
                ],
            }
        selected = candidates[0]
        await set_last_selected(selected)

    return {
        "object": "list",
        "data": [selected.to_virtual_model_dict(config.virtual_model_id, config.virtual_model_name)],
    }


async def handle_explicit_model_request(
    client: httpx.AsyncClient,
    body: Dict[str, Any],
    model_req: str,
    req_id: str,
) -> Response:
    """Handle request with explicitly specified upstream model (always buffered stream)."""
    log.info("Explicitly requested upstream model: %s", model_req)

    # Force upstream streaming, buffer until [DONE]
    upstream_body = dict(body)
    upstream_body["model"] = model_req
    upstream_body["stream"] = True

    resp = await upstream_client.chat_completion(client, upstream_body, model_req)

    # Handle error responses
    if resp.status_code != 200:
        snippet = await upstream_client.read_error_snippet(resp)
        await resp.aclose()
        with contextlib.suppress(Exception):
            await client.aclose()
        return StreamingResponse(
            sse_streamer.error_response(
                snippet or f"Upstream error {resp.status_code}", config.virtual_model_id
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Always use buffered streaming logic
    aiter = resp.aiter_lines()
    ok, first_event, _, reason = await sse_validator.peek_first_sse_choices(aiter)
    if not ok or first_event is None:
        log.warning(
            "SSE invalid (no choices) req_id=%s model=%s reason=%s",
            req_id,
            model_req,
            reason,
        )
        await resp.aclose()
        with contextlib.suppress(Exception):
            await client.aclose()
        return StreamingResponse(
            sse_streamer.error_response(
                f"Upstream stream invalid (no choices). reason={reason} model={model_req} req_id={req_id}",
                config.virtual_model_id,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    first_bytes = sse_event_to_bytes(first_event)

    # Grab reverse tool id map (short->orig) generated by upstream client
    tool_id_map_short_to_orig = None
    try:
        m = upstream_body.get("_tool_id_map_short_to_orig")
        if isinstance(m, dict) and m:
            tool_id_map_short_to_orig = m
    except Exception:
        tool_id_map_short_to_orig = None

    # Buffer until [DONE]
    buf_bytes = len(first_bytes)
    last_activity = time.monotonic()
    failed_reason: str | None = None
    done = False
    normalized_chunks: List[dict] = []

    # Parse the first event
    try:
        first_txt = first_bytes.decode("utf-8", errors="replace")
        for line in first_txt.splitlines():
            if not line.startswith("data:"):
                continue
            payload_txt = line[5:].strip()
            if not payload_txt or payload_txt == "[DONE]":
                continue
            obj = json.loads(payload_txt)
            norm = normalize_upstream_chunk_to_openai(
                obj,
                out_id=req_id,
                out_model=config.virtual_model_id,
                tool_id_map_short_to_orig=tool_id_map_short_to_orig,
            )
            if norm:
                if _check_and_log_tool_garbage(norm, req_id, model_req, "explicit-first"):
                    failed_reason = "tool-garbage"
                    break
                normalized_chunks.extend(_split_content_and_tool_calls(norm))
    except Exception:
        pass

    if failed_reason:
        await resp.aclose()
        with contextlib.suppress(Exception):
            await client.aclose()
        return StreamingResponse(
            sse_streamer.error_response(
                f"Tool garbage detected. model={model_req} req_id={req_id}",
                config.virtual_model_id,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Buffer remaining events until [DONE]
    while True:
        now = time.monotonic()
        if now - last_activity > max(1.0, float(config.stream_idle_timeout_s)):
            failed_reason = "buffer-idle-timeout"
            break
        if buf_bytes > int(config.max_buffered_sse_bytes):
            failed_reason = "buffer-too-large"
            break

        try:
            event_lines = await asyncio.wait_for(
                read_next_sse_event(aiter), timeout=1.0
            )
        except asyncio.TimeoutError:
            continue
        except Exception:
            failed_reason = "buffer-read-error"
            break

        if event_lines is None:
            failed_reason = "early-eof-before-done"
            break

        if not event_lines:
            last_activity = time.monotonic()
            continue

        last_activity = time.monotonic()
        data = sse_event_data_text(event_lines)

        if data and data.strip() != "[DONE]":
            try:
                obj = json.loads(data)
                norm = normalize_upstream_chunk_to_openai(
                    obj,
                    out_id=req_id,
                    out_model=config.virtual_model_id,
                    tool_id_map_short_to_orig=tool_id_map_short_to_orig,
                )
                if norm:
                    if _check_and_log_tool_garbage(norm, req_id, model_req, "explicit-body"):
                        failed_reason = "tool-garbage"
                        break
                    normalized_chunks.extend(_split_content_and_tool_calls(norm))
            except Exception:
                pass

        try:
            buf_bytes += sum(len(x) for x in event_lines)
        except Exception:
            buf_bytes += 0

        if data and data.strip() == "[DONE]":
            done = True
            break

    await resp.aclose()

    if not done or not normalized_chunks:
        with contextlib.suppress(Exception):
            await client.aclose()
        return StreamingResponse(
            sse_streamer.error_response(
                f"Stream failed. reason={failed_reason} model={model_req} req_id={req_id}",
                config.virtual_model_id,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Success: stream buffered content
    async def gen() -> AsyncGenerator[bytes, None]:
        first_obj = _first_choice_chunk(req_id=req_id, model_id=config.virtual_model_id)
        yield _sse_data(first_obj)
        for ch in normalized_chunks:
            yield _sse_data(ch)
        yield _sse_done()

    with contextlib.suppress(Exception):
        await client.aclose()

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _create_chunk_dict(
    req_id: str,
    model_id: str,
    delta: dict | None = None,
    finish_reason: str | None = None
) -> dict:
    """Create a standard OpenAI chat completion chunk dictionary."""
    return {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
    }


def _first_choice_chunk(req_id: str, model_id: str) -> dict:
    """Copilot-compatible first chunk with role + empty content."""
    return _create_chunk_dict(req_id, model_id, {"role": "assistant", "content": ""})


def _sse_data(obj: dict) -> bytes:
    """Encode dict as SSE data event (compact JSON, no space after data:)."""
    return f"data:{json.dumps(obj, ensure_ascii=False, separators=(',', ':'))}\n\n".encode("utf-8")


def _sse_done() -> bytes:
    """SSE [DONE] event."""
    return b"data: [DONE]\n\n"


def _normalized_has_payload(chunks: List[dict]) -> bool:
    """
    Return True if normalized chunks contain at least one:
      - non-empty delta.content after .strip()
      - non-empty delta.tool_calls list

    This detects "empty stop" completions which break Copilot ("Sorry, no response was returned.")
    and should trigger model failover.
    """
    try:
        for chunk in chunks or []:
            choices = chunk.get("choices") or []
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue

                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    return True

                content = delta.get("content")
                if isinstance(content, str) and content.strip():
                    return True
    except Exception:
        # On error: be permissive and assume "has payload" to avoid accidental failovers.
        return True
    return False


def _is_chunk_meaningful(chunk: dict) -> bool:
    """
    Check if a chunk contains meaningful content that should be emitted.

    Filters out empty/redundant chunks with:
    - Empty delta (role only, no content/tool_calls)
    - Empty content and no tool_calls
    - No finish_reason and no actual data

    This prevents duplicate "role=assistant, content=''" chunks.
    """
    try:
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return False

        for choice in choices:
            if not isinstance(choice, dict):
                continue

            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue

            finish_reason = choice.get("finish_reason")

            # Has finish_reason? Always meaningful (final chunk)
            if finish_reason:
                return True

            # Has tool_calls? Meaningful
            if delta.get("tool_calls"):
                return True

            # Has non-empty content? Meaningful
            content = delta.get("content")
            if isinstance(content, str) and content:
                return True

            # Only role field with empty/no content? Skip (redundant)
            if "role" in delta and not content and not delta.get("tool_calls"):
                continue

            # Empty delta? Skip
            if not delta:
                continue

        return False
    except Exception:
        # On error, emit the chunk to be safe
        return True


def _traffic_dump_bytes(tag: str, req_id: str, model_id: str, b: bytes) -> None:
    """
    Dump outbound/inbound traffic to dedicated logger.
    tag examples: OUT_FIRST, OUT_KEEPALIVE, OUT_CHUNK, OUT_DONE, IN_EVENT, IN_DATA, IN_RAW
    """
    if not getattr(config, "debug_sse_traffic", False):
        return
    try:
        # Mirror some traffic into main log (so you see it even if traffic file doesn't append)
        if getattr(config, "debug_sse_traffic_mirror_main", True):
            # budget per request to avoid blowing up copilot-ai.log instantly
            if req_id not in _traffic_mirror_budget:
                # Prevent memory leak: if dict is too large, clear old entries
                if len(_traffic_mirror_budget) >= _traffic_mirror_budget_max_size:
                    # Remove roughly half of the oldest entries
                    items_to_remove = list(_traffic_mirror_budget.keys())[:_traffic_mirror_budget_max_size // 2]
                    for k in items_to_remove:
                        _traffic_mirror_budget.pop(k, None)

                _traffic_mirror_budget[req_id] = int(
                    getattr(config, "debug_sse_traffic_mirror_main_max_chunks", 5)
                )
            if _traffic_mirror_budget[req_id] > 0:
                _traffic_mirror_budget[req_id] -= 1
                # show first ~2k chars in main log
                try:
                    shown = b[:2048].decode("utf-8", errors="replace")
                except Exception:
                    shown = repr(b[:256])
                log.debug(
                    "[TRAFFIC-MIRROR] %s req_id=%s model=%s bytes=%d preview=%r",
                    tag,
                    req_id,
                    model_id,
                    len(b),
                    shown,
                )

        trunc = int(getattr(config, "debug_sse_traffic_truncate_bytes", 0))
        if trunc and len(b) > trunc:
            shown = b[:trunc]
            suffix = f"\n...[truncated {len(b) - trunc} bytes]..."
        else:
            shown = b
            suffix = ""
        txt = shown.decode("utf-8", errors="replace")

        # If no handlers, fall back to main log (rare, but happens with multi-worker / reload scenarios)
        if not traffic_log.handlers:
            log.debug(
                "[TRAFFIC-FALLBACK] %s req_id=%s model=%s bytes=%d\n%s%s",
                tag,
                req_id,
                model_id,
                len(b),
                txt,
                suffix,
            )
            return

        traffic_log.debug(
            "%s req_id=%s model=%s bytes=%d\n%s%s",
            tag,
            req_id,
            model_id,
            len(b),
            txt,
            suffix,
        )
    except Exception:
        # Never break request because of logging.
        try:
            log.exception("traffic dump failed tag=%s req_id=%s model=%s", tag, req_id, model_id)
        except Exception:
            pass


def _traffic_dump_text(tag: str, req_id: str, model_id: str, txt: str) -> None:
    if not getattr(config, "debug_sse_traffic", False):
        return
    try:
        trunc = int(getattr(config, "debug_sse_traffic_truncate_bytes", 0))
        if trunc and len(txt.encode("utf-8", errors="replace")) > trunc:
            # approximate by chars; good enough for debug
            traffic_log.debug(
                "%s req_id=%s model=%s\n%s\n...[truncated]...",
                tag,
                req_id,
                model_id,
                txt[: max(1, trunc // 4)],
            )
        else:
            traffic_log.debug("%s req_id=%s model=%s\n%s", tag, req_id, model_id, txt)
    except Exception:
        pass


async def handle_auto_route_request_buffered_stream(
    client: httpx.AsyncClient,
    body: Dict[str, Any],
    req_id: str,
) -> Response:
    """Buffered streaming: upstream stream=True, client gets only keepalives until [DONE]."""
    cached = await get_last_selected()
    if cached is not None:
        use_cached_only = True
        log.info(
            "Auto-route using cached model upstream_id=%s ttl_s=%s",
            cached.id,
            getattr(config, "last_selected_ttl_s", None),
        )
    else:
        use_cached_only = False

    # Force upstream streaming, but do not forward content until we have a full [DONE]-terminated stream.
    base_body = dict(body)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "X-Copilot-AI-Virtual-Model": config.virtual_model_id,
        "X-Copilot-AI-Mode": "buffered-until-done",
    }

    async def gen() -> AsyncGenerator[bytes, None]:
        nonlocal use_cached_only
        total = 1 if use_cached_only else 0
        # IMPORTANT: send a minimal "data:" chunk immediately (role=assistant).
        # Copilot can error on "comments-only" streams.
        first_obj = _first_choice_chunk(req_id=req_id, model_id=config.virtual_model_id)
        first_b = _sse_data(first_obj)
        # explicit ping before first yield
        _traffic_dump_text(
            "GEN_START",
            req_id,
            config.virtual_model_id,
            f"pid={os.getpid()} total_candidates={total}",
        )
        _traffic_dump_bytes("OUT_FIRST", req_id, config.virtual_model_id, first_b)
        yield first_b

        keepalive_every = float(config.buffer_stream_keepalive_s)
        keepalive_every = min(20.0, max(10.0, keepalive_every))
        last_keepalive = time.monotonic()

        async def maybe_keepalive() -> bytes | None:
            nonlocal last_keepalive
            now = time.monotonic()
            if now - last_keepalive >= keepalive_every:
                last_keepalive = now
                b = b": keepalive\n\n"
                _traffic_dump_bytes("OUT_KEEPALIVE", req_id, config.virtual_model_id, b)
                return b
            return None

        try:
            round_no = 0
            attempt_global = 0
            start_idx = 0  # round-robin start so we don't always hit the same 429-first model

            while True:
                round_no += 1

                if use_cached_only:
                    assert cached is not None
                    current_candidates = [cached]
                    total = 1
                    log.info(
                        "Auto-route candidates=%d (cached) [buffered_stream_until_done]",
                        total,
                    )
                else:
                    # Refresh candidates each round so models can recover quickly from 429 bursts.
                    models = await model_cache.get_models(client, config)
                    current_candidates = model_selector.choose_candidates(models, config)
                    total = len(current_candidates)
                    log.info(
                        "Auto-route candidates=%d (after filters tools+>=MIN_CTX+price/ban) [buffered_stream_until_done]",
                        total,
                    )

                if not current_candidates:
                    # No candidates right now; keep the connection alive and retry.
                    ka = await maybe_keepalive()
                    if ka:
                        yield ka
                    await asyncio.sleep(0.25)
                    continue

                rotated = current_candidates[start_idx:] + current_candidates[:start_idx]
                start_idx = (start_idx + 1) % total if total else 0

                for i, mdl in enumerate(rotated, start=1):
                    attempt_global += 1
                    log.info(
                        "Attempt %d (round %d %d/%d) -> %s [buffered]",
                        attempt_global,
                        round_no,
                        i,
                        total,
                        mdl.id,
                    )
                    _traffic_dump_text(
                        "ATTEMPT_LOOP",
                        req_id,
                        mdl.id,
                        f"attempt_global={attempt_global} round={round_no} attempt={i}/{total}",
                    )

                    ka = await maybe_keepalive()
                    if ka:
                        yield ka

                    resp: httpx.Response | None = None
                    try:
                        _traffic_dump_text(
                            "ATTEMPT_START",
                            req_id,
                            mdl.id,
                            f"attempt={i}/{total} upstream_model={mdl.id} upstream_stream=True",
                        )
                        # NOTE: upstream client may inject _tool_id_map_short_to_orig into body.
                        upstream_body = dict(base_body)
                        upstream_body["model"] = mdl.id
                        upstream_body["stream"] = True
                        resp = await upstream_client.chat_completion(client, upstream_body, mdl.id)
                        if resp.status_code != 200:
                            try:
                                snippet = (await resp.aread()).decode("utf-8", errors="replace")[:512]
                            except Exception:
                                snippet = ""
                            _traffic_dump_text(
                                "UPSTREAM_NON200",
                                req_id,
                                mdl.id,
                                f"status={resp.status_code} body_snip={snippet!r}",
                            )
                            log.warning(
                                "Attempt %d/%d upstream-status req_id=%s model=%s status=%s body=%r; trying next",
                                i,
                                total,
                                req_id,
                                mdl.id,
                                resp.status_code,
                                snippet,
                            )
                            _traffic_dump_text(
                                "ATTEMPT_CONTINUE", req_id, mdl.id, "reason=upstream-non200"
                            )
                            continue

                        aiter = resp.aiter_lines()
                        ok, first_event, _, reason = await sse_validator.peek_first_sse_choices(aiter)
                        if not ok or first_event is None:
                            _traffic_dump_text(
                                "UPSTREAM_PEEK_FAIL", req_id, mdl.id, f"ok={ok} reason={reason!r}"
                            )
                            log.warning(
                                "Attempt %d/%d stream-invalid req_id=%s model=%s reason=%s; trying next",
                                i,
                                total,
                                req_id,
                                mdl.id,
                                reason,
                            )
                            _traffic_dump_text(
                                "ATTEMPT_CONTINUE", req_id, mdl.id, f"reason=peek_fail:{reason}"
                            )
                            continue
                        first_bytes = sse_event_to_bytes(first_event)
                        sse_validator.reset_tool_repair()  # per-attempt state

                        # Grab reverse tool id map (short->orig) generated by upstream client.
                        tool_id_map_short_to_orig = None
                        try:
                            m = upstream_body.get("_tool_id_map_short_to_orig")
                            if isinstance(m, dict) and m:
                                tool_id_map_short_to_orig = m
                                preview = list(m.items())[:10]
                                _traffic_dump_text(
                                    "TOOL_ID_MAP_SHORT_TO_ORIG",
                                    req_id,
                                    mdl.id,
                                    f"entries={len(m)} preview={preview!r}",
                                )
                        except Exception:
                            tool_id_map_short_to_orig = None

                        # Buffer until [DONE], but DO NOT forward upstream bytes to Copilot.
                        # Instead, normalize to strict OpenAI chat.completion.chunk on commit.
                        #
                        # Why: some OpenRouter providers emit chunks Copilot rejects silently,
                        # leading to "Sorry, no response was returned."
                        buf_bytes = len(first_bytes)
                        last_activity = time.monotonic()
                        failed_reason: str | None = None
                        done = False
                        normalized_chunks: List[dict] = []

                        _traffic_dump_bytes("IN_FIRST_EVENT_RAW", req_id, mdl.id, first_bytes)

                        # parse the first event (already validated to contain choices)
                        try:
                            first_txt = first_bytes.decode("utf-8", errors="replace")
                            # first_bytes may contain multiple lines "data: ...\n\n"
                            # Extract JSON from each data line.
                            for line in first_txt.splitlines():
                                if not line.startswith("data:"):
                                    continue
                                payload_txt = line[5:].strip()
                                if not payload_txt or payload_txt == "[DONE]":
                                    continue
                                _traffic_dump_text("IN_FIRST_DATA_LINE", req_id, mdl.id, payload_txt)
                                obj = json.loads(payload_txt)
                                if not sse_validator.repair_tool_markup_inplace(obj):
                                    continue
                                norm = normalize_upstream_chunk_to_openai(
                                    obj,
                                    out_id=req_id,
                                    out_model=config.virtual_model_id,
                                    tool_id_map_short_to_orig=tool_id_map_short_to_orig,
                                )
                                if norm:
                                    # DONE roulette mode: native tool_calls are OK (Copilot needs them!).
                                    # normalize_upstream_chunk_to_openai already parses <tool_call> tags from
                                    # content and converts them to native tool_calls. We only reject if
                                    # garbage remains AFTER normalization (unparseable or incomplete markup).
                                    if _check_and_log_tool_garbage(norm, req_id, mdl.id, "first"):
                                        failed_reason = "tool-garbage"
                                        break
                                    normalized_chunks.extend(_split_content_and_tool_calls(norm))
                        except Exception as e:
                            _traffic_dump_text(
                                "IN_FIRST_PARSE_ERR", req_id, mdl.id, f"{type(e).__name__}: {e}"
                            )
                            # Ignore parse errors here; we still rely on later events.
                            pass

                        if failed_reason:
                            log.warning(
                                "Attempt %d (round %d %d/%d) tool-garbage req_id=%s model=%s; trying next",
                                attempt_global,
                                round_no,
                                i,
                                total,
                                req_id,
                                mdl.id,
                            )
                            _traffic_dump_text(
                                "ATTEMPT_CONTINUE", req_id, mdl.id, f"reason={failed_reason}"
                            )
                            continue

                        while True:
                            ka = await maybe_keepalive()
                            if ka:
                                yield ka

                            now = time.monotonic()
                            if now - last_activity > max(1.0, float(config.stream_idle_timeout_s)):
                                failed_reason = "buffer-idle-timeout"
                                break
                            if buf_bytes > int(config.max_buffered_sse_bytes):
                                failed_reason = "buffer-too-large"
                                break

                            try:
                                event_lines = await asyncio.wait_for(
                                    read_next_sse_event(aiter), timeout=1.0
                                )
                            except asyncio.TimeoutError:
                                continue
                            except Exception:
                                failed_reason = "buffer-read-error"
                                break

                            if event_lines is None:
                                failed_reason = "early-eof-before-done"
                                break

                            if not event_lines:
                                last_activity = time.monotonic()
                                continue

                            last_activity = time.monotonic()
                            data = sse_event_data_text(event_lines)
                            # dump raw event (joined)
                            try:
                                raw = "\n".join(event_lines).encode("utf-8", errors="replace")
                                _traffic_dump_bytes("IN_EVENT_RAW", req_id, mdl.id, raw)
                            except Exception:
                                pass

                            if data and data.strip() != "[DONE]":
                                # Parse JSON and extract normalized content fragments.
                                try:
                                    _traffic_dump_text("IN_DATA", req_id, mdl.id, data)
                                    obj = json.loads(data)
                                    if not sse_validator.repair_tool_markup_inplace(obj):
                                        continue
                                    norm = normalize_upstream_chunk_to_openai(
                                        obj,
                                        out_id=req_id,
                                        out_model=config.virtual_model_id,
                                        tool_id_map_short_to_orig=tool_id_map_short_to_orig,
                                    )
                                    if norm:
                                        # DONE roulette mode: native tool_calls are OK (Copilot needs them!).
                                        # normalize_upstream_chunk_to_openai already parses <tool_call> tags from
                                        # content and converts them to native tool_calls. We only reject if
                                        # garbage remains AFTER normalization (unparseable or incomplete markup).
                                        if _check_and_log_tool_garbage(norm, req_id, mdl.id, "body"):
                                            failed_reason = "tool-garbage"
                                            break
                                        normalized_chunks.extend(_split_content_and_tool_calls(norm))
                                except Exception as exc:
                                    _traffic_dump_text(
                                        "IN_PARSE_ERR",
                                        req_id,
                                        mdl.id,
                                        f"{type(exc).__name__}: {exc} data={data!r}",
                                    )
                                    pass

                            # count bytes (raw size) for safety limits
                            try:
                                buf_bytes += sum(len(x) for x in event_lines)
                            except Exception:
                                buf_bytes += 0

                            if data and data.strip() == "[DONE]":
                                _traffic_dump_text("IN_DONE", req_id, mdl.id, "received [DONE]")
                                done = True
                                break

                        if not done:
                            _traffic_dump_text(
                                "ATTEMPT_FAIL",
                                req_id,
                                mdl.id,
                                f"reason={failed_reason} buf_bytes={buf_bytes}",
                            )
                            log.warning(
                                "Attempt %d/%d buffered-stream failed req_id=%s model=%s reason=%s; trying next",
                                i,
                                total,
                                req_id,
                                mdl.id,
                                failed_reason,
                            )
                            _traffic_dump_text(
                                "ATTEMPT_CONTINUE",
                                req_id,
                                mdl.id,
                                f"reason=not_done:{failed_reason}",
                            )
                            continue

                        if not normalized_chunks:
                            failed_reason = "no-normalized-chunks"
                            log.warning(
                                "Attempt %d/%d buffered-stream no-normalized-chunks req_id=%s model=%s; trying next",
                                i,
                                total,
                                req_id,
                                mdl.id,
                            )
                            continue

                        # If normalization produced only empty chunks (no content/tool_calls),
                        # treat as model failure and failover to the next candidate.
                        if not _normalized_has_payload(normalized_chunks):
                            failed_reason = "empty-completion"
                            log.warning(
                                "Attempt %d/%d buffered-stream empty-completion req_id=%s model=%s; trying next",
                                i,
                                total,
                                req_id,
                                mdl.id,
                            )
                            _traffic_dump_text(
                                "ATTEMPT_CONTINUE",
                                req_id,
                                mdl.id,
                                "reason=empty-completion",
                            )
                            continue

                        # Success: remember selected model and forward buffered content.
                        with contextlib.suppress(Exception):
                            await set_last_selected(mdl)

                        log.info(
                            "Buffered stream committed req_id=%s model=%s buffered_events=%d bytes=%d",
                            req_id,
                            mdl.id,
                            # events count isn't tracked now; keep old key but approximate as parts count
                            len(normalized_chunks),
                            buf_bytes,
                        )
                        _traffic_dump_text(
                            "COMMIT_SUMMARY",
                            req_id,
                            mdl.id,
                            f"buf_bytes={buf_bytes} chunks={len(normalized_chunks)}",
                        )

                        # Emit normalized chunks + DONE (strict SSE)
                        # Filter out empty/redundant chunks to avoid duplicates
                        for ch in normalized_chunks:
                            if _is_chunk_meaningful(ch):
                                out_b = _sse_data(ch)
                                _traffic_dump_bytes("OUT_CHUNK", req_id, config.virtual_model_id, out_b)
                                yield out_b
                        done_b = _sse_done()
                        _traffic_dump_bytes("OUT_DONE", req_id, config.virtual_model_id, done_b)
                        yield done_b
                        _traffic_dump_text("GEN_RETURN", req_id, mdl.id, "success=1")

                        return
                    except Exception:
                        log.exception(
                            "Attempt %d/%d buffered-stream exception req_id=%s model=%s",
                            i,
                            total,
                            req_id,
                            mdl.id,
                        )
                        continue
                    finally:
                        if resp is not None:
                            with contextlib.suppress(Exception):
                                await resp.aclose()

                # All candidates in this round failed; retry forever (roulette).
                if use_cached_only:
                    # Cached model failed; fall back to full candidate list.
                    assert cached is not None
                    log.info(
                        "Cached model failed; falling back to full candidate list req_id=%s model=%s",
                        req_id,
                        cached.id,
                    )
                    use_cached_only = False
                    continue
                await asyncio.sleep(0.05)
                continue
        except Exception:
            # Never let the generator crash the connection: emit an SSE error + [DONE].
            log.exception("Buffered stream generator crashed req_id=%s", req_id)
            async for b in sse_streamer.error_response(
                "Internal server error (buffered stream).",
                config.virtual_model_id,
            ):
                _traffic_dump_bytes("OUT_ERROR", req_id, config.virtual_model_id, b)
                yield b
            _traffic_dump_text("GEN_RETURN", req_id, config.virtual_model_id, "success=0 (crash)")
        finally:
            with contextlib.suppress(Exception):
                await client.aclose()

    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


@app.post("/v1/chat/completions")
async def v1_chat_completions(
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> Response:
    """Handle chat completion requests."""
    if not config.openrouter_api_key:
        raise HTTPException(
            status_code=500, detail="OPENROUTER_API_KEY environment variable required"
        )

    # Basic request size guard (prevents trivial DoS via huge JSON bodies).
    cl = request.headers.get("content-length")
    if cl:
        try:
            n = int(cl)
            if n < 0:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid Content-Length: must be non-negative",
                )
            if n > config.max_request_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request too large: {n} bytes (max {config.max_request_bytes})",
                )
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Content-Length header: {cl!r}",
            )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON body: expected object")

    # Validate messages structure
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(
            status_code=400,
            detail="Invalid request: 'messages' field must be an array"
        )
    if not messages:
        raise HTTPException(
            status_code=400,
            detail="Invalid request: 'messages' array cannot be empty"
        )

    client_ip = request.client.host if request.client else "unknown"
    req_id = (
        (request.headers.get("x-request-id") or "").strip()
        or (request.headers.get("x-correlation-id") or "").strip()
        or (request.headers.get("x-trace-id") or "").strip()
        or uuid.uuid4().hex
    )

    cmd_resp = await _handle_prompt_commands(request, authorization, messages, req_id)
    if cmd_resp is not None:
        return cmd_resp

    # Active system prompt injection (from prompt.db selection).
    inject_system_prompt(body)

    raw_model_req = body.get("model") or ""
    model_req = str(raw_model_req).strip()

    log.info(
        "Incoming chat req_id=%s from=%s request_model=%r",
        req_id,
        client_ip,
        raw_model_req,
    )

    # Normalize virtual model requests
    if model_req.lower() in {
        config.virtual_model_id.lower(),
        config.virtual_model_name.lower(),
        "auto",
    }:
        log.info("Model treated as virtual/auto: %r", model_req)
        model_req = ""

    # Always use streaming mode - no read timeout to avoid killing long pauses
    connect_timeout = min(30.0, float(config.request_timeout_s))

    # Get proxy configuration from upstream client
    proxy_url = upstream_client.get_proxy_url()

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=connect_timeout, write=connect_timeout, pool=connect_timeout, read=None),
        proxy=proxy_url
    )

    try:
        if model_req:
            return await handle_explicit_model_request(client, body, model_req, req_id)
        else:
            return await handle_auto_route_request_buffered_stream(client, body, req_id)
    except Exception:
        with contextlib.suppress(Exception):
            await client.aclose()
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=config.port, reload=False)
