import json

import httpx
import pytest

import copilot_ai_service as service


def _write_prompt_db(path, entries):
    lines = []
    for title, content_lines in entries:
        lines.append(f"---- {title} ----")
        for line in content_lines:
            lines.append(f"- {line}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_sse_content(raw_text: str) -> str:
    parts = []
    for event in raw_text.split("\n\n"):
        for line in event.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str):
                    parts.append(content)
    return "".join(parts)


def _session_headers(session_id="test-session"):
    return {"x-session-id": session_id}


@pytest.fixture(autouse=True)
def reset_prompt_state():
    service._prompt_command_states.clear()
    service._active_prompt = None
    yield
    service._prompt_command_states.clear()
    service._active_prompt = None


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=service.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


async def _post_chat(client, content, headers=None):
    body = {"model": "copilot-ai", "messages": [{"role": "user", "content": content}]}
    response = await client.post("/v1/chat/completions", json=body, headers=headers)
    raw = await response.aread()
    return response, raw.decode("utf-8", errors="replace")


@pytest.mark.asyncio
async def test_help_command_basic(client):
    response, text = await _post_chat(client, ":help")
    assert response.status_code == 200
    content = _extract_sse_content(text)
    assert "Доступные команды:" in content
    assert ":prompt_select" in content


@pytest.mark.asyncio
async def test_help_command_blockquote(client):
    response, text = await _post_chat(client, ">:help")
    assert response.status_code == 200
    content = _extract_sse_content(text)
    assert "Доступные команды:" in content


@pytest.mark.asyncio
async def test_help_command_with_attachments_block(client):
    content = "<attachments>\nfile.txt\n</attachments>\n:help"
    response, text = await _post_chat(client, content)
    assert response.status_code == 200
    assert "Доступные команды:" in _extract_sse_content(text)


@pytest.mark.asyncio
async def test_unknown_command_returns_help(client):
    response, text = await _post_chat(client, ":promp")
    assert response.status_code == 200
    content = _extract_sse_content(text)
    assert "Доступные команды:" in content


@pytest.mark.asyncio
async def test_command_in_text_parts(client):
    content = [{"type": "text", "text": ":help"}]
    response, text = await _post_chat(client, content)
    assert response.status_code == 200
    assert "Доступные команды:" in _extract_sse_content(text)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection_input",
    [
        "1",
        "<userRequest>1</userRequest>",
        "<userRequest>(.*?)</userRequest>\n1",
    ],
)
async def test_prompt_select_by_number_variants(client, tmp_path, monkeypatch, selection_input):
    entries = [
        ("copilot", ["line1"]),
        ("ida-pro-mcp", ["line2"]),
        (":test1", ["line3"]),
    ]
    prompt_db = tmp_path / "prompt.db"
    _write_prompt_db(prompt_db, entries)
    monkeypatch.setattr(service, "PROMPT_DB_PATH", prompt_db)

    headers = _session_headers()
    response, text = await _post_chat(client, ":prompt_select", headers=headers)
    assert response.status_code == 200
    content = _extract_sse_content(text)
    assert "Оглавления системных промптов:" in content
    assert "1. copilot" in content
    assert "Выберите номер оглавления" in content

    response, text = await _post_chat(client, selection_input, headers=headers)
    assert response.status_code == 200
    content = _extract_sse_content(text)
    assert "Активирован системный промпт: copilot" in content


@pytest.mark.asyncio
async def test_prompt_list_empty_db(client, tmp_path, monkeypatch):
    prompt_db = tmp_path / "prompt.db"
    monkeypatch.setattr(service, "PROMPT_DB_PATH", prompt_db)

    response, text = await _post_chat(client, ":prompt_list")
    assert response.status_code == 200
    assert "База prompt.db пуста." in _extract_sse_content(text)


@pytest.mark.asyncio
async def test_prompt_list_non_empty_db(client, tmp_path, monkeypatch):
    entries = [
        ("copilot", ["line1"]),
        ("ida-pro-mcp", ["line2"]),
    ]
    prompt_db = tmp_path / "prompt.db"
    _write_prompt_db(prompt_db, entries)
    monkeypatch.setattr(service, "PROMPT_DB_PATH", prompt_db)

    response, text = await _post_chat(client, ":prompt_list")
    assert response.status_code == 200
    content = _extract_sse_content(text)
    assert "Оглавления системных промптов:" in content
    assert "1. copilot" in content
    assert "2. ida-pro-mcp" in content


@pytest.mark.asyncio
async def test_prompt_select_empty_db(client, tmp_path, monkeypatch):
    prompt_db = tmp_path / "prompt.db"
    monkeypatch.setattr(service, "PROMPT_DB_PATH", prompt_db)

    response, text = await _post_chat(client, ":prompt_select")
    assert response.status_code == 200
    assert "База prompt.db пуста." in _extract_sse_content(text)


@pytest.mark.asyncio
@pytest.mark.parametrize("selection_input", ["0", "3"])
async def test_prompt_select_out_of_range(client, tmp_path, monkeypatch, selection_input):
    entries = [
        ("copilot", ["line1"]),
        ("ida-pro-mcp", ["line2"]),
    ]
    prompt_db = tmp_path / "prompt.db"
    _write_prompt_db(prompt_db, entries)
    monkeypatch.setattr(service, "PROMPT_DB_PATH", prompt_db)

    headers = _session_headers("range-session")
    response, text = await _post_chat(client, ":prompt_select", headers=headers)
    assert response.status_code == 200
    assert "Выберите номер оглавления" in _extract_sse_content(text)

    response, text = await _post_chat(client, selection_input, headers=headers)
    assert response.status_code == 200
    content = _extract_sse_content(text)
    assert "Номер вне диапазона" in content
    assert "1..2" in content


@pytest.mark.asyncio
async def test_prompt_select_by_title_still_supported(client, tmp_path, monkeypatch):
    entries = [
        ("copilot", ["line1"]),
        ("ida-pro-mcp", ["line2"]),
    ]
    prompt_db = tmp_path / "prompt.db"
    _write_prompt_db(prompt_db, entries)
    monkeypatch.setattr(service, "PROMPT_DB_PATH", prompt_db)

    headers = _session_headers("title-session")
    response, text = await _post_chat(client, ":prompt_select", headers=headers)
    assert response.status_code == 200
    assert "Выберите номер оглавления" in _extract_sse_content(text)

    response, text = await _post_chat(client, "ida-pro-mcp", headers=headers)
    assert response.status_code == 200
    content = _extract_sse_content(text)
    assert "Активирован системный промпт: ida-pro-mcp" in content


@pytest.mark.asyncio
async def test_prompt_add_flow(client, tmp_path, monkeypatch):
    prompt_db = tmp_path / "prompt.db"
    monkeypatch.setattr(service, "PROMPT_DB_PATH", prompt_db)

    headers = _session_headers("add-session")
    response, text = await _post_chat(client, ":prompt_add", headers=headers)
    assert response.status_code == 200
    assert "Введите оглавление" in _extract_sse_content(text)

    response, text = await _post_chat(client, "new-title", headers=headers)
    assert response.status_code == 200
    assert "Введите содержимое системного промпта" in _extract_sse_content(text)

    response, text = await _post_chat(client, "line1\nline2", headers=headers)
    assert response.status_code == 200
    assert "Промпт сохранён: new-title" in _extract_sse_content(text)
    assert prompt_db.exists()
    db_text = prompt_db.read_text(encoding="utf-8")
    assert "new-title" in db_text
    assert "line1" in db_text
    assert "line2" in db_text
    response, text = await _post_chat(client, ":prompt_list")
    assert response.status_code == 200
    list_content = _extract_sse_content(text)
    assert "1. new-title" in list_content


@pytest.mark.asyncio
async def test_prompt_add_duplicate_title(client, tmp_path, monkeypatch):
    entries = [("dup", ["line1"])]
    prompt_db = tmp_path / "prompt.db"
    _write_prompt_db(prompt_db, entries)
    monkeypatch.setattr(service, "PROMPT_DB_PATH", prompt_db)

    headers = _session_headers("dup-session")
    response, text = await _post_chat(client, ":prompt_add", headers=headers)
    assert response.status_code == 200
    assert "Введите оглавление" in _extract_sse_content(text)

    response, text = await _post_chat(client, "dup", headers=headers)
    assert response.status_code == 200
    content = _extract_sse_content(text)
    assert "Оглавление уже существует" in content


@pytest.mark.asyncio
async def test_prompt_edit_flow(client, tmp_path, monkeypatch):
    entries = [("edit-me", ["old1", "old2"])]
    prompt_db = tmp_path / "prompt.db"
    _write_prompt_db(prompt_db, entries)
    monkeypatch.setattr(service, "PROMPT_DB_PATH", prompt_db)

    headers = _session_headers("edit-session")
    response, text = await _post_chat(client, ":prompt_edit", headers=headers)
    assert response.status_code == 200
    assert "Введите номер оглавления для редактирования" in _extract_sse_content(text)

    response, text = await _post_chat(client, "edit-me", headers=headers)
    assert response.status_code == 200
    assert "Введите новое содержимое системного промпта" in _extract_sse_content(text)

    response, text = await _post_chat(client, "new1\nnew2", headers=headers)
    assert response.status_code == 200
    assert "Промпт обновлён: edit-me" in _extract_sse_content(text)
    db_text = prompt_db.read_text(encoding="utf-8")
    assert "new1" in db_text
    assert "new2" in db_text
    assert "old1" not in db_text
    response, text = await _post_chat(client, ":prompt_list")
    assert response.status_code == 200
    assert "1. edit-me" in _extract_sse_content(text)


@pytest.mark.asyncio
async def test_prompt_edit_title_not_found(client, tmp_path, monkeypatch):
    entries = [("one", ["line1"])]
    prompt_db = tmp_path / "prompt.db"
    _write_prompt_db(prompt_db, entries)
    monkeypatch.setattr(service, "PROMPT_DB_PATH", prompt_db)

    headers = _session_headers("edit-miss-session")
    response, text = await _post_chat(client, ":prompt_edit", headers=headers)
    assert response.status_code == 200
    assert "Введите номер оглавления для редактирования" in _extract_sse_content(text)

    response, text = await _post_chat(client, "missing", headers=headers)
    assert response.status_code == 200
    content = _extract_sse_content(text)
    assert "Оглавление не найдено" in content
    assert ":prompt_list" in content


@pytest.mark.asyncio
async def test_prompt_delete_flow(client, tmp_path, monkeypatch):
    entries = [
        ("del-me", ["line1"]),
        ("keep", ["line2"]),
    ]
    prompt_db = tmp_path / "prompt.db"
    _write_prompt_db(prompt_db, entries)
    monkeypatch.setattr(service, "PROMPT_DB_PATH", prompt_db)

    headers = _session_headers("delete-session")
    response, text = await _post_chat(client, ":prompt_delete", headers=headers)
    assert response.status_code == 200
    assert "Введите номер оглавления для удаления" in _extract_sse_content(text)

    response, text = await _post_chat(client, "del-me", headers=headers)
    assert response.status_code == 200
    assert "Промпт удалён: del-me" in _extract_sse_content(text)
    db_text = prompt_db.read_text(encoding="utf-8")
    assert "del-me" not in db_text
    assert "keep" in db_text
    response, text = await _post_chat(client, ":prompt_list")
    assert response.status_code == 200
    list_content = _extract_sse_content(text)
    assert "del-me" not in list_content
    assert "keep" in list_content


@pytest.mark.asyncio
async def test_prompt_delete_active_prompt(client, tmp_path, monkeypatch):
    entries = [
        ("active", ["line1"]),
        ("other", ["line2"]),
    ]
    prompt_db = tmp_path / "prompt.db"
    _write_prompt_db(prompt_db, entries)
    monkeypatch.setattr(service, "PROMPT_DB_PATH", prompt_db)

    service._active_prompt = ("active", "line1")
    headers = _session_headers("delete-active-session")
    response, text = await _post_chat(client, ":prompt_delete", headers=headers)
    assert response.status_code == 200
    assert "Введите номер оглавления для удаления" in _extract_sse_content(text)

    response, text = await _post_chat(client, "active", headers=headers)
    assert response.status_code == 200
    content = _extract_sse_content(text)
    assert "Промпт удалён и деактивирован: active" in content
    assert service._active_prompt is None


@pytest.mark.asyncio
async def test_prompt_edit_by_number(client, tmp_path, monkeypatch):
    entries = [
        ("first", ["line1"]),
        ("second", ["line2"]),
    ]
    prompt_db = tmp_path / "prompt.db"
    _write_prompt_db(prompt_db, entries)
    monkeypatch.setattr(service, "PROMPT_DB_PATH", prompt_db)

    headers = _session_headers("edit-number-session")
    response, text = await _post_chat(client, ":prompt_edit", headers=headers)
    assert response.status_code == 200
    assert "Введите номер оглавления для редактирования" in _extract_sse_content(text)

    response, text = await _post_chat(client, "2", headers=headers)
    assert response.status_code == 200
    assert "Введите новое содержимое системного промпта" in _extract_sse_content(text)

    response, text = await _post_chat(client, "new-line", headers=headers)
    assert response.status_code == 200
    assert "Промпт обновлён: second" in _extract_sse_content(text)
    db_text = prompt_db.read_text(encoding="utf-8")
    assert "new-line" in db_text
    assert "line2" not in db_text


@pytest.mark.asyncio
async def test_prompt_delete_by_number(client, tmp_path, monkeypatch):
    entries = [
        ("first", ["line1"]),
        ("second", ["line2"]),
    ]
    prompt_db = tmp_path / "prompt.db"
    _write_prompt_db(prompt_db, entries)
    monkeypatch.setattr(service, "PROMPT_DB_PATH", prompt_db)

    headers = _session_headers("delete-number-session")
    response, text = await _post_chat(client, ":prompt_delete", headers=headers)
    assert response.status_code == 200
    assert "Введите номер оглавления для удаления" in _extract_sse_content(text)

    response, text = await _post_chat(client, "1", headers=headers)
    assert response.status_code == 200
    assert "Промпт удалён: first" in _extract_sse_content(text)
    db_text = prompt_db.read_text(encoding="utf-8")
    assert "first" not in db_text
    assert "second" in db_text
