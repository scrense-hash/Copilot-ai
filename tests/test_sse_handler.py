"""
Tests for SSE (Server-Sent Events) handler module.

Tests cover:
- SSE line parsing and validation
- Inline tool call detection
- Stream prebuffering and validation
- Watchdog functionality
- Error response generation
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from sse_handler import (
    SSEStreamer,
    SSEValidator,
    chunk_has_done_data_line,
    extract_content_fragments,
    is_done_data_line,
    is_sse_activity_line,
    looks_like_inline_tool_call,
)


# ============================================================================
# Helper Functions Tests
# ============================================================================

class TestSSEHelpers:
    """Test SSE helper functions."""

    def test_is_sse_activity_line(self):
        """Test SSE activity line detection."""
        assert is_sse_activity_line("data: test") is True
        assert is_sse_activity_line("event: message") is True
        assert is_sse_activity_line("id: 123") is True
        assert is_sse_activity_line("retry: 1000") is True
        assert is_sse_activity_line(": comment") is True
        assert is_sse_activity_line(" continuation") is True
        assert is_sse_activity_line("not an sse line") is False
        assert is_sse_activity_line("") is False

    def test_is_done_data_line(self):
        """Test DONE data line detection."""
        assert is_done_data_line("data:[DONE]") is True
        assert is_done_data_line("data: [DONE]") is True
        assert is_done_data_line("data:  [DONE]  ") is True
        assert is_done_data_line("data: [DONE] extra") is False
        assert is_done_data_line("data: something") is False
        assert is_done_data_line("not data") is False

    def test_chunk_has_done_data_line(self):
        """Test DONE detection in byte chunks."""
        assert chunk_has_done_data_line(b"data: [DONE]\n\n") is True
        assert chunk_has_done_data_line(b"data: test\n\ndata: [DONE]\n\n") is True
        assert chunk_has_done_data_line(b"data: test\n\n") is False
        assert chunk_has_done_data_line(b"") is False
        assert chunk_has_done_data_line(b"invalid utf-8 \xff\xfe") is False

    def test_looks_like_inline_tool_call(self):
        """Test inline tool call detection."""
        # Valid inline tool call
        valid_call = '<tool_call><function=test><parameter=value></parameter></function></tool_call>'
        assert looks_like_inline_tool_call(valid_call) is True

        # Case insensitive
        valid_call_upper = '<TOOL_CALL><FUNCTION=test><PARAMETER=value></PARAMETER></FUNCTION></TOOL_CALL>'
        assert looks_like_inline_tool_call(valid_call_upper) is True

        # Missing closing tag
        invalid_call = '<tool_call><function=test><parameter=value></parameter></function>'
        assert looks_like_inline_tool_call(invalid_call) is False

        # Missing function
        invalid_call2 = '<tool_call><parameter=value></parameter></tool_call>'
        assert looks_like_inline_tool_call(invalid_call2) is False

        # Regular text
        assert looks_like_inline_tool_call("just regular text") is False
        assert looks_like_inline_tool_call("") is False

    def test_extract_content_fragments(self):
        """Test extracting content fragments from SSE data."""
        # Valid data with content
        data = {
            "choices": [
                {"delta": {"content": "Hello"}},
                {"delta": {"content": "World"}},
            ]
        }
        result = extract_content_fragments(data)
        assert result == ["Hello", "World"]

        # Message instead of delta
        data2 = {
            "choices": [
                {"message": {"content": "Test"}},
            ]
        }
        result2 = extract_content_fragments(data2)
        assert result2 == ["Test"]

        # No content
        data3 = {
            "choices": [
                {"delta": {}},
            ]
        }
        result3 = extract_content_fragments(data3)
        assert result3 == []

        # Empty choices
        data4 = {"choices": []}
        result4 = extract_content_fragments(data4)
        assert result4 == []

        # Invalid input
        assert extract_content_fragments(None) == []
        assert extract_content_fragments("string") == []


# ============================================================================
# SSEValidator Tests
# ============================================================================

class TestSSEValidator:
    """Test SSE validation and prebuffering."""

    @pytest.fixture
    def validator(self):
        """Create SSE validator."""
        return SSEValidator()

    @pytest.mark.asyncio
    async def test_peek_first_sse_choices_success(self, validator):
        """Test successful first SSE chunk peek."""
        lines = [
            "event: ping",  # Meta event before data
            "",  # End meta event
            "",  # Empty keepalive line
            "data: " + json.dumps({"choices": [{"delta": {"content": "test"}}]}),
            "",
        ]

        async def mock_aiter():
            for line in lines:
                yield line

        mock_response = MagicMock()
        mock_response.aiter_lines = lambda: mock_aiter()

        ok, first_event, _, reason = await validator.peek_first_sse_choices(mock_response.aiter_lines())

        assert ok is True
        assert first_event is not None
        assert reason == ""

    @pytest.mark.asyncio
    async def test_peek_first_sse_choices_no_choices(self, validator):
        """Test first SSE chunk without choices."""
        lines = [
            "data: " + json.dumps({"no_choices": True}),
        ]

        async def mock_aiter():
            for line in lines:
                yield line

        mock_response = MagicMock()
        mock_response.aiter_lines = lambda: mock_aiter()

        ok, _, _, reason = await validator.peek_first_sse_choices(mock_response.aiter_lines())

        assert ok is False
        assert "no-choices" in reason or "no choices" in reason

    @pytest.mark.asyncio
    async def test_peek_first_sse_choices_done_first(self, validator):
        """Test stream that sends DONE before any data."""
        lines = ["data: [DONE]"]

        async def mock_aiter():
            for line in lines:
                yield line

        mock_response = MagicMock()
        mock_response.aiter_lines = lambda: mock_aiter()

        ok, _, _, reason = await validator.peek_first_sse_choices(mock_response.aiter_lines())

        assert ok is False
        assert reason != ""

    @pytest.mark.asyncio
    async def test_peek_first_sse_choices_eof(self, validator):
        """Test stream that ends before any data."""
        async def mock_aiter():
            return
            yield  # Never reached

        mock_response = MagicMock()
        mock_response.aiter_lines = lambda: mock_aiter()

        ok, _, _, reason = await validator.peek_first_sse_choices(mock_response.aiter_lines())

        assert ok is False
        assert reason != ""

    @pytest.mark.asyncio
    async def test_prebuffer_success(self, validator):
        """Test successful prebuffering."""
        first_bytes = b'data: ' + json.dumps({"choices": [{"delta": {"content": "first"}}]}).encode() + b'\n\n'

        lines = [
            'data: ' + json.dumps({"choices": [{"delta": {"content": "second"}}]}),
            "",
            'data: [DONE]',
            "",
        ]

        async def mock_aiter():
            for line in lines:
                yield line

        ok, initial_bytes, reason = await validator.prebuffer_before_commit(
            aiter=mock_aiter(),
            first_bytes=first_bytes,
            min_data_events=2,
            window_s=10.0,
            idle_timeout_s=5.0,
        )

        assert ok is True
        assert len(initial_bytes) >= 2
        assert reason is None

    @pytest.mark.asyncio
    async def test_prebuffer_inline_tool_detection(self, validator):
        """Test detection of inline tool calls during prebuffering."""
        tool_call = '<tool_call><function=test><parameter=value></parameter></function></tool_call>'
        first_bytes = b'data: ' + json.dumps({
            "choices": [{"delta": {"content": tool_call}}]
        }).encode() + b'\n\n'

        async def mock_aiter():
            return
            yield  # Never reached

        ok, _, reason = await validator.prebuffer_before_commit(
            aiter=mock_aiter(),
            first_bytes=first_bytes,
            min_data_events=1,
            window_s=10.0,
            idle_timeout_s=5.0,
        )

        assert ok is False
        assert reason == "inline-tool-markup"

    @pytest.mark.asyncio
    async def test_prebuffer_early_eof(self, validator):
        """Test early EOF detection during prebuffering."""
        first_bytes = b'data: ' + json.dumps({"choices": [{"delta": {"content": "test"}}]}).encode() + b'\n\n'

        async def mock_aiter():
            return  # Immediate EOF
            yield

        ok, _, reason = await validator.prebuffer_before_commit(
            aiter=mock_aiter(),
            first_bytes=first_bytes,
            min_data_events=3,  # Require more events than provided
            window_s=10.0,
            idle_timeout_s=5.0,
        )

        assert ok is False
        assert "eof" in reason.lower()


# ============================================================================
# SSEStreamer Tests
# ============================================================================

class TestSSEStreamer:
    """Test SSE streaming functionality."""

    @pytest.mark.asyncio
    async def test_stream_with_watchdog_normal(self):
        """Test normal streaming without watchdog trigger."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.aclose = AsyncMock()
        mock_client.aclose = AsyncMock()

        initial_bytes = [b"data: initial\n\n"]
        lines = ["data: line1", "", "data: line2", "", "data: [DONE]", ""]

        async def mock_aiter():
            for line in lines:
                yield line

        result_bytes = []
        async for chunk in SSEStreamer.stream_with_watchdog(
            client=mock_client,
            resp=mock_response,
            aiter=mock_aiter(),
            initial_bytes=initial_bytes,
            idle_timeout_s=10.0,
            req_id="test-req",
            model_id="test-model",
        ):
            result_bytes.append(chunk)

        # Check that initial bytes and stream data were yielded
        assert len(result_bytes) > 0
        assert b"initial" in result_bytes[0]

    @pytest.mark.asyncio
    async def test_stream_with_watchdog_stall(self):
        """Test watchdog closes stalled stream."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.aclose = AsyncMock()
        mock_client.aclose = AsyncMock()

        initial_bytes = [b"data: initial\n\n"]

        async def mock_aiter():
            yield "data: first"
            yield ""
            await asyncio.sleep(3.0)  # Simulate stall
            yield "data: never reached"
            yield ""

        result_bytes = []
        async for chunk in SSEStreamer.stream_with_watchdog(
            client=mock_client,
            resp=mock_response,
            aiter=mock_aiter(),
            initial_bytes=initial_bytes,
            idle_timeout_s=1.0,  # Short timeout
            req_id="test-req",
            model_id="test-model",
        ):
            result_bytes.append(chunk)

        # Should have initial bytes and first data, plus auto-generated DONE
        assert len(result_bytes) > 0
        # Check if DONE was added
        assert any(b"[DONE]" in chunk for chunk in result_bytes)

    @pytest.mark.asyncio
    async def test_stream_with_watchdog_eof_injects_done(self):
        """EOF without explicit [DONE] should be terminated cleanly."""
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.aclose = AsyncMock()
        mock_client.aclose = AsyncMock()

        async def mock_aiter():
            yield "data: hello"
            yield ""
            return
            yield

        result_bytes = []
        async for chunk in SSEStreamer.stream_with_watchdog(
            client=mock_client,
            resp=mock_response,
            aiter=mock_aiter(),
            initial_bytes=[],
            idle_timeout_s=10.0,
            req_id="test-req",
            model_id="test-model",
        ):
            result_bytes.append(chunk)

        assert result_bytes
        assert not any(b"[DONE]" in chunk for chunk in result_bytes)

    # Ban-related tests removed - ban system no longer exists in current architecture

    @pytest.mark.asyncio
    async def test_error_response(self):
        """Test error response generation."""
        result_bytes = []
        async for chunk in SSEStreamer.error_response("Test error message", "test-model"):
            result_bytes.append(chunk)

        # Should have error message and DONE
        assert len(result_bytes) == 2
        assert b"Test error message" in result_bytes[0]
        assert b"[DONE]" in result_bytes[1]

        # Parse the error chunk
        error_data = result_bytes[0].decode("utf-8")
        assert "data:" in error_data
        json_part = error_data.split("data:")[1].strip()
        parsed = json.loads(json_part)
        assert parsed["model"] == "test-model"
        assert parsed["choices"][0]["delta"]["content"] == "Test error message"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
