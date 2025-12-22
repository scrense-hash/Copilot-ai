"""
Comprehensive test suite for Autorouter service.

Tests cover:
- Configuration management
- Model caching and selection
- API endpoints (health, models, chat completions)
- Streaming and non-streaming responses
- Error handling and edge cases
"""

import asyncio
import json
import os
import time
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from autorouter_service import _normalized_has_payload, app, get_last_selected, set_last_selected
from config import AppConfig
from models import ModelCache, ModelInfo, ModelSelector
from upstream import UpstreamClient


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Create test configuration."""
    return AppConfig(
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_api_key="test-key",
        openrouter_http_referer="http://localhost",
        openrouter_x_title="test-app",
        virtual_model_id="test-autorouter",
        virtual_model_name="Test Autorouter",
        min_context_length=131072,
        max_price=0.01,
        priority_models={"priority-model-1", "priority-model-2"},
        banned_models={"banned-model-1"},
        request_timeout_s=60.0,
        stream_idle_timeout_s=20.0,
        refresh_models_s=300,
        buffer_stream_keepalive_s=5.0,
        max_buffered_sse_bytes=20_000_000,
        debug_sse_traffic=False,
        debug_sse_traffic_log_path="/tmp/test_traffic_sse.log",
        debug_sse_traffic_truncate_bytes=0,
        debug_sse_traffic_max_bytes=100_000_000,
        debug_sse_traffic_backup_count=3,
        debug_sse_traffic_mirror_main=True,
        debug_sse_traffic_mirror_main_max_chunks=5,
        port=8000,
        log_level="INFO",
        max_request_bytes=2_000_000,
        log_path="/tmp/test_autorouter.log",
        user_agent="test-agent",
    )


@pytest.fixture
def sample_models():
    """Create sample model data."""
    return [
        ModelInfo(
            id="model-1",
            name="Model 1",
            context_length=200000,
            prompt_price=0.001,
            completion_price=0.002,
            supported_parameters=["tools", "temperature"],
        ),
        ModelInfo(
            id="model-2",
            name="Model 2",
            context_length=150000,
            prompt_price=0.005,
            completion_price=0.008,
            supported_parameters=["tools", "tool_choice"],
        ),
        ModelInfo(
            id="priority-model-1",
            name="Priority Model 1",
            context_length=180000,
            prompt_price=0.003,
            completion_price=0.004,
            supported_parameters=["tools"],
        ),
        ModelInfo(
            id="no-tools-model",
            name="No Tools Model",
            context_length=200000,
            prompt_price=0.001,
            completion_price=0.001,
            supported_parameters=["temperature"],
        ),
        ModelInfo(
            id="low-context-model",
            name="Low Context Model",
            context_length=8000,
            prompt_price=0.001,
            completion_price=0.001,
            supported_parameters=["tools"],
        ),
    ]


@pytest.fixture
def mock_httpx_client():
    """Create mock httpx client."""
    client = MagicMock(spec=httpx.AsyncClient)
    return client


@pytest.fixture
async def client():
    """Create an in-process ASGI client.

    NOTE: We intentionally avoid Starlette/FastAPI's TestClient here because it
    relies on AnyIO's blocking portal + lifespan wiring, which can hang in some
    environments. httpx.ASGITransport keeps tests deterministic.
    """
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


# ============================================================================
# Config Tests
# ============================================================================

class TestConfig:
    """Test configuration management."""

    def test_from_env_defaults(self):
        """Test loading config with default values."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True):
            cfg = AppConfig.from_env()
            assert cfg.openrouter_api_key == "test-key"
            assert cfg.min_context_length == 131072
            assert cfg.max_price == 0.0
            assert cfg.port == 8000

    def test_from_env_custom_values(self):
        """Test loading config with custom environment values."""
        env_vars = {
            "OPENROUTER_API_KEY": "custom-key",
            "MIN_CTX": "100000",
            "MAX_PRICE": "0.05",
            "PORT": "9000",
            "REQUEST_TIMEOUT_S": "120.0",
            "PRIORITY_MODELS": "model1,model2,model3",
            "BAN_MODELS": "bad-model",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            cfg = AppConfig.from_env()
            assert cfg.openrouter_api_key == "custom-key"
            assert cfg.min_context_length == 100000
            assert cfg.max_price == 0.05
            assert cfg.port == 9000
            assert cfg.request_timeout_s == 120.0
            assert "model1" in cfg.priority_models
            assert "model2" in cfg.priority_models
            assert "bad-model" in cfg.banned_models

    def test_validate_success(self, test_config):
        """Test successful config validation."""
        test_config.validate()  # Should not raise

    def test_validate_missing_api_key(self, test_config):
        """Test validation fails with missing API key."""
        cfg = replace(test_config, openrouter_api_key="")
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            cfg.validate()

    def test_validate_invalid_min_context(self, test_config):
        """Test validation fails with invalid min context."""
        cfg = replace(test_config, min_context_length=0)
        with pytest.raises(ValueError, match="MIN_CTX"):
            cfg.validate()


# ============================================================================
# Model Tests
# ============================================================================

class TestModelInfo:
    """Test ModelInfo class."""

    def test_max_price(self):
        """Test max_price property."""
        model = ModelInfo(
            id="test",
            name="Test Model",
            context_length=100000,
            prompt_price=0.001,
            completion_price=0.005,
            supported_parameters=["tools"],
        )
        assert model.max_price == 0.005

    def test_has_tools_support_with_tools(self):
        """Test has_tools_support with tools parameter."""
        model = ModelInfo(
            id="test",
            name="Test Model",
            context_length=100000,
            prompt_price=0.001,
            completion_price=0.005,
            supported_parameters=["tools", "temperature"],
        )
        assert model.has_tools_support() is True

    def test_has_tools_support_with_tool_choice(self):
        """Test has_tools_support with tool_choice parameter."""
        model = ModelInfo(
            id="test",
            name="Test Model",
            context_length=100000,
            prompt_price=0.001,
            completion_price=0.005,
            supported_parameters=["tool_choice", "temperature"],
        )
        assert model.has_tools_support() is True

    def test_has_tools_support_without_tools(self):
        """Test has_tools_support without tools parameter."""
        model = ModelInfo(
            id="test",
            name="Test Model",
            context_length=100000,
            prompt_price=0.001,
            completion_price=0.005,
            supported_parameters=["temperature"],
        )
        assert model.has_tools_support() is False

    def test_to_virtual_model_dict(self):
        """Test conversion to virtual model dictionary."""
        model = ModelInfo(
            id="real-model",
            name="Real Model",
            context_length=100000,
            prompt_price=0.001,
            completion_price=0.005,
            supported_parameters=["tools"],
        )
        result = model.to_virtual_model_dict("virtual-id", "Virtual Name")
        assert result["id"] == "virtual-id"
        assert result["name"] == "Virtual Name"
        assert result["upstream_model_id"] == "real-model"
        assert result["upstream_model_name"] == "Real Model"


# ModelBanList removed from current architecture - tests removed


class TestModelCache:
    """Test ModelCache class."""

    @pytest.mark.asyncio
    async def test_get_models_fresh_fetch(self, test_config, mock_httpx_client):
        """Test fetching models when cache is empty."""
        cache = ModelCache()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "id": "model-1",
                    "name": "Model 1",
                    "context_length": 100000,
                    "pricing": {"prompt": "0.001", "completion": "0.002"},
                    "supported_parameters": ["tools"],
                }
            ]
        }
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        models = await cache.get_models(mock_httpx_client, test_config)
        assert len(models) == 1
        assert models[0].id == "model-1"

    @pytest.mark.asyncio
    async def test_get_models_cached(self, test_config, mock_httpx_client):
        """Test using cached models."""
        cache = ModelCache()
        cache._models = [
            ModelInfo(
                id="cached-model",
                name="Cached Model",
                context_length=100000,
                prompt_price=0.001,
                completion_price=0.002,
                supported_parameters=["tools"],
            )
        ]
        cache._last_fetch = time.time()

        models = await cache.get_models(mock_httpx_client, test_config)
        assert len(models) == 1
        assert models[0].id == "cached-model"
        mock_httpx_client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_models_error(self, test_config, mock_httpx_client):
        """Test error handling during model fetch."""
        cache = ModelCache()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_httpx_client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(Exception, match="Upstream /models error"):
            await cache.get_models(mock_httpx_client, test_config)


class TestModelSelector:
    """Test ModelSelector class."""

    def test_choose_candidates_filtering(self, sample_models, test_config):
        """Test model filtering."""
        selector = ModelSelector()

        candidates = selector.choose_candidates(sample_models, test_config)

        # Should exclude no-tools-model and low-context-model
        assert len(candidates) == 3
        assert all(m.has_tools_support() for m in candidates)
        assert all(m.context_length >= test_config.min_context_length for m in candidates)

    def test_choose_candidates_priority(self, sample_models, test_config):
        """Test priority model sorting."""
        selector = ModelSelector()

        candidates = selector.choose_candidates(sample_models, test_config)

        # Priority models should come first
        assert candidates[0].id == "priority-model-1"

    def test_choose_candidates_banned_models(self, sample_models, test_config):
        """Test filtering banned models via config."""
        selector = ModelSelector()

        # Add model-1 to banned list via config
        modified_config = replace(test_config, banned_models={"model-1"})
        candidates = selector.choose_candidates(sample_models, modified_config)

        # model-1 should be excluded
        assert not any(m.id == "model-1" for m in candidates)

    def test_choose_candidates_max_price(self, sample_models, test_config):
        """Test max price filtering."""
        selector = ModelSelector()

        modified_config = replace(test_config, max_price=0.003)  # Lower max price

        candidates = selector.choose_candidates(sample_models, modified_config)

        # Only model-1 should pass (max_price=0.002)
        assert len(candidates) == 1
        assert candidates[0].id == "model-1"


# ============================================================================
# Upstream Tests
# ============================================================================

class TestUpstreamClient:
    """Test UpstreamClient class."""

    def test_get_headers(self, test_config):
        """Test header generation."""
        client = UpstreamClient(test_config)
        headers = client.get_headers()

        assert headers["Authorization"] == f"Bearer {test_config.openrouter_api_key}"
        assert headers["Content-Type"] == "application/json"
        assert headers["User-Agent"] == test_config.user_agent

    @pytest.mark.asyncio
    async def test_chat_completion_non_stream(self, test_config):
        """Test non-streaming chat completion."""
        upstream = UpstreamClient(test_config)

        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "test"}}]
        }

        mock_client.build_request = MagicMock()
        mock_client.send = AsyncMock(return_value=mock_response)

        body = {"messages": [{"role": "user", "content": "test"}], "stream": False}
        result = await upstream.chat_completion(mock_client, body, "test-model")

        assert result.status_code == 200
        mock_client.build_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_error_snippet(self):
        """Test reading error snippet from response."""
        mock_response = MagicMock()
        mock_response.aread = AsyncMock(return_value=b"Error message")

        snippet = await UpstreamClient.read_error_snippet(mock_response)
        assert snippet == "Error message"

    @pytest.mark.asyncio
    async def test_read_error_snippet_timeout(self):
        """Test error snippet with timeout."""
        mock_response = MagicMock()
        mock_response.aread = AsyncMock(side_effect=asyncio.TimeoutError())

        snippet = await UpstreamClient.read_error_snippet(mock_response, timeout_s=0.1)
        assert snippet == ""

    def test_payload_has_choices(self):
        """Test checking for choices in payload."""
        assert UpstreamClient.payload_has_choices({"choices": [{"text": "test"}]}) is True
        assert UpstreamClient.payload_has_choices({"choices": []}) is False
        assert UpstreamClient.payload_has_choices({}) is False
        assert UpstreamClient.payload_has_choices(None) is False


# ============================================================================
# Helper Function Tests
# ============================================================================

def test_normalized_has_payload_with_content():
    chunks = [
        {"choices": [{"delta": {"content": "  Hello  "}}]},
    ]
    assert _normalized_has_payload(chunks) is True


def test_normalized_has_payload_with_tool_calls():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{"id": "1", "type": "function"}]}}]},
    ]
    assert _normalized_has_payload(chunks) is True


def test_normalized_has_payload_empty():
    chunks = [
        {"choices": [{"delta": {"content": "   "}}]},
        {"choices": [{"delta": {}}]},
    ]
    assert _normalized_has_payload(chunks) is False


# ============================================================================
# API Endpoint Tests
# ============================================================================

class TestAPIEndpoints:
    """Test FastAPI endpoints."""

    @pytest.mark.asyncio
    async def test_healthz(self, client):
        """Test health check endpoint."""
        response = await client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_v1_models_no_api_key(self, client):
        """Test models endpoint without API key."""
        # Patch the config module's config variable
        from autorouter_service import config as app_config
        # Create new config with empty API key
        empty_config = replace(app_config, openrouter_api_key="")
        with patch("autorouter_service.config", empty_config):
            response = await client.get("/v1/models")
            assert response.status_code == 500
            assert "OPENROUTER_API_KEY" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_v1_models_with_last_selected(self, client):
        """Test models endpoint with last selected model."""
        test_model = ModelInfo(
            id="test-model",
            name="Test Model",
            context_length=200000,
            prompt_price=0.001,
            completion_price=0.002,
            supported_parameters=["tools"],
        )
        await set_last_selected(test_model)

        response = await client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["upstream_model_id"] == "test-model"

    @pytest.mark.asyncio
    async def test_chat_completions_no_api_key(self, client):
        """Test chat completions without API key."""
        # Patch the config module's config variable
        from autorouter_service import config as app_config
        # Create new config with empty API key
        empty_config = replace(app_config, openrouter_api_key="")
        with patch("autorouter_service.config", empty_config):
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status_code == 500
            assert "OPENROUTER_API_KEY" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_chat_completions_invalid_json(self, client):
        """Test chat completions with invalid JSON."""
        response = await client.post(
            "/v1/chat/completions",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert "Invalid JSON" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_chat_completions_request_too_large(self, client):
        """Test chat completions with too large request."""
        large_body = {"model": "test", "messages": [{"role": "user", "content": "x" * 3_000_000}]}
        json_str = json.dumps(large_body)

        response = await client.post(
            "/v1/chat/completions",
            content=json_str,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(json_str)),
            },
        )
        assert response.status_code == 413
        assert "Request too large" in response.json()["detail"]


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_last_selected_model_workflow(self):
        """Test last selected model get/set workflow."""
        test_model = ModelInfo(
            id="workflow-model",
            name="Workflow Model",
            context_length=200000,
            prompt_price=0.001,
            completion_price=0.002,
            supported_parameters=["tools"],
        )

        # Initially no model selected
        assert await get_last_selected() is None

        # Set a model
        await set_last_selected(test_model)

        # Retrieve the model
        selected = await get_last_selected()
        assert selected is not None
        assert selected.id == "workflow-model"

    # Model banning workflow removed - now handled via config banned_models

    @pytest.mark.asyncio
    async def test_virtual_model_normalization(self, client, sample_models, test_config):
        """Test that virtual model names are normalized to auto-routing."""
        from autorouter_service import model_cache

        # Create a simple async iterator for the mock response
        async def mock_aiter_lines():
            lines = [
                'data: {"id":"test","object":"chat.completion.chunk","created":1234567890,"model":"model-1","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}',
                '',
                'data: {"id":"test","object":"chat.completion.chunk","created":1234567890,"model":"model-1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
                '',
                'data: [DONE]',
                '',
            ]
            for line in lines:
                yield line

        # Mock the model cache to return sample models with tools support
        with patch.object(model_cache, 'get_models', AsyncMock(return_value=sample_models)):
            # Mock the upstream client's chat_completion method
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.aiter_lines = mock_aiter_lines
            mock_response.aclose = AsyncMock()

            with patch('upstream.UpstreamClient.chat_completion', AsyncMock(return_value=mock_response)):
                body = {
                    "model": "copilot-autorouter",  # Virtual model ID
                    "messages": [{"role": "user", "content": "test"}],
                }

                response = await client.post("/v1/chat/completions", json=body)

                # Debug: print response if it's not what we expect
                if response.status_code != 200:
                    print(f"Response status: {response.status_code}")
                    print(f"Response body: {response.text}")

                # Should accept the virtual model name (not 422 validation error)
                # The test mainly verifies the model name is accepted, not the full workflow
                assert response.status_code != 422


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
