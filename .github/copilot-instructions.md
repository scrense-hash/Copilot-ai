# Autorouter AI Coding Agent Instructions

This document provides guidelines for AI coding assistants working on the Autorouter project. Autorouter is an OpenAI-compatible proxy service that intelligently routes requests to various upstream models via OpenRouter.

## Architecture Overview

The service follows a layered architecture with clear separation of concerns:

```
Client → FastAPI → Model Selection → Upstream Client → OpenRouter API
          ↓              ↓              ↓              ↓
      Request     Model Filtering   Tool ID Mapping   SSE Validation
      Validation  & Prioritization  & Normalization   & Buffering
```

**Key Components:**
- `autorouter_service.py`: FastAPI application with main endpoints (`/v1/chat/completions`, `/v1/models`, `/healthz`, `/logs`)
- `config.py`: Environment-based configuration with validation
- `models.py`: Model metadata (`ModelInfo`), caching (`ModelCache`), and selection logic (`ModelSelector`)
- `upstream.py`: OpenRouter API communication (`UpstreamClient`) with tool ID mapping
- `sse_handler.py`: SSE stream handling (`SSEValidator`, `SSEStreamer`) with prebuffering and normalization
- `logger.py`: Structured logging with rotation and ANSI color support
- `utils.py`: Utility functions for environment loading and configuration dumping

## Key Conventions and Patterns

### 1. Configuration via Environment Variables
- **Primary Pattern**: All configuration is loaded from environment variables using `config.py`
- **Validation**: Configuration is validated at startup via `AppConfig.validate()`
- **Documentation**: See `ENV_PARAMS_RU.md` for detailed parameter descriptions
- **Example Critical Parameters**:
  - `OPENROUTER_API_KEY`: Required for API access
  - `MIN_CTX`: Minimum context length (default: 131072)
  - `MAX_PRICE`: Maximum price per 1M tokens (default: 0.0 - free only)
  - `PRIORITY_MODELS`: Comma-separated list of preferred models
  - `BAN_MODELS`: Comma-separated list of banned models
  - `DEBUG_SSE_TRAFFIC`: Enable verbose SSE traffic logging

### 2. Model Selection Process
The `ModelSelector` implements a sophisticated filtering and prioritization system:

```python
# Filtering logic in models.py
1. Exclude banned models (config.banned_models)
2. Require minimum context length (config.min_context_length)
3. Require tools support (tools or tool_choice in supported_parameters)
4. Apply price limit (config.max_price)

# Sorting logic
1. Priority models first (config.priority_models)
2. Then by price (lowest first)
3. Then by context length (highest first)
```

### 3. SSE Streaming Architecture
The service implements two streaming modes:

**Buffered Stream Mode** (default for virtual model):
- Buffers entire upstream response until `[DONE]` is received
- Validates and normalizes chunks before forwarding to client
- Prevents client errors from malformed SSE formats
- Implements watchdog timeout (`STREAM_IDLE_TIMEOUT_S`)

**Direct Stream Mode** (for explicit upstream models):
- Data is forwarded immediately but with first-event validation
- Used when client explicitly requests a specific upstream model

### 4. Tool Call Normalization
Critical pattern for Copilot compatibility:

**Problem**: Some models return tool calls as inline XML markup:
```xml
<tool_call>
  <function name="get_weather">
    <parameter name="city">Moscow</parameter>
  </function>
</tool_call>
```

**Solution**: `normalize_upstream_chunk_to_openai()` converts this to standard format:
```json
{
  "tool_calls": [{
    "id": "TLCLcpcErK",
    "function": {
      "arguments": "{\"city\": \"Moscow\"}",
      "name": "get_weather"
    },
    "type": "function"
  }]
}
```

### 5. Error Handling and Failover
- **Automatic Retry**: If a model fails, the system automatically tries the next candidate
- **Temporary Bans**: Models with issues (inline tool markup, early EOF) are temporarily banned
- **Graceful Degradation**: Always returns structured error responses to prevent "Sorry, no response was returned" messages

### 6. Logging and Observability
- **Structured Logging**: Uses `loguru` with `req_id` correlation
- **Traffic Logging**: Optional verbose SSE traffic logging (`DEBUG_SSE_TRAFFIC`)
- **Web UI**: Built-in logs viewer at `/logs` with ANSI color support
- **Health Checks**: `/healthz` endpoint for monitoring

## Developer Workflows

### Testing
```bash
# Run all tests
./run_tests.sh

# Run with coverage
./run_tests.sh -c

# Run specific test file
./run_tests.sh -t test_autorouter.py

# Run in parallel
./run_tests.sh -p
```

**Test Structure**:
- `tests/test_autorouter.py`: Main tests (config, models, API endpoints)
- `tests/test_sse_handler.py`: SSE handling tests
- `tests/README.md`: Comprehensive testing documentation

### Environment Setup
```bash
# Always use the virtual environment
source .venv/bin/activate

# Install development dependencies
pip install -r tests/requirements-dev.txt
```

### Development Dependencies
- **Core**: `fastapi`, `httpx`, `loguru`, `python-dotenv`
- **Testing**: `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-xdist`
- **Linting**: `flake8` (config in `.flake8`)

### System Prompt Injection
- Loaded from `router_prompt.txt` at startup
- Automatically injected into every request as a system message
- Supports multi-line prompts with exact newline preservation

## Critical Code Patterns

### 1. Tool ID Mapping
`upstream.py` implements bidirectional tool ID mapping for strict providers:

```python
# Convert long tool_call_ids to 9-char base62 format
def _base62_from_sha1(s: str, length: int = 9) -> str:
    hash_bytes = hashlib.sha1(s.encode("utf-8", errors="ignore")).digest()
    # ... conversion logic
```

### 2. SSE Validation and Repair
`sse_handler.py` implements stateful tool markup repair:

```python
class ToolCallTextRepairer:
    """Stateful repair for providers that emit tool calls as tagged text."""
    def consume(self, content: str) -> tuple[str, list[dict] | None, bool]:
        # Buffers partial markup until complete, then converts to native tool_calls
```

### 3. Buffered Streaming Logic
The core routing function in `autorouter_service.py`:

```python
async def handle_auto_route_request_buffered_stream(
    client: httpx.AsyncClient,
    body: Dict[str, Any],
    req_id: str,
) -> Response:
    # 1. Get models from cache
    # 2. Filter and sort candidates
    # 3. Try each candidate with failover
    # 4. Buffer until [DONE]
    # 5. Normalize and forward
```

### 4. Request Size Guarding
```python
# In v1_chat_completions endpoint
cl = request.headers.get("content-length")
if cl and int(cl) > config.max_request_bytes:
    raise HTTPException(status_code=413, detail="Request too large")
```

## Integration Points

### OpenRouter API
- **Base URL**: `https://openrouter.ai/api/v1`
- **Endpoints**: `/models`, `/chat/completions`
- **Headers**: `Authorization`, `HTTP-Referer`, `X-Title`

### FastAPI Application
- **Port**: Configurable via `PORT` (default: 8000)
- **CORS**: Enabled for web UI access
- **Endpoints**:
  - `POST /v1/chat/completions` - Main chat endpoint
  - `GET /v1/models` - List available models
  - `GET /healthz` - Health check
  - `GET /logs` - Web-based log viewer
  - `GET /logs/sse` - SSE log streaming
  - `POST /logs/download` - Log download

## Project-Specific Conventions

1. **Request ID Correlation**: All logs include `req_id` for tracing
2. **Traffic Mirroring**: Limited SSE traffic mirrored to main log (configurable)
3. **Watchdog Timers**: Prevent stalled connections
4. **Automatic Failover**: Round-robin retry with model rotation
5. **Stateful Repair**: Tool markup repair across chunk boundaries
6. **Strict Validation**: Early detection of problematic models

## Key Files for Understanding

- `autorouter_service.py`: Main application logic and endpoints
- `config.py`: Configuration management and validation
- `models.py`: Model selection and caching logic
- `sse_handler.py`: SSE stream handling and validation
- `upstream.py`: OpenRouter API communication
- `tests/test_autorouter.py`: Comprehensive test suite
- `ENV_PARAMS_RU.md`: Detailed parameter documentation

## Common Pitfalls

1. **Missing API Key**: Always ensure `OPENROUTER_API_KEY` is set
2. **Virtual Environment**: Always use `.venv/bin/activate`
3. **Request Size Limits**: Large requests are rejected (configurable)
4. **SSE Timeouts**: Watchdog closes idle connections
5. **Tool Markup**: Incomplete tool markup triggers model bans

## Getting Help

For questions about specific components:
- Model selection: See `ModelSelector` in `models.py`
- SSE handling: See `SSEValidator` and `SSEStreamer` in `sse_handler.py`
- Configuration: See `AppConfig` in `config.py`
- Testing: See `tests/README.md`

This documentation provides the essential knowledge to be immediately productive in the Autorouter codebase. Focus on understanding the model selection process, SSE streaming architecture, and tool call normalization patterns, as these are the core differentiators of this project.
