"""Essential Sprint 4 tests: Streamable HTTP transport, Bearer authentication, and scoped access."""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent_core import BenchmarkRadarAgent
from src.mcp_server import (
    BearerTokenVerifier,
    HTTP_AUTH_AVAILABLE,
    READ_ONLY_SCOPE,
    READ_WRITE_SCOPE,
    ScopeEnforcementMiddleware,
    resolve_transport,
)

pytestmark = pytest.mark.skipif(
    not HTTP_AUTH_AVAILABLE,
    reason="FastMCP HTTP auth primitives not available",
)


def make_token(scopes):
    """Helper to mock an AccessToken with specific scopes."""
    return SimpleNamespace(token="test-token", client_id="test-client", scopes=list(scopes))


# ============================================================================
# 1. Bearer Token Verification & Scope Mapping
# ============================================================================

@pytest.mark.asyncio
async def test_bearer_token_verifier():
    """Verify write key maps to read-write, read-only key to read-only, and unknown tokens fail."""
    verifier = BearerTokenVerifier(write_key="secret-write", read_only_key="secret-read")

    # Write key grants full read-write scope
    write_token = await verifier.verify_token("secret-write")
    assert write_token is not None
    assert write_token.scopes == [READ_WRITE_SCOPE]

    # Read-only key grants restricted read-only scope
    read_token = await verifier.verify_token("secret-read")
    assert read_token is not None
    assert read_token.scopes == [READ_ONLY_SCOPE]

    # Invalid and empty tokens are rejected (HTTP 401 path)
    assert await verifier.verify_token("wrong-token") is None
    assert await verifier.verify_token("") is None
    assert await verifier.verify_token(None) is None


# ============================================================================
# 2. Scope Enforcement Middleware
# ============================================================================

@pytest.mark.asyncio
async def test_scope_enforcement_middleware():
    """Verify write tool is blocked for read-only tokens and allowed for write tokens."""
    middleware = ScopeEnforcementMiddleware()
    save_ctx = SimpleNamespace(message=SimpleNamespace(name="save_research_brief"))
    query_ctx = SimpleNamespace(message=SimpleNamespace(name="query_benchmarks"))
    inner = AsyncMock(return_value="tool-result")

    # Read-only token cannot save research briefs
    with patch("src.mcp_server.get_access_token", return_value=make_token([READ_ONLY_SCOPE])):
        with pytest.raises(Exception, match="Access denied"):
            await middleware.on_call_tool(save_ctx, inner)
    inner.assert_not_called()

    # Read-write token can save research briefs
    with patch("src.mcp_server.get_access_token", return_value=make_token([READ_WRITE_SCOPE])):
        result = await middleware.on_call_tool(save_ctx, inner)
        assert result == "tool-result"
    assert inner.call_count == 1

    # Read-only token can call read-only tools without restrictions
    inner.reset_mock()
    with patch("src.mcp_server.get_access_token", return_value=make_token([READ_ONLY_SCOPE])):
        result = await middleware.on_call_tool(query_ctx, inner)
        assert result == "tool-result"
    assert inner.call_count == 1


# ============================================================================
# 3. Transport Resolution
# ============================================================================

def test_resolve_transport():
    """Verify CLI argument overrides MCP_TRANSPORT env var, defaulting to stdio."""
    with patch.dict(os.environ, {"MCP_TRANSPORT": ""}):
        assert resolve_transport("") == "stdio"
        assert resolve_transport("http") == "http"
        assert resolve_transport("stdio") == "stdio"
        assert resolve_transport("unknown") == "stdio"

    with patch.dict(os.environ, {"MCP_TRANSPORT": "http"}):
        assert resolve_transport("") == "http"
        assert resolve_transport("stdio") == "stdio"  # CLI overrides env var


# ============================================================================
# 4. Agent Core HTTP Transport Connection
# ============================================================================

@pytest.mark.asyncio
async def test_agent_http_transport_connection(monkeypatch):
    """Verify BenchmarkRadarAgent in HTTP mode configures the client with Bearer Authorization header."""
    monkeypatch.setenv("MCP_TRANSPORT", "http")
    monkeypatch.setenv("MCP_API_KEY", "test-secret-token")
    monkeypatch.setenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")

    agent = BenchmarkRadarAgent()
    assert agent.transport == "http"
    assert agent.bearer_token == "test-secret-token"

    with patch("src.agent_core.create_mcp_http_client") as mock_create_http, \
         patch("src.agent_core.streamable_http_client") as mock_stream_http:
        mock_stream_http.return_value.__aenter__.return_value = (MagicMock(), MagicMock())
        async with agent._open_server_connection() as streams:
            assert streams is not None
        mock_create_http.assert_called_once_with(
            headers={"Authorization": "Bearer test-secret-token"}
        )
        