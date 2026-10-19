"""Unit tests for BenchmarkRadarAgent and MCP agent loop logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from src.agent_core import BenchmarkRadarAgent


def test_agent_missing_api_key(monkeypatch):
    """Verify agent reports missing API key gracefully without crashing."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    agent = BenchmarkRadarAgent()
    agent.api_key = ""

    import asyncio

    result = asyncio.run(agent.run("Find lowest accuracy benchmark"))
    assert result.get("error") is not None
    assert "GEMINI_API_KEY is not configured" in result["error"]


def test_schema_conversion():
    """Verify MCP tool schema is correctly translated to GenAI FunctionDeclaration."""
    agent = BenchmarkRadarAgent()

    # Mock an MCP tool definition
    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "query_benchmarks"
    mock_mcp_tool.description = "Query local benchmarks"
    mock_mcp_tool.inputSchema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "dataset": {"type": "string", "description": "Dataset name"},
            "min_accuracy": {"type": "number", "description": "Min accuracy"},
        },
        "required": [],
    }

    genai_tools = agent._convert_mcp_to_genai_tools([mock_mcp_tool])
    assert len(genai_tools) == 1
    assert len(genai_tools[0].function_declarations) == 1

    decl = genai_tools[0].function_declarations[0]
    assert decl.name == "query_benchmarks"
    assert decl.description == "Query local benchmarks"
    # Verify $schema was removed for clean API consumption
    assert "$schema" not in decl.parameters_json_schema
    assert "dataset" in decl.parameters_json_schema["properties"]


@pytest.mark.asyncio
async def test_agent_loop_mocked(monkeypatch):
    """Verify full 4-step agent loop intercepts function call, invokes MCP tool, and returns final answer."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-test-key")
    agent = BenchmarkRadarAgent()
    agent.api_key = "dummy-test-key"
    agent.genai_client = MagicMock()

    # Step 1: First model response requests a tool call
    mock_call = MagicMock()
    mock_call.name = "query_benchmarks"
    mock_call.args = {"dataset": "PlantVillage"}

    resp1 = MagicMock()
    resp1.function_calls = [mock_call]
    resp1.candidates = [MagicMock(content=types.Content(role="model", parts=[]))]

    # Step 2: Second model response produces final text answer
    resp2 = MagicMock()
    resp2.function_calls = None
    resp2.text = "The baseline accuracy for PlantVillage MobileNetV2 is 0.842."

    agent.genai_client.models.generate_content.side_effect = [resp1, resp2]

    # Mock MCP Client Session
    mock_mcp_tool = MagicMock()
    mock_mcp_tool.name = "query_benchmarks"
    mock_mcp_tool.description = "Query benchmarks"
    mock_mcp_tool.inputSchema = {"type": "object", "properties": {}}

    mock_session = AsyncMock()
    mock_session.initialize.return_value = None
    mock_session.list_tools.return_value = MagicMock(tools=[mock_mcp_tool])

    mock_tool_result = MagicMock()
    mock_tool_part = MagicMock()
    mock_tool_part.text = "| ID | Dataset | Model |\n| 1 | PlantVillage | MobileNetV2 |"
    mock_tool_result.content = [mock_tool_part]
    mock_session.call_tool.return_value = mock_tool_result

    # Mock stdio_client context manager
    with patch("src.agent_core.stdio_client") as mock_stdio, patch(
        "src.agent_core.ClientSession"
    ) as mock_session_cls:
        mock_stdio.return_value.__aenter__.return_value = (MagicMock(), MagicMock())
        mock_session_cls.return_value.__aenter__.return_value = mock_session

        result = await agent.run("What is our baseline for PlantVillage?")

        assert result.get("error") is None
        assert "MobileNetV2 is 0.842" in result["final_answer"]
        assert len(result["tool_trace"]) == 1
        assert result["tool_trace"][0]["tool"] == "query_benchmarks"
    