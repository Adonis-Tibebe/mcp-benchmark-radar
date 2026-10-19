"""Agent Core: UI-Agnostic MCP Client & Gemini Agent Loop.

Implements the explicit 4-step MCP agent loop:
1. Connects to the MCP server (stdio or HTTP transport).
2. Calls list_tools() to fetch external tool definitions.
3. Translates MCP tool schemas into Google GenAI FunctionDeclarations.
4. Executes model generation, intercepts function_call requests, invokes MCP tools,
   records the tool execution trace, and feeds results back until a final text answer is produced.

Decoupled from CLI and Web UI for modularity and reusability.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Configure logging strictly to sys.stderr
logger = logging.getLogger("agent_core")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)

SYSTEM_INSTRUCTION = """You are the AI Research & Benchmark Radar assistant for machine learning students and engineers.
Your goal is to investigate model benchmark baselines, inspect failure modes, search for state-of-the-art upgrades,
and synthesize comprehensive research briefs.

Always use your tools when requested:
1. Query benchmarks in the SQLite database to inspect baselines.
2. Read research briefs to review detailed failure modes and dataset notes.
3. Search web and GitHub for cutting-edge architectures, papers, and repositories.
4. Save structured upgrade briefs with actionable recommendations to data/research_briefs/.

Be analytical, concise, and structured in your explanations."""


def _extract_root_exception(exc: BaseException) -> BaseException:
    """Recursively extract the deepest root cause from nested ExceptionGroups."""
    current = exc
    while hasattr(current, "exceptions") and getattr(current, "exceptions", None):
        current = current.exceptions[0]
    return current


class BenchmarkRadarAgent:
    """UI-agnostic MCP Client and Gemini reasoning agent."""

    def __init__(
        self,
        server_script_path: Optional[Path] = None,
        model_name: Optional[str] = None,
        env_file_path: Optional[Path] = None,
    ):
        """Initialize the agent with environment configuration."""
        # Load environment variables
        env_path = env_file_path or (PROJECT_ROOT / ".env")
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()  # Fallback to default search

        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        # Fallback to gemini-3.8-flash if model is unset or set to deprecated gemini-2.5-flash
        raw_model = (model_name or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")).strip()
        if raw_model in ("gemini-2.5-flash", "gemini-2.5-flash-lite"):
            logger.info("Auto-updating deprecated model '%s' to 'gemini-3.8-flash'", raw_model)
            self.model_name = "gemini-3.8-flash"
        else:
            self.model_name = raw_model
        self.transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
        self.server_script = server_script_path or (PROJECT_ROOT / "src" / "mcp_server.py")

        # Validate API key presence without logging the secret value
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            logger.warning(
                "GEMINI_API_KEY is not set or contains the default placeholder. "
                "Agent calls to Gemini will fail until a valid key is provided in .env."
            )

        # Initialize Google GenAI client if key is available
        self.genai_client = genai.Client(api_key=self.api_key) if self.api_key else None

    def _convert_mcp_to_genai_tools(self, mcp_tools: List[Any]) -> List[types.Tool]:
        """Convert MCP tool definitions into Google GenAI Tool declarations."""
        declarations: List[types.FunctionDeclaration] = []

        for tool in mcp_tools:
            # Clean up schema for Gemini FunctionDeclaration (support v1 and v2 attribute names).
            # Use isinstance guards: on MagicMock, getattr returns a truthy MagicMock for any attr,
            # so we must verify the value is actually a dict before using it.
            _input_schema = getattr(tool, "input_schema", None)
            _input_schema_alt = getattr(tool, "inputSchema", None)
            if isinstance(_input_schema, dict):
                raw_schema = _input_schema
            elif isinstance(_input_schema_alt, dict):
                raw_schema = _input_schema_alt
            else:
                raw_schema = {}
            cleaned_schema = dict(raw_schema)
            # Remove JSON Schema meta-keywords that may interfere with API
            cleaned_schema.pop("$schema", None)

            # Ensure type is object
            if "type" not in cleaned_schema:
                cleaned_schema["type"] = "object"
            if "properties" not in cleaned_schema:
                cleaned_schema["properties"] = {}

            decl = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or f"Tool {tool.name}",
                parameters_json_schema=cleaned_schema,
            )
            declarations.append(decl)

        return [types.Tool(function_declarations=declarations)]

    async def run(
        self,
        user_prompt: str,
        on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_end: Optional[Callable[[str, str], None]] = None,
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        """Execute the 4-step MCP Agent loop.

        Args:
            user_prompt: User request or research task.
            on_tool_start: Optional callback (tool_name, tool_args) called before running a tool.
            on_tool_end: Optional callback (tool_name, tool_result_str) called after tool finishes.
            max_iterations: Safety threshold to prevent infinite tool-calling loops.

        Returns:
            Dict containing:
                - 'final_answer': str
                - 'tool_trace': list of dicts with 'tool', 'args', 'result'
                - 'error': Optional[str]
        """
        tool_trace: List[Dict[str, Any]] = []

        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            return {
                "final_answer": "",
                "tool_trace": [],
                "error": (
                    "GEMINI_API_KEY is not configured. Please copy .env.example to .env "
                    "and populate your valid Google Gemini API key."
                ),
            }

        # Verify server script exists
        if not self.server_script.exists():
            return {
                "final_answer": "",
                "tool_trace": [],
                "error": f"MCP server script not found at {self.server_script}.",
            }

        # Step 1: Establish MCP Client connection
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_script)],
            env=dict(os.environ),
        )

        try:
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    # Initialize MCP session
                    await session.initialize()

                    # Step 2: List available tools
                    tools_response = await session.list_tools()
                    mcp_tools = tools_response.tools
                    logger.info("Retrieved %d tools from MCP server", len(mcp_tools))

                    # Step 3: Convert schemas and pass to Gemini
                    genai_tools = self._convert_mcp_to_genai_tools(mcp_tools)

                    # Initialize multi-turn contents list
                    contents: List[Any] = [
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=user_prompt)],
                        )
                    ]

                    # Step 4: Iterative Tool Interception & Reasoning Loop
                    iteration = 0
                    while iteration < max_iterations:
                        iteration += 1
                        logger.info("Agent iteration %d querying model %s", iteration, self.model_name)

                        try:
                            response = self.genai_client.models.generate_content(
                                model=self.model_name,
                                contents=contents,
                                config=types.GenerateContentConfig(
                                    tools=genai_tools,
                                    system_instruction=SYSTEM_INSTRUCTION,
                                    temperature=0.2,
                                ),
                            )
                        except Exception as gen_err:
                            root_gen_err = _extract_root_exception(gen_err)
                            err_str = str(root_gen_err)
                            # Handle model-specific quota exhaustion with fallback model
                            if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and self.model_name != "gemini-3.1-flash-lite":
                                logger.warning(
                                    "Model '%s' hit rate limit (429). Retrying iteration %d with fallback model 'gemini-3.1-flash-lite'...",
                                    self.model_name,
                                    iteration,
                                )
                                self.model_name = "gemini-3.1-flash-lite"
                                response = self.genai_client.models.generate_content(
                                    model=self.model_name,
                                    contents=contents,
                                    config=types.GenerateContentConfig(
                                        tools=genai_tools,
                                        system_instruction=SYSTEM_INSTRUCTION,
                                        temperature=0.2,
                                    ),
                                )
                            else:
                                raise root_gen_err

                        # Check if model requested tool execution
                        if not response.function_calls:
                            # Final answer reached
                            final_text = response.text or ""
                            return {
                                "final_answer": final_text,
                                "tool_trace": tool_trace,
                                "error": None,
                            }

                        # Append model's tool call candidate to conversation history
                        candidate_content = response.candidates[0].content
                        contents.append(candidate_content)

                        # Execute all requested function calls
                        tool_response_parts: List[types.Part] = []

                        for call in response.function_calls:
                            tool_name = call.name
                            tool_args = call.args or {}
                            logger.info("Executing MCP tool '%s' with args %s", tool_name, tool_args)

                            if on_tool_start:
                                try:
                                    on_tool_start(tool_name, tool_args)
                                except Exception as cb_err:
                                    logger.warning("on_tool_start callback error: %s", cb_err)

                            # Invoke tool via MCP session
                            try:
                                mcp_result = await session.call_tool(tool_name, tool_args)
                                # Extract text content from MCP response parts
                                result_texts = [
                                    part.text
                                    for part in mcp_result.content
                                    if hasattr(part, "text")
                                ]
                                result_str = "\n".join(result_texts) if result_texts else "Tool completed successfully."
                            except Exception as call_err:
                                logger.error("MCP tool execution error: %s", call_err)
                                result_str = f"Error calling MCP tool '{tool_name}': {str(call_err)}"

                            if on_tool_end:
                                try:
                                    on_tool_end(tool_name, result_str)
                                except Exception as cb_err:
                                    logger.warning("on_tool_end callback error: %s", cb_err)

                            # Record in tool trace
                            tool_trace.append(
                                {
                                    "tool": tool_name,
                                    "args": tool_args,
                                    "result": result_str,
                                }
                            )

                            # Create response part for Gemini
                            part = types.Part.from_function_response(
                                name=tool_name,
                                response={"result": result_str},
                            )
                            tool_response_parts.append(part)

                        # Append tool responses as 'user' role for Gemini API (not 'tool')
                        contents.append(
                            types.Content(
                                role="user",
                                parts=tool_response_parts,
                            )
                        )

                    return {
                        "final_answer": "Loop limit reached without completing synthesis.",
                        "tool_trace": tool_trace,
                        "error": "Exceeded maximum tool calling iterations.",
                    }

        except Exception as conn_err:
            root_err = _extract_root_exception(conn_err)
            err_msg = f"{type(root_err).__name__}: {str(root_err)}"
            logger.error("Agent execution failed: %s", err_msg)
            return {
                "final_answer": "",
                "tool_trace": tool_trace,
                "error": f"Failed to execute agent loop: {err_msg}",
            }
