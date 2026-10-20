"""Command Line Interface (CLI) for AI Research & Benchmark Radar.

Provides an interactive terminal session and single-command execution modes,
connecting to the MCP Server and Gemini agent core via either stdio or HTTP.
Displays live MCP tool calls as they execute.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure stdout and stderr handle UTF-8 symbols cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_core import BenchmarkRadarAgent

BANNER = """
========================================================================
       AI Research & Benchmark Radar (MCP + Google Gemini)
========================================================================
Interactive assistant for ML baseline exploration, live research,
and literature synthesis briefs.
Type your prompt below, or type 'exit' / 'quit' to exit.
------------------------------------------------------------------------
"""


def print_connection_info(agent: BenchmarkRadarAgent) -> None:
    """Print connection details (transport, URL, permissions) for user visibility."""
    print("------------------------------------------------------------------------")
    if agent.transport == "http":
        print(f"[MCP Connection] Mode:       Streamable HTTP")
        print(f"[MCP Connection] Server URL: {agent.server_url}")
        print(f"[MCP Connection] Permission: {agent.permission_scope}")
    else:
        print(f"[MCP Connection] Mode:       stdio (local subprocess: {agent.server_script.name})")
    print("------------------------------------------------------------------------")


def format_tool_start(tool_name: str, args: dict) -> None:
    """Print live tool execution notification with summarized arguments."""
    summarized_args = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 60:
            summarized_args[k] = v[:57].replace("\n", " ") + f"... ({len(v)} chars)"
        else:
            summarized_args[k] = v
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in summarized_args.items())
    print(f"\n[MCP Tool Executing] -> {tool_name}({args_str})")


def format_tool_end(tool_name: str, result: str) -> None:
    """Print live tool result preview."""
    preview = result.strip().split("\n")[0]
    if len(preview) > 100:
        preview = preview[:97] + "..."
    line_count = len(result.strip().split("\n"))
    print(f"[MCP Tool Completed] <- {tool_name}: {line_count} line(s) returned ({preview})")


async def execute_prompt(agent: BenchmarkRadarAgent, prompt: str) -> None:
    """Run a single prompt through the agent and print results."""
    print("\n[Radar Agent] Analyzing request and coordinating MCP tools...")
    result = await agent.run(
        user_prompt=prompt,
        on_tool_start=format_tool_start,
        on_tool_end=format_tool_end,
    )

    if result.get("error"):
        print(f"\n[Error] {result['error']}")
        return

    print("\n" + "=" * 72)
    print("                      RADAR RESEARCH BRIEF")
    print("=" * 72)
    print(result.get("final_answer", "").strip())
    print("=" * 72)

    # Print trace summary
    trace = result.get("tool_trace", [])
    if trace:
        tools_used = " -> ".join([t["tool"] for t in trace])
        print(f"\n[Tool Execution Trace ({len(trace)} tools)]: {tools_used}")
    else:
        print("\n[Tool Execution Trace]: Direct response (no external tools required).")


async def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="AI Research & Benchmark Radar CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default=None,
        help="MCP transport to use ('stdio' or 'http'). Defaults to MCP_TRANSPORT env var or 'stdio'.",
    )
    parser.add_argument(
        "--token",
        "--bearer-token",
        dest="bearer_token",
        default=None,
        help="Bearer token for HTTP authentication. Defaults to MCP_API_KEY from .env.",
    )
    parser.add_argument(
        "--url",
        dest="server_url",
        default=None,
        help="MCP server URL for HTTP transport (default: http://127.0.0.1:8000/mcp).",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional prompt to execute once and exit. If omitted, starts interactive session.",
    )
    args = parser.parse_args()

    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print("\n[Warning] No .env file found in project root.")
        print("Please copy .env.example to .env and add your GEMINI_API_KEY.\n")

    # Initialize agent with CLI arguments (or defaults from environment)
    agent = BenchmarkRadarAgent(
        transport=args.transport,
        bearer_token=args.bearer_token,
        server_url=args.server_url,
    )

    # If prompt arguments provided on command line, execute once and exit
    if args.prompt:
        prompt_text = " ".join(args.prompt).strip()
        print_connection_info(agent)
        await execute_prompt(agent, prompt_text)
        return

    # Interactive Loop
    print(BANNER)
    print_connection_info(agent)
    while True:
        try:
            user_input = input("\nRadar > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nExiting AI Research & Benchmark Radar. Goodbye!")
                break

            await execute_prompt(agent, user_input)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting AI Research & Benchmark Radar.")
            break


if __name__ == "__main__":
    asyncio.run(main())
