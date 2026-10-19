"""Command Line Interface (CLI) for AI Research & Benchmark Radar.

Provides an interactive terminal session and single-command execution modes,
connecting to the MCP Server and Gemini agent core.
Displays live MCP tool calls as they execute.
"""

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
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print("\n[Warning] No .env file found in project root.")
        print("Please copy .env.example to .env and add your GEMINI_API_KEY.\n")

    agent = BenchmarkRadarAgent()

    # If arguments provided on command line, execute once and exit
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:]).strip()
        await execute_prompt(agent, prompt)
        return

    # Interactive Loop
    print(BANNER)
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
