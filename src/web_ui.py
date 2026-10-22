"""Lightweight Streamlit Web UI Wrapper for AI Research & Benchmark Radar.

Provides an interactive web chat interface on top of the UI-agnostic
BenchmarkRadarAgent (src/agent_core.py) and Model Context Protocol (MCP) server.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_core import BenchmarkRadarAgent


def parse_cli_args():
    """Extract optional command-line arguments passed to streamlit after '--'."""
    parser = argparse.ArgumentParser(description="Radar Web UI CLI Options")
    parser.add_argument("--transport", choices=["stdio", "http"], default=None)
    parser.add_argument("--token", "--bearer-token", dest="bearer_token", default=None)
    parser.add_argument("--url", dest="server_url", default=None)

    # Only parse arguments that appear after '--' in sys.argv
    cli_args = sys.argv[1:]
    if "--" in cli_args:
        idx = cli_args.index("--")
        cli_args = cli_args[idx + 1 :]
    else:
        cli_args = []

    parsed, _ = parser.parse_known_args(cli_args)
    return parsed


# ============================================================================
# Streamlit Page Setup
# ============================================================================

st.set_page_config(
    page_title="AI Research & Benchmark Radar",
    page_icon="📡",
    layout="wide",
)

cli_options = parse_cli_args()

# Sidebar: Connection & Configuration
st.sidebar.title("📡 Radar Settings")
st.sidebar.markdown("Configure the MCP transport and server connection.")

transport_options = ["stdio", "http"]
default_transport = (
    cli_options.transport or os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
)
if default_transport not in transport_options:
    default_transport = "stdio"

selected_transport = st.sidebar.radio(
    "MCP Transport",
    options=transport_options,
    index=transport_options.index(default_transport),
    help="Select 'stdio' for a local subprocess or 'http' for Streamable HTTP.",
)

if selected_transport == "http":
    default_url = (
        cli_options.server_url
        or os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp").strip()
    )
    server_url = st.sidebar.text_input("MCP Server URL", value=default_url)

    default_token = (
        cli_options.bearer_token
        or os.getenv("MCP_API_KEY", "")
        or os.getenv("MCP_READ_ONLY_API_KEY", "")
    ).strip()
    bearer_token = st.sidebar.text_input(
        "Bearer Token",
        value=default_token,
        type="password",
        help="Use MCP_API_KEY for write access or MCP_READ_ONLY_API_KEY for read-only access.",
    )
else:
    server_url = None
    bearer_token = None

# Initialize Agent
agent = BenchmarkRadarAgent(
    transport=selected_transport,
    bearer_token=bearer_token,
    server_url=server_url,
)

# Display connection & permission badge in sidebar
if selected_transport == "http":
    st.sidebar.info(f"**URL:** `{agent.server_url}`\n\n**Scope:** {agent.permission_scope}")
else:
    st.sidebar.success(f"**Mode:** stdio (subprocess `{agent.server_script.name}`)")

# Sidebar: Quick Action Prompts
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Quick Queries")
quick_queries = [
    "What datasets are currently tracked in our benchmarks database?",
    "Find our lowest accuracy benchmark and summarize its failure notes.",
    "List all available research briefs in data/research_briefs/.",
    "Find the lowest accuracy benchmark, search GitHub for modern upgrades, and save a brief called upgrade_plan.md",
]

quick_prompt = None
for q in quick_queries:
    if st.sidebar.button(q, key=f"quick_{q[:20]}"):
        quick_prompt = q

if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# ============================================================================
# Main Chat Interface
# ============================================================================

st.title("📡 AI Research & Benchmark Radar")
st.caption(
    "LLM-powered assistant for ML baseline exploration, live SOTA research, and literature synthesis briefs via MCP."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tool_trace"):
            with st.expander(
                f"🛠️ MCP Tool Execution Trace ({len(message['tool_trace'])} tools used)",
                expanded=False,
            ):
                for i, step in enumerate(message["tool_trace"], 1):
                    st.markdown(f"**Step {i}: `{step['tool']}`**")
                    if step.get("args"):
                        st.json(step["args"])
                    with st.expander(f"Result Preview: {step['tool']}", expanded=False):
                        st.markdown(step.get("result", ""))

# Prompt input (from user text input or quick prompt button)
user_input = st.chat_input("Ask about ML benchmarks, upgrades, or briefs...") or quick_prompt

if user_input:
    # Append and show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Execute query through BenchmarkRadarAgent
    with st.chat_message("assistant"):
        with st.spinner("Analyzing request and coordinating MCP tools..."):
            result = asyncio.run(agent.run(user_input))

        if result.get("error"):
            st.error(f"Error: {result['error']}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ Error: {result['error']}"}
            )
        else:
            final_text = result.get("final_answer", "").strip()
            tool_trace = result.get("tool_trace", [])

            st.markdown(final_text)

            if tool_trace:
                with st.expander(
                    f"🛠️ MCP Tool Execution Trace ({len(tool_trace)} tools used)",
                    expanded=False,
                ):
                    for i, step in enumerate(tool_trace, 1):
                        st.markdown(f"**Step {i}: `{step['tool']}`**")
                        if step.get("args"):
                            st.json(step["args"])
                        with st.expander(f"Result Preview: {step['tool']}", expanded=False):
                            st.markdown(step.get("result", ""))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_text,
                    "tool_trace": tool_trace,
                }
            )
