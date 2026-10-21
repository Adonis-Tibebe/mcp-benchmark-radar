# Project Summary: AI Research & Benchmark Radar

## 1. What We Built

The **AI Research & Benchmark Radar** is an AI research assistant for machine
learning students and engineers. It helps a user compare local benchmark
results with current research and code, then produces a practical upgrade
brief.

The application uses MCP end to end:

1. A FastMCP server exposes benchmark data, local notes, web search, GitHub
   search, and brief-writing operations.
2. A custom MCP client connects to that server.
3. The client gives the available tools to Google Gemini.
4. Gemini requests tools when needed, and the client runs those requests.
5. Tool results are returned to Gemini so it can produce the final answer.

This directly addresses the assignment: the server, client, and AI application
work together instead of relying on a pre-built host.

## 2. System Diagram

```text
+--------------------------------------------------------------------------+
| 1. Presentation Tier                                                     |
|    - src/cli.py: Interactive REPL & Single-Prompt CLI Runner             |
|    - src/web_ui.py: Streamlit Web Chat Interface                         |
|    - Live tool-execution formatting & Connection permission banner       |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
| 2. Agent Core & Custom MCP Client Tier                                   |
|    - src/agent_core.py: BenchmarkRadarAgent                             |
|    - Dual-transport context manager (_open_server_connection)            |
|    - Dynamic token resolution & permission inspection                    |
|    - Ingestion of MCP Resource (schema) & MCP Prompt (workflow)           |
|    - JSON-Schema to Google GenAI FunctionDeclaration translation           |
|    - Multi-turn tool execution loop with model-level fallback             |
+-------------------+----------------------------------+-------------------+
                    |                                  |
         (stdio JSON-RPC Subprocess)       (Streamable HTTP + Bearer token)
                    |                                  |
                    +-----------------+----------------+
                                      |
                                      v
+--------------------------------------------------------------------------+
| 3. FastMCP Server Tier                                                   |
|    - src/mcp_server.py: FastMCP("Benchmark-Radar-Server")                |
|    - Zero-stdout enforcement on stdio (show_banner=False, stderr logs)   |
|    - BearerTokenVerifier: maps tokens to 'read-write' or 'read-only'     |
|    - ScopeEnforcementMiddleware: blocks write tools for read-only tokens  |
|    - Resource: radar://database-schema                                   |
|    - Prompt:   benchmark_upgrade_workflow                                |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
| 4. External Resources Tier                                               |
|    - SQLite Database (data/benchmarks.db) -> query_benchmarks()          |
|    - Web Search (DuckDuckGo Search)       -> search_ml_web_and_github()  |
|    - GitHub REST API (httpx)              -> search_ml_web_and_github()  |
|    - Filesystem (data/research_briefs/)   -> read_research_brief()       |
|                                           -> save_research_brief()        |
|                                           -> list_research_briefs()      |
+--------------------------------------------------------------------------+
```

## 3. How the Project Meets the Task

### Use case

The assistant is for users who have a local ML benchmark and want to
understand its weaknesses, find modern approaches that may improve it, and
save a clear upgrade plan.

### External resources

The server uses several external or persistent resources:

- **SQLite:** stores datasets, model names, accuracy, F1 score, parameter
  counts, and links to research notes.
- **Local Markdown files:** store baseline failure notes and generated briefs.
- **DuckDuckGo:** provides live web research results.
- **GitHub REST API:** finds public repositories and implementations.

### MCP tools

The server exposes five tools with clear names, descriptions, typed inputs,
validation, and useful error results:

| Tool | Purpose |
|---|---|
| `query_benchmarks` | Find local baseline records using dataset and accuracy filters. |
| `list_research_briefs` | Discover available local notes and previous plans. |
| `read_research_brief` | Read a selected Markdown note safely. |
| `search_ml_web_and_github` | Search current web or GitHub research and implementations. |
| `save_research_brief` | Save a generated plan; HTTP mode restricts this to write scope. |

The optional MCP primitives are also included:

- Resource `radar://database-schema`: supplies the database structure and
  dataset summary.
- Prompt `benchmark_upgrade_workflow`: supplies a reusable upgrade
  investigation workflow.

### Client and agent loop

`src/agent_core.py` contains the user-written MCP client. Its flow is:

1. Connect to the server through stdio or Streamable HTTP.
2. Initialize the MCP session and call `list_tools()`.
3. Convert the tool schemas into Gemini function declarations.
4. Read the database-schema resource and upgrade-workflow prompt.
5. Send the user request, context, and tools to Gemini.
6. Execute any tool calls requested by Gemini.
7. Return each tool result to Gemini until Gemini gives the final answer.

This demonstrates the required working integration: the application calls the
server's tools through an MCP client and uses the results in a real response.

## 4. Main Components

### `src/cli.py`

Provides interactive and one-shot use. With default settings it connects over
stdio and starts the MCP server as a local subprocess. With
`--transport http`, it connects to an already-running HTTP server; that server
must be started separately.

### `src/web_ui.py`

Provides an optional browser-based chat interface using Streamlit. It imports
the exact same `BenchmarkRadarAgent` as the CLI, renders synthesized research
briefs in structured Markdown, and groups MCP tool execution steps in
collapsible accordions to keep the conversation clean.

### `src/agent_core.py`

Contains the custom client and the multi-turn Gemini agent loop. It keeps tool
execution separate from the terminal UI and reports progress through
callbacks.

### `src/mcp_server.py`

Defines the FastMCP server, five tools, one resource, and one prompt. In stdio
mode it never writes logs to stdout, which keeps the MCP protocol stream valid.
In HTTP mode it supports bearer-token authentication and read-only/read-write
scopes.

### `data/seed_db.py`

Creates the deterministic benchmark database and starter research notes needed
for a demonstration.

## 5. Safety and Reliability

- SQL values are passed as parameters rather than concatenated into queries.
- File paths are checked so notes cannot escape the research briefs directory.
- API keys are read from environment variables and are not hard-coded.
- Tokens are not printed in logs; the CLI reports only the permission scope.
- HTTP mode refuses to start without at least one configured token.
- Stdio mode keeps stdout reserved for MCP protocol messages.
- Missing files, denied permissions, and search failures return useful messages
  so the model can recover or explain the problem.

## 6. Testing and Demonstration

The MCP Inspector can inspect the server before the client is connected:

```bash
npx @modelcontextprotocol/inspector python src/mcp_server.py
```

The automated tests cover tool validation, SQL and path safety, the custom
client loop, HTTP transport resolution, authentication, and scope enforcement.

A representative demonstration asks the assistant to find the lowest-accuracy
baseline, read its failure notes, search for modern solutions, and save an
upgrade brief. This uses at least two tools in one task and shows the complete
server-client-LLM workflow required by the assignment.
