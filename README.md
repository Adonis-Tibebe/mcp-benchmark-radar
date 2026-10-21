# AI Research & Benchmark Radar (MCP End-to-End)

> **Use Case:** AI Research & Benchmark Radar is an LLM-powered research assistant for machine learning students and engineers. It queries local ML benchmark baselines stored in a SQLite database, searches live web and GitHub repositories for state-of-the-art improvements, and saves structured literature synthesis briefs to local markdown files via the Model Context Protocol (MCP).

---

## Architecture Overview

```
+-------------------------------------------------------------------------+
|                              User / CLI                                 |
|                       (src/cli.py - Terminal UI)                        |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  BenchmarkRadarAgent (src/agent_core.py)                |
|  - MCP Client Session (stdio subprocess or Streamable HTTP client)      |
|  - Ingests MCP Resource (database-schema) & Prompt (upgrade_workflow)   |
|  - Translates MCP Tool Schemas -> Google GenAI FunctionDeclarations     |
|  - 4-Step Loop: Model Call -> Tool Interception -> Tool Execution ->    |
|    Synthesized Final Answer                                             |
+-------------------+--------------------------------+--------------------+
                    |                                |
       (stdio JSON-RPC subprocess)        (HTTP JSON-RPC with Bearer Auth)
                    |                                |
                    +----------------+---------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  FastMCP Server (src/mcp_server.py)                     |
|  - Strict Zero-stdout in stdio mode (logging strictly to stderr)        |
|  - BearerTokenVerifier & ScopeEnforcementMiddleware                     |
|  - 5 External Tools:                                                    |
|    * query_benchmarks()          -> SQLite DB (data/benchmarks.db)      |
|    * search_ml_web_and_github()  -> DuckDuckGo Search & GitHub REST API |
|    * read_research_brief()       -> Local FS (data/research_briefs/)    |
|    * save_research_brief()       -> Local FS (write-protected scope)    |
|    * list_research_briefs()       -> Local FS (discovery of notes)      |
|  - 1 MCP Resource: radar://database-schema                              |
|  - 1 MCP Prompt:   benchmark_upgrade_workflow                           |
+-------------------------------------------------------------------------+
```

---

## Features & Highlights

- **Full MCP End-to-End Integration:** Custom user-defined MCP client built from scratch (no reliance on pre-built third-party host applications).
- **Dual Transport Protocols:**
  - **`stdio` (Default):** Zero-configuration subprocess transport with zero stdout leakage (`show_banner=False`).
  - **`Streamable HTTP` (Stretch Goal):** Remote HTTP transport (`http://127.0.0.1:8000/mcp`) supporting Bearer token authentication.
- **Role-Based Scoped Access (Stretch Goal):**
  - `read-write` token (`MCP_API_KEY`): Unrestricted access to all tools, including `save_research_brief`.
  - `read-only` token (`MCP_READ_ONLY_API_KEY`): Access to analytical tools (`query_benchmarks`, `search_ml_web_and_github`, `read_research_brief`, `list_research_briefs`), with automatic interception and helpful recovery errors if a write action is attempted.
- **MCP Resource & Prompt Integration:** Automatically injects `radar://database-schema` and `benchmark_upgrade_workflow` as contextual background knowledge.
- **Defensive Engineering & Safety:** Parameterized SQL queries (`?`) prevent SQL injection, path normalization prevents directory traversal (`../`), and timeouts/rate-limit traps handle web search disruptions gracefully.

---

## Quickstart & Setup Instructions

### 1. Prerequisites
- Python 3.10 or higher
- Node.js & npm (optional, only needed for running `@modelcontextprotocol/inspector`)

### 2. Clone and Create Virtual Environment
```bash
git clone https://github.com/<your-username>/mcp-benchmark-radar.git
cd mcp-benchmark-radar

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows (Git Bash):
source .venv/Scripts/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in your Gemini API key:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
MCP_TRANSPORT=stdio
MCP_SERVER_URL=http://127.0.0.1:8000/mcp
MCP_API_KEY=dev-secret-token
MCP_READ_ONLY_API_KEY=dev-read-only-token
```

> **Gemini availability note:** Gemini models may occasionally return `5XX` server
> errors or `429` rate-limit errors. If this happens, update `GEMINI_MODEL` in
> your `.env` file to another available Gemini model and restart the application.

The example HTTP commands and authentication tests use the bearer tokens
defined in `.env.example`: `dev-secret-token` for read-write access and
`dev-read-only-token` for read-only access. To test different tokens, add them
to your local `.env` file and ensure `src/mcp_server.py` loads the corresponding
environment variables through its token configuration before starting the HTTP
server. Do not commit real secrets or place them in source code.

### 5. Seed the Benchmark Database
Generate the SQLite database (`data/benchmarks.db`) and baseline failure notes:
```bash
python data/seed_db.py
```
*Expected Output: `Seeded 8 benchmark records into data/benchmarks.db and populated baseline notes in data/research_briefs/`.*

---

## Running the Application

### Mode A: Standard Terminal CLI (`stdio` Subprocess Mode)

**Interactive Mode:**
```bash
python src/cli.py
```
You will be greeted with the connection banner:
```
========================================================================
       AI Research & Benchmark Radar (MCP + Google Gemini)
========================================================================
------------------------------------------------------------------------
[MCP Connection] Mode:       stdio (local subprocess: mcp_server.py)
------------------------------------------------------------------------

Radar > Find our lowest accuracy benchmark in the DB, search GitHub for modern upgrades, and save a brief called cifar100_upgrade.md
```

**Single Query Execution:**
```bash
python src/cli.py "What datasets are currently tracked in our benchmarks database?"
```

---

### Mode B: Streamable HTTP Transport with Scoped Authentication

#### Step 1: Start the MCP Server in HTTP Mode
In your first terminal:
```bash
python src/mcp_server.py --transport http --port 8000
```
*Output: `Starting Benchmark-Radar FastMCP Server on Streamable HTTP at http://127.0.0.1:8000/mcp ...`*

#### Step 2: Run the CLI in HTTP Mode

**Test with Full Write Permissions (`read-write` token):**
```bash
python src/cli.py --transport http --token dev-secret-token
```
*Displays: `[MCP Connection] Permission: read-write (full access: queries, search, save_research_brief)`*

**Test with Restricted Read-Only Permissions (`read-only` token):**
```bash
python src/cli.py --transport http --token dev-read-only-token
```
*Displays: `[MCP Connection] Permission: read-only (restricted: save_research_brief is blocked)`*
*If you prompt the agent to save a file in read-only mode, the server blocks `save_research_brief` and returns a descriptive error message instructing the LLM to complete its response without saving.*

---

### Mode C: Streamlit Web UI Wrapper (Optional Bonus)

You can launch a clean, interactive browser interface:

```bash
# Launch in default stdio mode:
streamlit run src/web_ui.py

# Launch pointing to an external Streamable HTTP server:
streamlit run src/web_ui.py -- --transport http --token dev-secret-token
```

The web interface features:
- Clean chat interface with structured Markdown rendering for research briefs and tables.
- Collapsible MCP tool execution traces that keep detailed logs neatly out of the main chat flow.
- Sidebar controls to dynamically select transport modes (`stdio` vs. `http`), configure endpoints/tokens, or test with quick-query buttons.

---

### Command-Line Arguments and Defaults

The project has four runnable Python files. The following table explains what
each argument does and what happens when it is omitted.

#### `src/cli.py` — application client

```text
--transport {stdio,http}
--token TOKEN / --bearer-token TOKEN
--url URL
prompt ...
```

- `--transport` selects the client connection. If omitted, it reads
  `MCP_TRANSPORT`; if that is missing, it uses `stdio`.
- `--token` (also `--bearer-token`) supplies the HTTP Bearer token. If omitted,
  it reads `MCP_API_KEY`, or `MCP_READ_ONLY_API_KEY` if the write key is not
  set. It has no effect on stdio.
- `--url` supplies the HTTP MCP endpoint. If omitted, it reads
  `MCP_SERVER_URL`; otherwise it uses `http://127.0.0.1:8000/mcp`.
- `prompt ...` is optional. If supplied, the words are joined into one prompt
  and the program exits after one response. If omitted, the CLI starts its
  interactive `Radar >` loop.

In HTTP mode, the CLI is only a client: it does not start the server process.
Start `src/mcp_server.py` separately first. If the CLI has no token, it
attempts an unauthenticated connection, which the authenticated server rejects.

#### `src/mcp_server.py` — MCP server

```text
--transport {stdio,http}
--host HOST
--port PORT
```

- `--transport` selects the server transport. If omitted, it checks
  `MCP_TRANSPORT`; missing or invalid values fall back to `stdio`.
- `--host` controls the HTTP bind address and is used only in HTTP mode.
  The default is `127.0.0.1`.
- `--port` controls the HTTP listening port and is used only in HTTP mode.
  The default is `8000`.
- The HTTP endpoint path is fixed at `/mcp`, so the default URL is
  `http://127.0.0.1:8000/mcp`.

HTTP mode loads tokens from `.env` and refuses to start unless at least one of
`MCP_API_KEY` or `MCP_READ_ONLY_API_KEY` is configured. Stdio mode starts
without host, port, or token arguments and keeps stdout reserved for MCP
protocol messages.

```bash
# Defaults to stdio.
python src/mcp_server.py

# HTTP on the default host and port.
python src/mcp_server.py --transport http

# HTTP on a custom interface and port.
python src/mcp_server.py --transport http --host 0.0.0.0 --port 9000
```

#### `data/seed_db.py` — local data setup

This script takes no command-line arguments:

```bash
python data/seed_db.py
```

It creates or refreshes `data/benchmarks.db` and writes starter notes to
`data/research_briefs/`. The agent core (`src/agent_core.py`) is an imported
library component, not a standalone command-line program; `src/cli.py` and
`src/web_ui.py` are its application entry points.

#### `src/web_ui.py` — web chat client

```text
streamlit run src/web_ui.py [-- --transport {stdio,http} --token TOKEN --url URL]
```

Accepts optional `--transport`, `--token`, and `--url` arguments passed after `--`.
If omitted, defaults are read from `.env` or adjusted live using the sidebar.

## Testing with MCP Inspector

You can inspect all tools, resources, and prompts interactively via the official `@modelcontextprotocol/inspector`:

### 1. Testing `stdio` Mode
```bash
npx @modelcontextprotocol/inspector python src/mcp_server.py
```

### 2. Testing `Streamable HTTP` Mode with Bearer Authentication
1. Start the server in HTTP mode:
   ```bash
   python src/mcp_server.py --transport http --port 8000
   ```
2. In a second terminal, open the Inspector web UI:
   ```bash
   npx @modelcontextprotocol/inspector
   ```
3. In the Inspector UI:
   - **Transport Type:** Select `Streamable HTTP` (or `HTTP`).
   - **URL:** Enter `http://127.0.0.1:8000/mcp`.
   - **Custom Headers:** Add `Authorization: Bearer dev-secret-token` (or `Bearer dev-read-only-token`).
   - Click **Connect** to inspect tools, call functions, and view returned schemas.

---

## Complete Tool Catalog

The MCP Server exposes **5 tools**, **1 resource**, and **1 prompt**:

### 1. `query_benchmarks`
* **Description:** Query local model benchmarks in SQLite. Allows filtering by dataset name and min/max accuracy bounds.
* **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "dataset": {"type": "string", "description": "Optional dataset name filter"},
      "min_accuracy": {"type": "number", "default": 0.0, "description": "Lower accuracy bound (0.0 to 1.0)"},
      "max_accuracy": {"type": "number", "default": 1.0, "description": "Upper accuracy bound (0.0 to 1.0)"}
    }
  }
  ```
* **Example Call:** `query_benchmarks(dataset="PlantVillage", min_accuracy=0.8)`
* **Example Output:**
  ```markdown
  ### Benchmark Query Results (2 records matched)
  | ID | Dataset | Model Name | Task Type | Accuracy | F1 Score | Params (M) | Notes File | Status |
  |---|---|---|---|---|---|---|---|---|
  | 1 | PlantVillage | MobileNetV2-Baseline | Image Classification | 0.842 | 0.835 | 3.4 | plantvillage_baseline.md | Baseline |
  | 2 | PlantVillage | ResNet50-Augmented | Image Classification | 0.915 | 0.910 | 25.6 | plantvillage_baseline.md | Active |
  ```

---

### 2. `search_ml_web_and_github`
* **Description:** Search the live web (via DuckDuckGo) or public GitHub repositories (via GitHub REST API) for state-of-the-art ML models and architectures.
* **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search keywords or model architecture query"},
      "source": {"type": "string", "enum": ["web", "github"], "default": "web"},
      "max_results": {"type": "integer", "default": 4, "description": "Number of results (1 to 10)"}
    },
    "required": ["query"]
  }
  ```
* **Example Call:** `search_ml_web_and_github(query="Amharic OCR vision transformer", source="github", max_results=3)`
* **Example Output:**
  ```markdown
  ### GitHub Repository Search Results for 'Amharic OCR vision transformer' (3 results)
  1. [repo-owner/ethiopic-trocr](https://github.com/...): Vision Transformer for Ethiopic and Amharic text recognition. (Stars: 142)
  ...
  ```

---

### 3. `read_research_brief`
* **Description:** Read the contents of a local markdown research brief or failure mode note inside `data/research_briefs/`. Protected against directory traversal.
* **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "filename": {"type": "string", "description": "Filename ending with .md (e.g. 'cifar100_experiments.md')"}
    },
    "required": ["filename"]
  }
  ```
* **Example Call:** `read_research_brief(filename="cifar100_experiments.md")`
* **Example Output:** Returns the markdown contents detailing failure modes (e.g., small image resolution, class confusion). If the file is not found, returns a list of available files.

---

### 4. `save_research_brief`
* **Description:** Save structured research briefs and upgrade plans to `data/research_briefs/`. Requires `read-write` scope in HTTP mode.
* **Input Schema:**
  ```json
  {
    "type": "object",
    "properties": {
      "filename": {"type": "string", "description": "Target filename ending in .md"},
      "content": {"type": "string", "description": "Markdown content to persist"}
    },
    "required": ["filename", "content"]
  }
  ```
* **Example Call:** `save_research_brief(filename="cifar100_upgrade_plan.md", content="# Upgrade Plan...")`
* **Example Output:** `Successfully saved research brief to 'cifar100_upgrade_plan.md' (842 characters written in data/research_briefs/).`

---

### 5. `list_research_briefs`
* **Description:** Discover all existing markdown briefs and upgrade plans in `data/research_briefs/`. Enables sessions to locate prior upgrade plans.
* **Input Schema:** `{ "type": "object", "properties": {} }`
* **Example Call:** `list_research_briefs()`
* **Example Output:**
  ```markdown
  Available research briefs in data/research_briefs/:
  - amharic_ocr_notes.md
  - chest_xray_notes.md
  - cifar100_experiments.md
  - fraud_detection_notes.md
  - plantvillage_baseline.md
  - upgrade_plan.md
  ```

---

### MCP Resources & Prompts

* **Resource `radar://database-schema`:** Returns SQLite table schema DDL, available datasets, model counts, and baseline accuracy boundaries.
* **Prompt `benchmark_upgrade_workflow`:** Parameterized workflow template (`dataset_name`) instructing the model step-by-step how to query baselines, inspect failure notes, search external sources, and synthesize an upgrade brief.

---

## Running Automated Tests

Run the complete test suite with `pytest`:
```bash
pytest tests/ -v
```

The test suite uses mocks to test application behavior and MCP interactions.
It does not make real calls to the Google Gemini API, live web search
services, or GitHub while the tests are running. This keeps tests
deterministic, fast, and safe to run without consuming API quotas.

All 25 automated tests verify:
- MCP server tool input validations, parameterization, and path traversal protection (`tests/test_server_tools.py`)
- User-defined MCP client loop, schema translation, and Gemini model interactions (`tests/test_agent_core.py`)
- HTTP transport resolution, Bearer token scope mapping, and middleware access enforcement (`tests/test_http_auth.py`)

---
