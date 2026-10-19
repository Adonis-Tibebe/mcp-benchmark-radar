"""Core MCP Server: AI Research & Benchmark Radar.

Exposes external resources (SQLite DB, Web/GitHub Search, Local Research Briefs)
via the Model Context Protocol (MCP).

CRITICAL CONSTRAINTS:
- ZERO stdout: All logging goes strictly to sys.stderr. No print() statements.
- Parameterized SQL (?) only.
- Strict path traversal protection inside data/research_briefs/.
- Safe error recovery strings returned to LLM on invalid inputs or missing data.
"""

import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from duckduckgo_search import DDGS

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from fastmcp import FastMCP

# Configure logging strictly to sys.stderr (NEVER stdout in stdio mode)
logger = logging.getLogger("mcp_server")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "benchmarks.db"
BRIEFS_DIR = DATA_DIR / "research_briefs"

# Initialize FastMCP Server
mcp = FastMCP("Benchmark-Radar-Server")


# ============================================================================
# Helpers
# ============================================================================

def _resolve_safe_brief_path(filename: str) -> Path:
    """Resolve filename inside BRIEFS_DIR and verify it does not escape via path traversal.

    Raises PermissionError if path traversal is attempted.
    """
    clean_name = filename.strip().replace("\\", "/")
    candidate = (BRIEFS_DIR / clean_name).resolve()
    base_resolved = BRIEFS_DIR.resolve()

    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        raise PermissionError(
            f"Path traversal detected: '{filename}' attempts to access paths outside data/research_briefs/."
        )

    if candidate == base_resolved:
        raise PermissionError(
            f"Invalid target: '{filename}' resolves to the directory itself, not a file."
        )

    return candidate


# ============================================================================
# Tool 1: SQLite Benchmark Database Query
# ============================================================================

@mcp.tool()
def query_benchmarks(
    dataset: str = "",
    min_accuracy: float = 0.0,
    max_accuracy: float = 1.0,
) -> str:
    """Query local ML benchmark results from SQLite database (data/benchmarks.db).

    Use this tool whenever you need baseline models, benchmark metrics (accuracy, F1),
    parameter counts, failure notes file references, or model status. Supports filtering
    by dataset name and accuracy range.

    Args:
        dataset: Optional dataset name (e.g., 'PlantVillage', 'Amharic-OCR', 'CIFAR-100',
                 'CreditCard-Fraud', 'Medical-ChestXRay'). Partial or case-insensitive matches accepted.
        min_accuracy: Minimum accuracy filter (between 0.0 and 1.0). Default is 0.0.
        max_accuracy: Maximum accuracy filter (between 0.0 and 1.0). Default is 1.0.

    Returns:
        A markdown-formatted table of matching benchmarks, or an informative error/recovery string.
    """
    logger.info(
        "query_benchmarks called: dataset='%s', min_acc=%s, max_acc=%s",
        dataset,
        min_accuracy,
        max_accuracy,
    )

    # Validate accuracy bounds
    if not (0.0 <= min_accuracy <= 1.0):
        return f"Error: min_accuracy must be between 0.0 and 1.0. Received {min_accuracy}."
    if not (0.0 <= max_accuracy <= 1.0):
        return f"Error: max_accuracy must be between 0.0 and 1.0. Received {max_accuracy}."
    if min_accuracy > max_accuracy:
        return f"Error: min_accuracy ({min_accuracy}) cannot be greater than max_accuracy ({max_accuracy})."

    # Check database file existence
    if not DB_PATH.exists():
        return (
            f"Error: Database file not found at '{DB_PATH}'. "
            "Please run 'python data/seed_db.py' to initialize the benchmark database."
        )

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Parameterized query with ? placeholders
        query_sql = """
            SELECT id, dataset, model_name, task_type, accuracy, f1_score, params_millions, notes_file, status
            FROM model_benchmarks
            WHERE accuracy >= ? AND accuracy <= ?
        """
        params: List[Any] = [min_accuracy, max_accuracy]

        if dataset.strip():
            query_sql += " AND LOWER(dataset) LIKE LOWER(?)"
            params.append(f"%{dataset.strip()}%")

        query_sql += " ORDER BY accuracy ASC"
        cursor.execute(query_sql, params)
        rows = cursor.fetchall()

        if not rows:
            # Provide available datasets for graceful LLM self-correction
            cursor.execute("SELECT DISTINCT dataset FROM model_benchmarks ORDER BY dataset ASC;")
            available = [r[0] for r in cursor.fetchall()]
            conn.close()
            return (
                f"No benchmarks found matching criteria (dataset='{dataset}', "
                f"accuracy range=[{min_accuracy}, {max_accuracy}]). "
                f"Available datasets in database: {', '.join(available)}."
            )

        conn.close()

        # Format markdown table
        lines = [
            f"### Benchmark Query Results ({len(rows)} record{'s' if len(rows) != 1 else ''} found)",
            "",
            "| ID | Dataset | Model Name | Task Type | Accuracy | F1 Score | Params (M) | Notes File | Status |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            r_id, r_data, r_model, r_task, r_acc, r_f1, r_params, r_notes, r_status = row
            lines.append(
                f"| {r_id} | {r_data} | {r_model} | {r_task} | {r_acc:.3f} | {r_f1:.3f} | {r_params:.1f} | `{r_notes}` | {r_status} |"
            )

        return "\n".join(lines)

    except Exception as exc:
        logger.error("Database query failed: %s", exc)
        return f"Error executing database query: {str(exc)}"


# ============================================================================
# Tool 2: Web & GitHub Search
# ============================================================================

@mcp.tool()
def search_ml_web_and_github(
    query: str,
    source: str = "web",
    max_results: int = 4,
) -> str:
    """Search live web (DuckDuckGo) or public GitHub repositories for ML papers, models, and code.

    Use this tool when researching improvements for a benchmark baseline, finding state-of-the-art
    architectures, discovering GitHub repositories, or checking recent literature.

    Args:
        query: Search keywords (e.g., 'PlantVillage vision transformer state of the art',
               'Amharic OCR deep learning github').
        source: Search engine to use: 'web' (DuckDuckGo live web) or 'github' (GitHub REST API).
                Defaults to 'web'.
        max_results: Number of search results to return (integer between 1 and 10). Defaults to 4.

    Returns:
        Structured search summary containing titles, URLs, and snippets/descriptions,
        or an informative error/fallback string.
    """
    logger.info(
        "search_ml_web_and_github called: query='%s', source='%s', max_results=%s",
        query,
        source,
        max_results,
    )

    clean_query = query.strip()
    if not clean_query:
        return "Error: Search query cannot be empty. Please provide relevant ML keywords, model names, or topics."

    clean_source = source.strip().lower()
    if clean_source not in ("web", "github"):
        return f"Error: Invalid search source '{source}'. Supported sources are 'web' and 'github'."

    if not (1 <= max_results <= 10):
        return f"Error: max_results must be between 1 and 10. Received {max_results}."

    # 1. Web Search via DuckDuckGo
    if clean_source == "web":
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(clean_query, max_results=max_results))

            if not raw_results:
                return f"No web search results found for query: '{clean_query}'. Try broader keywords or switch source to 'github'."

            lines = [f"### Web Search Results for '{clean_query}' ({len(raw_results)} results):", ""]
            for idx, item in enumerate(raw_results, start=1):
                title = item.get("title", "Untitled")
                url = item.get("href") or item.get("link") or item.get("url", "")
                snippet = item.get("body") or item.get("snippet", "No summary available.")
                lines.append(f"**{idx}. [{title}]({url})**")
                lines.append(f"> {snippet}")
                lines.append("")

            return "\n".join(lines).strip()

        except Exception as exc:
            logger.warning("DuckDuckGo search error: %s", exc)
            return (
                f"Error performing web search: {str(exc)}. "
                "You can retry with a different query or switch source to 'github'."
            )

    # 2. GitHub Search via Public REST API
    else:
        try:
            url = "https://api.github.com/search/repositories"
            headers = {
                "Accept": "application/vnd.github+json",
                "User-Agent": "MCP-Benchmark-Radar/1.0",
            }
            params = {
                "q": clean_query,
                "sort": "stars",
                "order": "desc",
                "per_page": max_results,
            }

            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=headers, params=params)

            if response.status_code == 403:
                return (
                    "Error: GitHub API rate limit reached (HTTP 403). "
                    "Please switch source to 'web' for DuckDuckGo search or retry shortly."
                )
            if response.status_code != 200:
                return (
                    f"Error querying GitHub API (HTTP {response.status_code}): "
                    f"{response.text[:200]}. You may retry with source='web'."
                )

            data = response.json()
            items = data.get("items", [])
            if not items:
                return (
                    f"No GitHub repositories found matching query: '{clean_query}'. "
                    "Try using broader keywords or source='web'."
                )

            lines = [f"### GitHub Repositories for '{clean_query}' ({len(items)} results):", ""]
            for idx, repo in enumerate(items, start=1):
                full_name = repo.get("full_name", "Unknown Repo")
                html_url = repo.get("html_url", "")
                stars = repo.get("stargazers_count", 0)
                description = repo.get("description") or "No description provided."
                language = repo.get("language") or "N/A"
                lines.append(f"**{idx}. [{full_name}]({html_url})** — ⭐ {stars:,} stars | Language: {language}")
                lines.append(f"> {description}")
                lines.append("")

            return "\n".join(lines).strip()

        except httpx.TimeoutException:
            logger.warning("GitHub request timed out for query: %s", clean_query)
            return "Error: GitHub search timed out after 10 seconds. Try source='web' or check network connectivity."
        except Exception as exc:
            logger.error("GitHub search error: %s", exc)
            return f"Error querying GitHub API: {str(exc)}. Try source='web' as an alternative."


# ============================================================================
# Tool 3: Read Local Research Brief
# ============================================================================

@mcp.tool()
def read_research_brief(filename: str) -> str:
    """Read the content of a local research brief markdown file from data/research_briefs/.

    Use this tool to inspect failure mode notes, previous experiment logs, and baseline
    observations associated with benchmark datasets.

    Args:
        filename: Name of the markdown file (e.g., 'plantvillage_baseline.md',
                  'amharic_ocr_notes.md', 'cifar100_experiments.md').

    Returns:
        The text content of the markdown file, or an informative recovery error listing available files.
    """
    logger.info("read_research_brief called: filename='%s'", filename)

    if not filename.strip():
        return "Error: filename cannot be empty. Please specify a file name inside data/research_briefs/."

    try:
        target_path = _resolve_safe_brief_path(filename)
    except PermissionError as pe:
        logger.warning("Path traversal attempt in read_research_brief: %s", pe)
        return f"Error: Access denied. Path traversal detected. '{filename}' must remain strictly inside data/research_briefs/."

    if not target_path.exists() or not target_path.is_file():
        # Discover existing .md files for LLM recovery
        available_files = []
        if BRIEFS_DIR.exists():
            available_files = sorted([p.name for p in BRIEFS_DIR.glob("*.md")])
        available_str = ", ".join(available_files) if available_files else "None"
        return f"Error: File '{filename}' not found. Available research briefs: {available_str}."

    try:
        content = target_path.read_text(encoding="utf-8")
        return content
    except Exception as exc:
        logger.error("Failed to read file '%s': %s", target_path, exc)
        return f"Error reading file '{filename}': {str(exc)}"


# ============================================================================
# Tool 4: Save Local Research Brief
# ============================================================================

@mcp.tool()
def save_research_brief(filename: str, content: str) -> str:
    """Save structured synthesis notes, upgrade plans, or research briefs to data/research_briefs/.

    Use this tool after synthesizing baseline metrics and external SOTA research to persist
    recommendations, experiment plans, or failure mode reviews to a markdown file.

    Args:
        filename: Target markdown filename with .md extension (e.g., 'plantvillage_upgrade_plan.md').
                  Must remain strictly inside data/research_briefs/.
        content: The complete markdown content to write to the file.

    Returns:
        A success confirmation string with character count, or an informative error message.
    """
    logger.info("save_research_brief called: filename='%s'", filename)

    clean_name = filename.strip()
    if not clean_name:
        return "Error: filename cannot be empty. Please provide a valid file name ending with .md."

    if not clean_name.lower().endswith(".md"):
        return f"Error: filename must have a '.md' extension (e.g., 'upgrade_plan.md'). Received: '{filename}'."

    if not content.strip():
        return "Error: Content cannot be empty. Please provide markdown text for the research brief."

    try:
        target_path = _resolve_safe_brief_path(clean_name)
    except PermissionError as pe:
        logger.warning("Path traversal attempt in save_research_brief: %s", pe)
        return f"Error: Access denied. Path traversal detected. '{filename}' must remain strictly inside data/research_briefs/."

    try:
        BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
        logger.info("Successfully wrote %d characters to %s", len(content), target_path.name)
        return (
            f"Successfully saved research brief to '{target_path.name}' "
            f"({len(content)} characters written in data/research_briefs/)."
        )
    except Exception as exc:
        logger.error("Failed to write research brief '%s': %s", target_path, exc)
        return f"Error saving file '{filename}': {str(exc)}"


# ============================================================================
# Stretch Goal Primitives: MCP Resource & MCP Prompt
# ============================================================================

@mcp.resource("radar://database-schema")
def get_database_schema() -> str:
    """MCP Resource exposing SQLite schema, table structure, and available datasets."""
    logger.info("Resource radar://database-schema requested")
    if not DB_PATH.exists():
        return (
            "# Database Schema\n\n"
            "Error: Database file data/benchmarks.db does not exist yet. Run data/seed_db.py."
        )

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Fetch CREATE TABLE statement
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='model_benchmarks';")
        schema_row = cursor.fetchone()
        schema_sql = schema_row[0] if schema_row else "Table 'model_benchmarks' not found."

        # Fetch summary of datasets
        cursor.execute(
            """
            SELECT dataset, COUNT(*) as model_count, MIN(accuracy) as min_acc, MAX(accuracy) as max_acc
            FROM model_benchmarks
            GROUP BY dataset
            ORDER BY dataset ASC;
            """
        )
        dataset_stats = cursor.fetchall()
        conn.close()

        lines = [
            "# SQLite Database Schema: `model_benchmarks`",
            "",
            "```sql",
            schema_sql,
            "```",
            "",
            "## Available Datasets & Baseline Performance Ranges:",
            "",
            "| Dataset | Model Count | Min Accuracy | Max Accuracy |",
            "|---|---|---|---|",
        ]
        for ds, count, min_a, max_a in dataset_stats:
            lines.append(f"| {ds} | {count} | {min_a:.3f} | {max_a:.3f} |")

        return "\n".join(lines)

    except Exception as exc:
        logger.error("Failed to generate database schema resource: %s", exc)
        return f"Error reading database schema: {str(exc)}"


@mcp.prompt("benchmark_upgrade_workflow")
def benchmark_upgrade_workflow(dataset_name: str = "") -> str:
    """MCP Prompt providing a structured workflow for comparing local baselines with web SOTA."""
    target_clause = (
        f"for the '{dataset_name}' dataset"
        if dataset_name.strip()
        else "for the benchmark baseline with the lowest accuracy in the database"
    )
    safe_slug = dataset_name.strip().lower().replace(" ", "_").replace("-", "_") if dataset_name.strip() else "upgrade"
    suggested_filename = f"{safe_slug}_plan.md"

    return (
        f"You are the AI Research & Benchmark Radar assistant. Execute the following upgrade analysis {target_clause}:\n\n"
        f"Step 1 [Query Database]: Use `query_benchmarks` to retrieve the current baseline accuracy, F1 score, and associated `notes_file`.\n"
        f"Step 2 [Read Local Notes]: Use `read_research_brief` with the retrieved notes file to understand the specific failure modes (e.g. lighting, resolution, class imbalance).\n"
        f"Step 3 [Search SOTA Literature & Code]: Use `search_ml_web_and_github` (with source='web' and source='github') to discover modern architectures or techniques that solve those failure modes.\n"
        f"Step 4 [Synthesize Upgrade Plan]: Draft a structured literature brief contrasting the baseline model with the proposed modern architecture.\n"
        f"Step 5 [Save Brief]: Use `save_research_brief` to save your structured plan as '{suggested_filename}'.\n"
        f"Step 6 [Final Answer]: Present a concise final summary of your findings and confirm the brief was saved."
    )


# ============================================================================
# Main Entry Point (stdio transport by default)
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Benchmark-Radar FastMCP Server on stdio transport...")
    mcp.run(transport="stdio", show_banner=False)
