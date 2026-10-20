"""Unit tests for Core MCP Server Tools, Resources, Prompts, and Safety Controls.

Verifies:
- Parameterized SQLite querying, dataset filtering, and accuracy bounds validation.
- Path traversal protection for both read_research_brief and save_research_brief.
- Graceful error recovery strings for LLM self-correction.
- Web & GitHub search validation and fallback behaviors.
- MCP resource schema inspection and upgrade prompt generation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.mcp_server import (
    BRIEFS_DIR,
    DB_PATH,
    benchmark_upgrade_workflow,
    get_database_schema,
    query_benchmarks,
    list_research_briefs,
    read_research_brief,
    save_research_brief,
    search_ml_web_and_github,
)


# ============================================================================
# Tool 1 Tests: query_benchmarks
# ============================================================================

def test_query_benchmarks_all():
    """Verify default query returns all 8 seeded baseline models."""
    result = query_benchmarks()
    assert "Benchmark Query Results" in result
    assert "PlantVillage" in result
    assert "Amharic-OCR" in result
    assert "CIFAR-100" in result
    assert "CreditCard-Fraud" in result
    assert "Medical-ChestXRay" in result


def test_query_benchmarks_filtered_dataset():
    """Verify filtering by dataset returns only relevant records."""
    result = query_benchmarks(dataset="PlantVillage")
    assert "MobileNetV2-Baseline" in result
    assert "ResNet50-Augmented" in result
    assert "Amharic-OCR" not in result
    assert "CreditCard-Fraud" not in result


def test_query_benchmarks_filtered_accuracy():
    """Verify filtering by accuracy range returns matching subset."""
    # Only records between 0.90 and 1.0 (ResNet50: 0.915, XGBoost: 0.981)
    result = query_benchmarks(min_accuracy=0.90, max_accuracy=1.0)
    assert "ResNet50-Augmented" in result
    assert "XGBoost-Default" in result
    assert "SimpleCNN-4Layer" not in result  # accuracy is 0.625


def test_query_benchmarks_invalid_bounds():
    """Verify negative or out-of-range bounds return clear recovery errors."""
    res_neg = query_benchmarks(min_accuracy=-0.2)
    assert "Error: min_accuracy must be between 0.0 and 1.0" in res_neg

    res_high = query_benchmarks(max_accuracy=1.5)
    assert "Error: max_accuracy must be between 0.0 and 1.0" in res_high

    res_inverted = query_benchmarks(min_accuracy=0.8, max_accuracy=0.4)
    assert "Error: min_accuracy (0.8) cannot be greater than max_accuracy (0.4)" in res_inverted


def test_query_benchmarks_no_match_recovery():
    """Verify querying a nonexistent dataset returns available datasets for recovery."""
    result = query_benchmarks(dataset="Quantum-NLP")
    assert "No benchmarks found matching criteria" in result
    assert "Available datasets in database:" in result
    assert "PlantVillage" in result
    assert "Amharic-OCR" in result


# ============================================================================
# Tool 2 Tests: search_ml_web_and_github
# ============================================================================

def test_search_input_validation():
    """Verify query validation, source restrictions, and max_results limits."""
    # Empty query
    assert "Error: Search query cannot be empty" in search_ml_web_and_github("")
    assert "Error: Search query cannot be empty" in search_ml_web_and_github("   ")

    # Invalid source
    res_source = search_ml_web_and_github("transformers", source="bing")
    assert "Error: Invalid search source 'bing'" in res_source

    # Out of range max_results
    assert "Error: max_results must be between 1 and 10" in search_ml_web_and_github("vit", max_results=0)
    assert "Error: max_results must be between 1 and 10" in search_ml_web_and_github("vit", max_results=20)


@patch("src.mcp_server.DDGS")
def test_search_web_mocked(mock_ddgs_cls):
    """Verify web search formats results as structured markdown links and snippets."""
    mock_instance = MagicMock()
    mock_instance.text.return_value = [
        {
            "title": "Swin Transformer: Hierarchical Vision Transformer",
            "href": "https://arxiv.org/abs/2103.14030",
            "body": "State of the art visual representation using shifted windows.",
        }
    ]
    mock_ddgs_cls.return_value.__enter__.return_value = mock_instance

    result = search_ml_web_and_github("Swin Transformer", source="web", max_results=1)
    assert "Web Search Results for 'Swin Transformer'" in result
    assert "[Swin Transformer: Hierarchical Vision Transformer](https://arxiv.org/abs/2103.14030)" in result
    assert "State of the art visual representation" in result


@patch("src.mcp_server.httpx.Client")
def test_search_github_mocked(mock_client_cls):
    """Verify GitHub search parses repository metadata correctly."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "items": [
            {
                "full_name": "microsoft/Swin-Transformer",
                "html_url": "https://github.com/microsoft/Swin-Transformer",
                "stargazers_count": 13500,
                "description": "Official PyTorch implementation of Swin Transformer",
                "language": "Python",
            }
        ]
    }
    mock_client.get.return_value = mock_response
    mock_client_cls.return_value.__enter__.return_value = mock_client

    result = search_ml_web_and_github("Swin-Transformer", source="github", max_results=1)
    assert "GitHub Repositories for 'Swin-Transformer'" in result
    assert "[microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)" in result
    assert "13,500 stars" in result
    assert "Language: Python" in result


@patch("src.mcp_server.httpx.Client")
def test_search_github_rate_limited(mock_client_cls):
    """Verify GitHub 403 rate limit returns fallback advice for LLM self-correction."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_client.get.return_value = mock_response
    mock_client_cls.return_value.__enter__.return_value = mock_client

    result = search_ml_web_and_github("OCR", source="github", max_results=2)
    assert "Error: GitHub API rate limit reached" in result
    assert "switch source to 'web'" in result


# ============================================================================
# Tool 3 & 4 Tests: Research Briefs & Path Traversal Protection
# ============================================================================

def test_read_research_brief_valid():
    """Verify reading an existing research brief returns its content."""
    content = read_research_brief("plantvillage_baseline.md")
    assert "PlantVillage Diagnostic Baselines" in content
    assert "MobileNetV2" in content


def test_read_research_brief_missing_recovery():
    """Verify reading a missing brief returns available files for LLM recovery."""
    result = read_research_brief("nonexistent_notes.md")
    assert "Error: File 'nonexistent_notes.md' not found" in result
    assert "Available research briefs:" in result
    assert "plantvillage_baseline.md" in result


def test_read_research_brief_path_traversal():
    """Verify path traversal attacks are blocked with descriptive error strings."""
    res_parent = read_research_brief("../seed_db.py")
    assert "Error: Access denied. Path traversal detected" in res_parent

    res_root = read_research_brief("../../requirements.txt")
    assert "Error: Access denied. Path traversal detected" in res_root

    res_win = read_research_brief("..\\..\\windows\\system32\\cmd.exe")
    assert "Error: Access denied. Path traversal detected" in res_win


def test_save_research_brief_valid(tmp_path):
    """Verify saving a valid markdown brief writes content and returns confirmation."""
    filename = "test_sprint2_upgrade.md"
    content = "# Sprint 2 Test Upgrade\n\nTesting save_research_brief functionality."

    result = save_research_brief(filename, content)
    assert "Successfully saved research brief" in result
    assert filename in result

    # Verify file was actually created on disk inside BRIEFS_DIR
    target = BRIEFS_DIR / filename
    assert target.exists()
    assert target.read_text(encoding="utf-8") == content

    # Cleanup test artifact
    target.unlink()


def test_save_research_brief_validation():
    """Verify filename extension and content validation."""
    # Must end in .md
    res_ext = save_research_brief("notes.txt", "Some content")
    assert "Error: filename must have a '.md' extension" in res_ext

    # Content cannot be empty
    res_empty = save_research_brief("empty_plan.md", "   ")
    assert "Error: Content cannot be empty" in res_empty


def test_save_research_brief_path_traversal():
    """Verify path traversal attacks are blocked during write operations."""
    res_traversal = save_research_brief("../evil_script.md", "malicious payload")
    assert "Error: Access denied. Path traversal detected" in res_traversal

    res_deep = save_research_brief("../../data/evil.md", "payload")
    assert "Error: Access denied. Path traversal detected" in res_deep


# ============================================================================
# Stretch Goal Primitives: Resource and Prompt Tests
# ============================================================================

def test_database_schema_resource():
    """Verify radar://database-schema returns valid table schema and dataset list."""
    schema = get_database_schema()
    assert "SQLite Database Schema: `model_benchmarks`" in schema
    assert "CREATE TABLE model_benchmarks" in schema
    assert "PlantVillage" in schema
    assert "Amharic-OCR" in schema


def test_benchmark_upgrade_workflow_prompt():
    """Verify MCP prompt renders complete 6-step guided upgrade instructions."""
    prompt_specific = benchmark_upgrade_workflow("PlantVillage")
    assert "for the 'PlantVillage' dataset" in prompt_specific
    assert "Step 1 [Query Database]" in prompt_specific
    assert "plantvillage_plan.md" in prompt_specific

    prompt_general = benchmark_upgrade_workflow()
    assert "lowest accuracy in the database" in prompt_general
    assert "upgrade_plan.md" in prompt_general

def test_list_research_briefs_valid():
    """Verify list_research_briefs returns existing markdown notes."""
    result = list_research_briefs()
    assert "Available research briefs in data/research_briefs/:" in result
    assert "plantvillage_baseline.md" in result
    assert "amharic_ocr_notes.md" in result

