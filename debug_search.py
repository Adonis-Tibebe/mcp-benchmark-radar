"""Small standalone harness for experimenting with the MCP search logic."""

import importlib
import os

import httpx


SEARCH_PACKAGE = os.getenv("SEARCH_PACKAGE", "duckduckgo_search")
DDGS = getattr(importlib.import_module(SEARCH_PACKAGE), "DDGS")


def search_web(query, max_results=4):
    clean_query = query.strip()
    if not clean_query:
        return "Error: Search query cannot be empty. Please provide relevant ML keywords, model names, or topics."
    if not (1 <= max_results <= 10):
        return f"Error: max_results must be between 1 and 10. Received {max_results}."

    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(clean_query, max_results=max_results))

        if not raw_results:
            return (
                f"No web search results found for query: '{clean_query}'. "
                "Try broader keywords or switch source to 'github'."
            )

        lines = [f"### Web Search Results for '{clean_query}' ({len(raw_results)} results):", ""]
        for idx, item in enumerate(raw_results, start=1):
            title = item.get("title", "Untitled")
            url = item.get("href") or item.get("link") or item.get("url", "")
            snippet = item.get("body") or item.get("snippet", "No summary available.")
            lines.extend([f"**{idx}. [{title}]({url})**", f"> {snippet}", ""])
        return "\n".join(lines).strip()
    except Exception as exc:
        return (
            f"Error performing web search: {exc}. "
            "You can retry with a different query or switch source to 'github'."
        )


def search_github(query, max_results=4):
    clean_query = query.strip()
    if not clean_query:
        return "Error: Search query cannot be empty. Please provide relevant ML keywords, model names, or topics."
    if not (1 <= max_results <= 10):
        return f"Error: max_results must be between 1 and 10. Received {max_results}."

    try:
        url = "https://api.github.com/search/repositories"
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "MCP-Benchmark-Radar/1.0",
        }
        params = {"q": clean_query, "sort": "stars", "order": "desc", "per_page": max_results}

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

        items = response.json().get("items", [])
        if not items:
            return (
                f"No GitHub repositories found matching query: '{clean_query}'. "
                "Try using broader keywords or source='web'."
            )

        lines = [f"### GitHub Repositories for '{clean_query}' ({len(items)} results):", ""]
        for idx, repo in enumerate(items, start=1):
            name = repo.get("full_name", "Unknown Repo")
            description = repo.get("description") or "No description provided."
            language = repo.get("language") or "N/A"
            lines.extend(
                [
                    f"**{idx}. [{name}]({repo.get('html_url', '')})** — "
                    f"⭐ {repo.get('stargazers_count', 0):,} stars | Language: {language}",
                    f"> {description}",
                    "",
                ]
            )
        return "\n".join(lines).strip()
    except httpx.TimeoutException:
        return "Error: GitHub search timed out after 10 seconds. Try source='web' or check network connectivity."
    except Exception as exc:
        return f"Error querying GitHub API: {exc}. Try source='web' as an alternative."


if __name__ == "__main__":
    print(f"Search package: {SEARCH_PACKAGE}")
    while True:
        try:
            source = input("\nSource [web/github, q to quit]: ").strip().lower()
            if source in {"q", "quit", "exit"}:
                break
            if source not in {"web", "github"}:
                print("Error: source must be 'web' or 'github'.")
                continue
            query = input("Query: ").strip()
            if not query:
                continue
            result = search_github(query) if source == "github" else search_web(query)
            print(f"\n{result}")
        except (KeyboardInterrupt, EOFError):
            print()
            break
