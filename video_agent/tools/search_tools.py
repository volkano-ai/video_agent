"""Search tools for the Video Agent using SerpAPI and web scraping."""

import os
import json
import httpx


def search_web(query: str, num_results: int = 10) -> dict:
    """Search the web using Google via SerpAPI and return relevant results.

    Use this tool to find recent articles, news, statistics, images, and
    other information relevant to the video topic. You can search for
    anything needed to build a well-researched video script.

    Args:
        query (str): The search query string.
        num_results (int): Number of results to return (default 10, max 20).

    Returns:
        dict: Search results containing organic results, knowledge graph,
              and related questions if available.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return {"status": "error", "error_message": "SERPAPI_API_KEY not found in environment."}

    num_results = min(num_results, 20)

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": num_results,
        "hl": "en",
        "gl": "us",
    }

    try:
        response = httpx.get("https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as e:
        return {"status": "error", "error_message": f"SerpAPI request failed: {e.response.status_code}"}
    except Exception as e:
        return {"status": "error", "error_message": f"Search failed: {str(e)}"}

    results = {"status": "success", "query": query}

    # Extract organic results
    if "organic_results" in data:
        results["organic_results"] = [
            {
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet"),
                "date": r.get("date"),
            }
            for r in data["organic_results"][:num_results]
        ]

    # Extract knowledge graph if available
    if "knowledge_graph" in data:
        kg = data["knowledge_graph"]
        results["knowledge_graph"] = {
            "title": kg.get("title"),
            "type": kg.get("type"),
            "description": kg.get("description"),
        }

    # Extract related questions (People Also Ask)
    if "related_questions" in data:
        results["related_questions"] = [
            {"question": q.get("question"), "snippet": q.get("snippet")}
            for q in data["related_questions"][:5]
        ]

    # Extract top stories if available
    if "top_stories" in data:
        results["top_stories"] = [
            {"title": s.get("title"), "link": s.get("link"), "source": s.get("source")}
            for s in data["top_stories"][:5]
        ]

    return results


def scrape_url(url: str) -> dict:
    """Fetch and extract text content from a given URL.

    Use this tool when you need to read the full content of an article,
    blog post, or webpage that was provided by the user or found during
    search. This extracts the main text content from the page.

    Args:
        url (str): The full URL to scrape (must start with http:// or https://).

    Returns:
        dict: The extracted text content from the page, truncated to
              a reasonable length for processing.
    """
    if not url.startswith(("http://", "https://")):
        return {"status": "error", "error_message": "URL must start with http:// or https://"}

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    try:
        response = httpx.get(url, headers=headers, timeout=20, follow_redirects=True)
        response.raise_for_status()
        html = response.text
    except Exception as e:
        return {"status": "error", "error_message": f"Failed to fetch URL: {str(e)}"}

    # Simple HTML to text extraction without heavy dependencies
    import re

    # Remove script and style elements
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Decode HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate to ~8000 chars to keep context manageable
    max_chars = 8000
    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"

    return {
        "status": "success",
        "url": url,
        "content": text,
        "content_length": len(text),
    }
