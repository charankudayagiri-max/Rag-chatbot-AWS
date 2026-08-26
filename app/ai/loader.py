"""
loader.py — Website Content Downloader

Responsibility: Download raw HTML from a given URL.
Nothing else.

This is the first stage of the AI pipeline:
    Loader → Parser → Cleaner → Chunker → Embeddings → Vector Store
"""

import requests
from app.config import logger

# Timeout for HTTP requests (connect, read) in seconds
REQUEST_TIMEOUT: tuple[int, int] = (10, 30)

# User-Agent header to avoid being blocked by websites
HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

def load_url(url: str) -> str:
    """
    Download the raw HTML content from a URL.

    Parameters
    ----------
    url : str
        The website URL to download.

    Returns
    -------
    str
        The raw HTML string.

    Raises
    ------
    ValueError
        If the URL is invalid or unreachable.
    ConnectionError
        If the request fails due to network issues.
    """
    logger.info("🌐 Loading URL: %s", url)

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        response.raise_for_status()

    except requests.exceptions.MissingSchema:
        raise ValueError(f"Invalid URL format: {url}")
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Could not connect to: {url}")
    except requests.exceptions.Timeout:
        raise ConnectionError(f"Request timed out for: {url}")
    except requests.exceptions.HTTPError as exc:
        raise ValueError(f"HTTP error {response.status_code} for {url}: {exc}")

    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        logger.warning("Unexpected content type: %s", content_type)

    html = response.text
    logger.info("✅ Downloaded %d characters from %s", len(html), url)
    return html
