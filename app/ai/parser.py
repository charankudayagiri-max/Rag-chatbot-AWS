"""
parser.py — HTML Text Extractor

Responsibility: Extract meaningful readable text from raw HTML.
Nothing else.

Pipeline stage 2:
    Loader → **Parser** → Cleaner → Chunker → Embeddings → Vector Store

Uses BeautifulSoup to:
  • Remove <script>, <style>, <nav>, <footer>, <header> tags
  • Extract text from <p>, <h1>–<h6>, <li>, <td>, <article>, <section>
  • Preserve headings as structural markers
"""

from bs4 import BeautifulSoup
from app.config import logger

# Tags that contain noise, not content
NOISE_TAGS: list[str] = [
    "script", "style", "noscript", "iframe",
    "nav", "footer", "header", "aside",
    "form", "button", "input", "select",
    "svg", "canvas", "video", "audio",
]

# Block level content tags that hold text
BLOCK_TAGS: set[str] = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "td", "th", "blockquote", "pre", "code",
}


def parse_html(html: str) -> str:
    """
    Extract readable text from raw HTML, avoiding duplicate extraction.

    Parameters
    ----------
    html : str
        The raw HTML string from the loader.

    Returns
    -------
    str
        The extracted plain text with structural markers.
    """
    logger.info("📄 Parsing HTML (%d chars)", len(html))

    soup = BeautifulSoup(html, "lxml")

    # Remove noise elements
    for tag_name in NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Extract text preserving structure
    text_parts: list[str] = []

    # Get the page title
    title_tag = soup.find("title")
    if title_tag and title_tag.string:
        text_parts.append(f"# {title_tag.string.strip()}")
        text_parts.append("")

    # Get the meta description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        text_parts.append(meta_desc["content"].strip())
        text_parts.append("")

    # Walk through the body recursively to avoid nested duplicates
    body = soup.find("body") or soup

    def traverse(element) -> None:
        if element.name in BLOCK_TAGS:
            text = element.get_text(separator=" ", strip=True)
            if not text or len(text) < 3:
                return

            if element.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(element.name[1])
                prefix = "#" * level
                text_parts.append(f"{prefix} {text}")
            else:
                text_parts.append(text)
            return

        # Check if the element contains any block tag descendants
        has_block_child = any(child.name in BLOCK_TAGS for child in element.find_all())

        if not has_block_child:
            # No block descendants, treat this container as a single text block
            text = element.get_text(separator=" ", strip=True)
            if text and len(text) >= 3:
                text_parts.append(text)
            return

        # Recurse into children, preserving direct text nodes
        for child in element.children:
            if child.name:
                traverse(child)
            elif isinstance(child, str) and not child.isspace():
                text = child.strip()
                if len(text) >= 3:
                    text_parts.append(text)

    traverse(body)

    extracted = "\n".join(text_parts)
    logger.info("✅ Extracted %d characters of text", len(extracted))
    return extracted

