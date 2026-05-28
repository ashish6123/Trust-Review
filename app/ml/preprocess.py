"""Text preprocessing – shared between training and inference."""

import html
import re

from app.core.config import MAX_TEXT_CHARS


def clean_text(text: str) -> str:
    """Normalise a single review string for ML/DL input."""
    if not isinstance(text, str):
        return ""
    # Truncate before running regexes to keep processing O(MAX_TEXT_CHARS)
    text = text[:MAX_TEXT_CHARS]
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
