"""Text preprocessing – shared between training and inference."""

import re
import html


def clean_text(text: str) -> str:
    """Normalise a single review string for ML/DL input."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)                       # decode HTML entities
    text = re.sub(r"<[^>]+>", " ", text)             # strip HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # remove URLs
    text = re.sub(r"\S+@\S+", "", text)              # remove emails
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'-]", " ", text)  # keep basic punctuation
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
