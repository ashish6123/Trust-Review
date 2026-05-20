"""Scrape reviews from a URL (best-effort, for demo purposes)."""

import re
import logging
import requests
from bs4 import BeautifulSoup

log = logging.getLogger("trust_review")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Common CSS selectors for review text across popular sites
REVIEW_SELECTORS = [
    # Generic
    "[class*='review-text']",
    "[class*='reviewText']",
    "[class*='review_text']",
    "[class*='review-body']",
    "[class*='reviewBody']",
    "[class*='comment-content']",
    "[class*='review-content']",
    "[data-hook='review-body']",          # Amazon
    ".review-content p",
    # Yelp-like
    "[class*='comment__']",
    "p[class*='comment']",
    # Fallback: paragraphs inside review containers
    "[class*='review'] p",
]


def scrape_reviews(url: str, max_reviews: int = 50) -> list[str]:
    """
    Attempt to extract review texts from the given URL.

    Returns a list of review strings. May return an empty list
    if the page blocks scraping or no reviews are found.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Failed to fetch URL %s: %s", url, exc)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    reviews: list[str] = []

    for selector in REVIEW_SELECTORS:
        elements = soup.select(selector)
        for el in elements:
            text = el.get_text(separator=" ", strip=True)
            # Keep only reviews with meaningful length
            if len(text) >= 30:
                reviews.append(text)
            if len(reviews) >= max_reviews:
                break
        if reviews:
            break  # stop after first successful selector

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for r in reviews:
        key = r[:100]  # first 100 chars as dedup key
        if key not in seen:
            seen.add(key)
            unique.append(r)

    log.info("Scraped %d reviews from %s", len(unique), url)
    return unique[:max_reviews]
