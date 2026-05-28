"""Scrape reviews from a URL (best-effort, for demo purposes)."""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import socket
import urllib.robotparser
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter

log = logging.getLogger("trust_review")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# Ordered from most-specific to least-specific to reduce noise
REVIEW_SELECTORS = [
    "[data-hook='review-body']",           # Amazon
    "[data-hook='review-collapsed-body']", # Amazon collapsed
    "[class*='review-text']",
    "[class*='reviewText']",
    "[class*='review_text']",
    "[class*='review-body']",
    "[class*='reviewBody']",
    "[class*='review-content']",
    "[class*='reviewContent']",
    "[class*='comment-content']",
    "[class*='comment__']",
    "p[class*='comment']",
    ".review-content p",
    "[class*='review'] p",
]

MAX_RESPONSE_BYTES = 5 * 1024 * 1024   # 5 MB hard cap
MIN_REVIEW_LEN = 50
MAX_REVIEW_LEN = 2000
MAX_REDIRECTS = 5
CONNECT_TIMEOUT = 5     # seconds to establish TCP connection
READ_TIMEOUT = 10       # seconds to wait for response data


class UnsafeURLError(ValueError):
    """Raised when a URL fails SSRF / scheme validation."""


def _is_private_ip(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        return True


def _check_host(host: str) -> None:
    """Resolve host and raise UnsafeURLError if any resolved address is private."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Could not resolve host: {host}") from exc
    for info in infos:
        addr = info[4][0]
        if _is_private_ip(addr):
            raise UnsafeURLError(f"Refusing to fetch private/loopback address: {addr}")


def validate_url(url: str) -> str:
    """Reject non-public http(s) URLs. Guards against SSRF."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURLError("URL must start with http:// or https://")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("URL has no host.")
    _check_host(host)
    return url


class _SafeAdapter(HTTPAdapter):
    """Re-validates the resolved IP at connect time to mitigate DNS rebinding."""

    def send(self, request, *args, **kwargs):
        parsed = urlparse(request.url)
        host = parsed.hostname
        if host:
            try:
                _check_host(host)
            except UnsafeURLError:
                raise
            except socket.gaierror:
                pass  # let requests surface its own connection error
        return super().send(request, *args, **kwargs)


def _build_session() -> requests.Session:
    session = requests.Session()
    adapter = _SafeAdapter()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _safe_get(url: str) -> requests.Response:
    """
    Fetch URL with:
    - SSRF validation at every redirect hop
    - Hard response-size cap
    - Separated connect / read timeouts
    """
    session = _build_session()
    current_url = url
    for _ in range(MAX_REDIRECTS + 1):
        resp = session.get(
            current_url,
            headers=HEADERS,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=False,
            stream=True,
        )
        if resp.is_redirect or resp.is_permanent_redirect:
            next_url = resp.headers.get("Location", "")
            if not next_url:
                break
            # Resolve relative redirects
            if next_url.startswith("/"):
                parsed = urlparse(current_url)
                next_url = f"{parsed.scheme}://{parsed.netloc}{next_url}"
            # Validate redirect target before following
            validate_url(next_url)
            current_url = next_url
            continue

        # Cap the response body to avoid memory exhaustion
        content = resp.raw.read(MAX_RESPONSE_BYTES, decode_content=True)
        resp._content = content  # make .text / .content work normally
        return resp

    raise requests.TooManyRedirects(f"Exceeded {MAX_REDIRECTS} redirects for {url}")


def _robots_allowed(url: str) -> bool:
    """Return True if robots.txt permits scraping the given URL."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return True  # assume allowed if we can't fetch robots.txt


def _extract_reviews(soup: BeautifulSoup, max_reviews: int) -> list[str]:
    """
    Try every selector; pick the one that yields the most valid-length reviews
    rather than short-circuiting on the first match (which may return noise).
    """
    best: list[str] = []
    for selector in REVIEW_SELECTORS:
        candidates = []
        for el in soup.select(selector):
            text = el.get_text(separator=" ", strip=True)
            if MIN_REVIEW_LEN <= len(text) <= MAX_REVIEW_LEN:
                candidates.append(text)
        if len(candidates) > len(best):
            best = candidates
        if len(best) >= max_reviews:
            break
    return best


def _dedup(reviews: list[str]) -> list[str]:
    """Deduplicate using full-text MD5 hash rather than a fragile prefix slice."""
    seen: set[str] = set()
    unique: list[str] = []
    for r in reviews:
        key = hashlib.md5(r.encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def scrape_reviews(url: str, max_reviews: int = 50) -> list[str]:
    """
    Attempt to extract review texts from the given URL.

    Returns a list of review strings. May return an empty list if the page
    blocks scraping or no reviews are found. Raises `UnsafeURLError` if the
    URL is not a public http(s) endpoint or resolves to a private address.
    """
    validate_url(url)

    if not _robots_allowed(url):
        log.warning("robots.txt disallows scraping %s", url)
        return []

    try:
        resp = _safe_get(url)
        resp.raise_for_status()
    except UnsafeURLError:
        raise
    except requests.RequestException as exc:
        log.warning("Failed to fetch URL %s: %s", url, exc)
        return []

    soup = BeautifulSoup(resp.content, "html.parser")
    reviews = _extract_reviews(soup, max_reviews)
    unique = _dedup(reviews)

    log.info("Scraped %d reviews from %s", len(unique), url)
    return unique[:max_reviews]
