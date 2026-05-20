"""Unit tests for the review scraper."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from unittest.mock import patch, MagicMock
from app.ml.scraper import scrape_reviews


class TestScrapeReviews:

    def test_returns_list(self):
        """scrape_reviews always returns a list."""
        with patch("app.ml.scraper.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = "<html><body><p class='review-text'>Great product loved it a lot</p></body></html>"
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = scrape_reviews("https://example.com")
            assert isinstance(result, list)

    def test_returns_empty_on_request_failure(self):
        """Returns empty list when request fails."""
        import requests
        with patch("app.ml.scraper.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Connection refused")
            result = scrape_reviews("https://bad-url.example.com")
            assert result == []

    def test_filters_short_texts(self):
        """Reviews shorter than 30 chars are filtered out."""
        with patch("app.ml.scraper.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = (
                "<html><body>"
                "<p class='review-text'>Short</p>"
                "<p class='review-text'>This is a long enough review that should be kept by the scraper</p>"
                "</body></html>"
            )
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = scrape_reviews("https://example.com")
            for review in result:
                assert len(review) >= 30

    def test_deduplicates_reviews(self):
        """Duplicate reviews are removed."""
        duplicate_text = "This is a great product and I love it very much indeed"
        with patch("app.ml.scraper.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = (
                f"<html><body>"
                f"<p class='review-text'>{duplicate_text}</p>"
                f"<p class='review-text'>{duplicate_text}</p>"
                f"</body></html>"
            )
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = scrape_reviews("https://example.com")
            assert len(result) == len(set(r[:100] for r in result))

    def test_respects_max_reviews_limit(self):
        """Number of results does not exceed max_reviews."""
        reviews_html = "".join(
            f"<p class='review-text'>This is review number {i} and it is long enough to pass the filter check</p>"
            for i in range(100)
        )
        with patch("app.ml.scraper.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = f"<html><body>{reviews_html}</body></html>"
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = scrape_reviews("https://example.com", max_reviews=10)
            assert len(result) <= 10
