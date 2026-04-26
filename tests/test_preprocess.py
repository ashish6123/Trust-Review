"""Unit tests for text preprocessing."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from app.ml.preprocess import clean_text


class TestCleanText:

    def test_basic_string(self):
        result = clean_text("This is a great product!")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_lowercases_text(self):
        result = clean_text("GREAT Product Amazing")
        assert result == result.lower()

    def test_removes_html_tags(self):
        result = clean_text("<b>Great</b> product <br/> love it")
        assert "<b>" not in result
        assert "<br/>" not in result
        assert "great" in result
        assert "love" in result

    def test_removes_urls(self):
        result = clean_text("Check this out https://example.com great product")
        assert "https" not in result
        assert "example.com" not in result

    def test_removes_emails(self):
        result = clean_text("Contact us at support@example.com for help")
        assert "@" not in result

    def test_decodes_html_entities(self):
        result = clean_text("Great &amp; amazing product")
        assert "&amp;" not in result

    def test_handles_empty_string(self):
        result = clean_text("")
        assert result == ""

    def test_handles_non_string(self):
        result = clean_text(None)
        assert result == ""

    def test_handles_numeric_input(self):
        result = clean_text(12345)
        assert result == ""

    def test_strips_extra_whitespace(self):
        result = clean_text("  too   many   spaces  ")
        assert "  " not in result
        assert result == result.strip()

    def test_keeps_meaningful_punctuation(self):
        result = clean_text("It's amazing! Really, truly great.")
        assert "'" in result or "its" in result  # apostrophe kept or word cleaned

    def test_removes_special_characters(self):
        result = clean_text("Great product 🎉🔥 loved it ###")
        assert "🎉" not in result
        assert "🔥" not in result
        assert "###" not in result

    def test_real_review_text(self):
        review = "I bought this product last week and it works perfectly. Highly recommended!"
        result = clean_text(review)
        assert "bought" in result
        assert "works" in result
        assert "recommended" in result

    def test_fake_review_pattern(self):
        review = "BEST PRODUCT EVER!!! BUY NOW!!! 5 STARS!!!"
        result = clean_text(review)
        assert isinstance(result, str)
        assert len(result) > 0
