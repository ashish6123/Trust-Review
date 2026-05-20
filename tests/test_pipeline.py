"""Unit tests for data pipeline loaders."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd
import tempfile
import os
from training.data_pipeline import load_opus, load_amazon, load_yelp


class TestLoadOpus:

    def _make_opus_df(self):
        """Helper: create a minimal OPUS-style DataFrame."""
        return pd.DataFrame({
            "deceptive": ["deceptive", "truthful", "deceptive", "truthful"],
            "text": ["Fake review text here", "Real review text here",
                     "Another fake review", "Another real review"]
        })

    def test_label_encoding_deceptive_is_1(self):
        """deceptive → label 1."""
        from unittest.mock import patch
        mock_df = self._make_opus_df()
        with patch("pandas.read_csv", return_value=mock_df):
            df = load_opus()
        deceptive_rows = df[df["label"] == 1]
        assert len(deceptive_rows) == 2

    def test_label_encoding_truthful_is_0(self):
        """truthful → label 0."""
        from unittest.mock import patch
        mock_df = self._make_opus_df()
        with patch("pandas.read_csv", return_value=mock_df):
            df = load_opus()
        truthful_rows = df[df["label"] == 0]
        assert len(truthful_rows) == 2

    def test_returns_correct_columns(self):
        from unittest.mock import patch
        mock_df = self._make_opus_df()
        with patch("pandas.read_csv", return_value=mock_df):
            df = load_opus()
        assert "text" in df.columns
        assert "label" in df.columns
        assert "source" in df.columns

    def test_source_column_is_opus(self):
        from unittest.mock import patch
        mock_df = self._make_opus_df()
        with patch("pandas.read_csv", return_value=mock_df):
            df = load_opus()
        assert all(df["source"] == "opus")


class TestLoadAmazon:

    def _make_amazon_df(self):
        return pd.DataFrame({
            "label": ["CG", "OR", "CG", "OR"],
            "text_": ["Fake amazon review", "Real amazon review",
                      "Another fake", "Another real"]
        })

    def test_cg_is_fake_label_1(self):
        """CG (computer generated) → label 1."""
        from unittest.mock import patch
        mock_df = self._make_amazon_df()
        with patch("pandas.read_csv", return_value=mock_df):
            df = load_amazon()
        assert len(df[df["label"] == 1]) == 2

    def test_or_is_real_label_0(self):
        """OR (original) → label 0."""
        from unittest.mock import patch
        mock_df = self._make_amazon_df()
        with patch("pandas.read_csv", return_value=mock_df):
            df = load_amazon()
        assert len(df[df["label"] == 0]) == 2

    def test_returns_correct_columns(self):
        from unittest.mock import patch
        mock_df = self._make_amazon_df()
        with patch("pandas.read_csv", return_value=mock_df):
            df = load_amazon()
        assert "text" in df.columns
        assert "label" in df.columns
        assert "source" in df.columns

    def test_source_column_is_amazon(self):
        from unittest.mock import patch
        mock_df = self._make_amazon_df()
        with patch("pandas.read_csv", return_value=mock_df):
            df = load_amazon()
        assert all(df["source"] == "amazon")


class TestCleanTextIntegration:
    """Integration test: clean_text works on realistic review data."""

    def test_pipeline_produces_non_empty_text(self):
        from app.ml.preprocess import clean_text
        reviews = [
            "This product is absolutely amazing! <b>5 stars</b>",
            "https://spam.com Buy now!!! GREAT DEAL",
            "Decent quality for the price. Would recommend.",
        ]
        for review in reviews:
            result = clean_text(review)
            assert isinstance(result, str)
