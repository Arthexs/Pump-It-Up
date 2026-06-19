"""
Data diagnosis toolkit.

Functions return structured DataFrames/dicts — never just print.
Call before AND after preprocessing to produce before/after comparisons.
"""

from __future__ import annotations

import pandas as pd


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-column missing count, missing %, zero count, zero % and dtype.
    Zero-inflation matters here: construction_year, gps_height, population
    all use 0 as a proxy for missing.
    """
    raise NotImplementedError


def cardinality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-column unique count, top value, top value frequency.
    Flags high-cardinality columns (funder, installer) and constants (recorded_by).
    """
    raise NotImplementedError


def distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Extended describe() for numeric columns: adds skew and kurtosis."""
    raise NotImplementedError


def label_distribution(y: pd.Series) -> pd.DataFrame:
    """Class counts and percentages. Shows imbalance for 'needs repair'."""
    raise NotImplementedError


def redundancy_report(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Spearman-correlated column pairs above threshold.
    Catches redundancy between extraction_type / _group / _class etc.
    """
    raise NotImplementedError


def full_report(df: pd.DataFrame, y: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    """Run all diagnostics and return as a named dict. Optionally include label stats."""
    raise NotImplementedError


def compare_reports(
    before: dict[str, pd.DataFrame],
    after: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Side-by-side missing/zero % before and after preprocessing."""
    raise NotImplementedError
