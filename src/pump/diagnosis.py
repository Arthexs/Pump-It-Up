"""
Data diagnosis toolkit.

Functions return structured DataFrames — never just print.
Call before AND after preprocessing to produce before/after comparisons.

Public API:
    diagnose(df, y)              → DiagnosisReport (all sub-reports in one call)
    diff_reports(before, after)  → dict[str, pd.DataFrame] (before/after comparison)
"""

import dataclasses

import pandas as pd


@dataclasses.dataclass
class DiagnosisReport:
    missing: pd.DataFrame
    cardinality: pd.DataFrame
    distribution: pd.DataFrame
    dtype_audit: pd.DataFrame
    redundancy: pd.DataFrame
    label_dist: pd.DataFrame | None = None

    def __repr__(self) -> str:
        parts = [
            f"  missing:      {self.missing.shape}",
            f"  cardinality:  {self.cardinality.shape}",
            f"  distribution: {self.distribution.shape}",
            f"  dtype_audit:  {self.dtype_audit.shape}",
            f"  redundancy:   {self.redundancy.shape}",
        ]
        if self.label_dist is not None:
            parts.append(f"  label_dist:   {self.label_dist.shape}")
        return "DiagnosisReport(\n" + "\n".join(parts) + "\n)"


# ── Individual tools ───────────────────────────────────────────────────────────


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    n_missing = df.isna().sum()
    pct_missing = n_missing / len(df)
    return pd.DataFrame(
        {"n_missing": n_missing, "pct_missing": pct_missing, "dtype": df.dtypes}
    ).sort_values("pct_missing", ascending=False)


def cardinality_report(df: pd.DataFrame, max_cats: int = 50) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        vc = s.value_counts(dropna=True)
        n_unique = s.nunique(dropna=True)
        top_value = vc.index[0] if len(vc) else None
        top_freq = int(vc.iloc[0]) if len(vc) else 0
        top_pct = top_freq / len(df) if len(df) else 0.0

        if pd.api.types.is_numeric_dtype(s):
            encoding = "numeric"
        elif n_unique == 1:
            encoding = "drop"
        elif n_unique == 2:
            encoding = "binary"
        elif n_unique <= max_cats:
            encoding = "onehot"
        else:
            encoding = "target_encode"

        rows.append(
            {
                "dtype": s.dtype,
                "n_unique": n_unique,
                "top_value": top_value,
                "top_freq": top_freq,
                "top_pct": top_pct,
                "suggested_encoding": encoding,
            }
        )

    return pd.DataFrame(rows, index=df.columns)


def label_distribution(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts(dropna=False)
    return pd.DataFrame({"count": counts, "pct": counts / len(y)})


def redundancy_report(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    cols = list(corr.columns)
    pairs = []
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            val = corr.loc[col_a, col_b]
            if abs(val) >= threshold:
                pairs.append({"col_a": col_a, "col_b": col_b, "correlation": val})

    if not pairs:
        return pd.DataFrame(columns=["col_a", "col_b", "correlation"])

    return (
        pd.DataFrame(pairs)
        .sort_values("correlation", key=abs, ascending=False)
        .reset_index(drop=True)
    )


def distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    desc = numeric.describe().T.rename(
        columns={"25%": "p25", "50%": "median", "75%": "p75"}
    )
    desc["skew"] = numeric.skew()
    desc["kurtosis"] = numeric.kurtosis()
    return desc.sort_values("skew", key=abs, ascending=False)


def dtype_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        current = str(s.dtype)
        suspected: str | None = None

        if pd.api.types.is_object_dtype(s) and len(non_null):
            try:
                pd.to_numeric(non_null)
                suspected = "float64 or int64"
            except (ValueError, TypeError):
                pass

            if suspected is None:
                bool_vals = {"true", "false", "yes", "no", "0", "1", "t", "f"}
                if non_null.astype(str).str.lower().isin(bool_vals).all():
                    suspected = "bool"

            if suspected is None:
                date_hints = ("date", "time", "year", "month", "day")
                if any(kw in col.lower() for kw in date_hints):
                    try:
                        pd.to_datetime(non_null)
                        suspected = "datetime64"
                    except (ValueError, TypeError):
                        pass

        elif pd.api.types.is_integer_dtype(s):
            if set(non_null.unique()) <= {0, 1}:
                suspected = "bool"

        elif pd.api.types.is_float_dtype(s) and len(non_null):
            if (non_null == non_null.round()).all():
                suspected = "int64 (no fractional part)"

        rows.append(
            {
                "current_dtype": current,
                "suspected_dtype": suspected,
                "flagged": suspected is not None,
            }
        )

    return pd.DataFrame(rows, index=df.columns)


# ── Combined report ────────────────────────────────────────────────────────────


def diagnose(
    df: pd.DataFrame,
    y: pd.Series | None = None,
    max_cats: int = 50,
    redundancy_threshold: float = 0.95,
) -> DiagnosisReport:
    return DiagnosisReport(
        missing=missing_summary(df),
        cardinality=cardinality_report(df, max_cats=max_cats),
        distribution=distribution_summary(df),
        dtype_audit=dtype_audit(df),
        redundancy=redundancy_report(df, threshold=redundancy_threshold),
        label_dist=label_distribution(y) if y is not None else None,
    )


# ── Before / after diff ────────────────────────────────────────────────────────


def diff_reports(before: DiagnosisReport, after: DiagnosisReport) -> dict[str, pd.DataFrame]:
    return {
        "missing": _diff_missing(before.missing, after.missing),
        "cardinality": _diff_cardinality(before.cardinality, after.cardinality),
        "distribution": _diff_distribution(before.distribution, after.distribution),
        "dtype_audit": _diff_dtype_audit(before.dtype_audit, after.dtype_audit),
        "redundancy": _diff_redundancy(before.redundancy, after.redundancy),
    }


def _diff_missing(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    joined = before[["pct_missing"]].join(
        after[["pct_missing"]], how="outer", lsuffix="_before", rsuffix="_after"
    )
    joined["pct_change"] = joined["pct_missing_after"] - joined["pct_missing_before"]
    return joined.sort_values("pct_change", key=abs, ascending=False)


def _diff_cardinality(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    cols = ["n_unique", "suggested_encoding"]
    joined = before[cols].join(after[cols], how="outer", lsuffix="_before", rsuffix="_after")
    joined["encoding_changed"] = (
        joined["suggested_encoding_before"] != joined["suggested_encoding_after"]
    )
    return joined


def _diff_distribution(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["mean", "std", "skew"]
    joined = before[key_cols].join(
        after[key_cols], how="outer", lsuffix="_before", rsuffix="_after"
    )
    joined["skew_change"] = joined["skew_after"] - joined["skew_before"]
    return joined.sort_values("skew_change", key=abs, ascending=False)


def _diff_dtype_audit(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    cols = ["flagged", "suspected_dtype"]
    joined = before[cols].join(after[cols], how="outer", lsuffix="_before", rsuffix="_after")
    mask = joined["flagged_before"].fillna(False) | joined["flagged_after"].fillna(False)
    return joined[mask]


def _diff_redundancy(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    def to_map(df: pd.DataFrame) -> dict[frozenset, float]:
        return {frozenset([r.col_a, r.col_b]): r.correlation for r in df.itertuples(index=False)}

    b_map = to_map(before)
    a_map = to_map(after)

    rows = []
    for key, corr in b_map.items():
        col_a, col_b = tuple(key)
        rows.append({
            "col_a": col_a, "col_b": col_b, "correlation": corr,
            "status": "unchanged" if key in a_map else "resolved",
        })
    for key, corr in a_map.items():
        if key not in b_map:
            col_a, col_b = tuple(key)
            rows.append({"col_a": col_a, "col_b": col_b, "correlation": corr, "status": "new"})

    if not rows:
        return pd.DataFrame(columns=["col_a", "col_b", "correlation", "status"])

    return pd.DataFrame(rows).reset_index(drop=True)
