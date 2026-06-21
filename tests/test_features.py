"""Tests for pump.features."""

import numpy as np
import pandas as pd
import pytest

from pump.configs import (
    CorrelationThresholdConfig,
    FrequencyEncoderConfig,
    GeoClusterConfig,
    MutualInfoSelectorConfig,
    PumpAgeConfig,
    RedundantColumnDropperConfig,
    VarianceThresholdConfig,
    XGBImportanceSelectorConfig,
)
from pump.features import (
    CorrelationThresholdSelector,
    FrequencyEncoder,
    GeoClusterFeature,
    MutualInfoSelector,
    PumpAgeFeature,
    RedundantColumnDropper,
    VarianceThresholdSelector,
    XGBImportanceSelector,
)
from pump.registry import SELECTORS, TRANSFORMERS

# ── Shared helpers ─────────────────────────────────────────────────────────────


def _num_df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


def _cat_df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(v, dtype="object") for k, v in kwargs.items()})


# ── RedundantColumnDropper ─────────────────────────────────────────────────────


class TestRedundantColumnDropper:
    def test_drops_configured_columns(self):
        X = _num_df(a=[1, 2], b=[3, 4], c=[5, 6])
        cfg = RedundantColumnDropperConfig(columns_to_drop=["a", "b"])
        result = RedundantColumnDropper(cfg).fit_transform(X)
        assert list(result.columns) == ["c"]

    def test_silently_skips_absent_columns(self):
        X = _num_df(a=[1, 2])
        cfg = RedundantColumnDropperConfig(columns_to_drop=["a", "nonexistent"])
        RedundantColumnDropper(cfg).fit_transform(X)  # must not raise

    def test_preserves_unconfigured_columns(self):
        X = _num_df(drop_me=[1, 2], keep_me=[3, 4])
        cfg = RedundantColumnDropperConfig(columns_to_drop=["drop_me"])
        result = RedundantColumnDropper(cfg).fit_transform(X)
        assert "keep_me" in result.columns
        assert "drop_me" not in result.columns

    def test_does_not_mutate_input(self):
        X = _num_df(a=[1, 2], b=[3, 4])
        original = X.copy()
        cfg = RedundantColumnDropperConfig(columns_to_drop=["a"])
        RedundantColumnDropper(cfg).fit_transform(X)
        pd.testing.assert_frame_equal(X, original)


# ── PumpAgeFeature ─────────────────────────────────────────────────────────────


class TestPumpAgeFeature:
    def test_computes_correct_age(self):
        X = pd.DataFrame(
            {
                "date_recorded": ["2013-01-01", "2020-06-15"],
                "construction_year": [2000.0, 2010.0],
            }
        )
        result = PumpAgeFeature().fit_transform(X)
        assert result["pump_age"].iloc[0] == 13.0
        assert result["pump_age"].iloc[1] == 10.0

    def test_negative_age_becomes_nan(self):
        X = pd.DataFrame(
            {
                "date_recorded": ["2005-01-01"],
                "construction_year": [2010.0],  # recorded before built
            }
        )
        result = PumpAgeFeature().fit_transform(X)
        assert pd.isna(result["pump_age"].iloc[0])

    def test_nan_construction_year_produces_nan_age(self):
        X = pd.DataFrame(
            {
                "date_recorded": ["2015-01-01"],
                "construction_year": [np.nan],
            }
        )
        result = PumpAgeFeature().fit_transform(X)
        assert pd.isna(result["pump_age"].iloc[0])

    def test_preserves_other_columns(self):
        X = pd.DataFrame(
            {
                "date_recorded": ["2010-01-01"],
                "construction_year": [2000.0],
                "other": ["x"],
            }
        )
        result = PumpAgeFeature().fit_transform(X)
        assert "other" in result.columns
        assert "date_recorded" in result.columns

    def test_custom_output_column(self):
        X = pd.DataFrame({"date_recorded": ["2010-01-01"], "construction_year": [2000.0]})
        cfg = PumpAgeConfig(output_column="age_years")
        result = PumpAgeFeature(cfg).fit_transform(X)
        assert "age_years" in result.columns
        assert "pump_age" not in result.columns

    def test_does_not_mutate_input(self):
        X = pd.DataFrame({"date_recorded": ["2010-01-01"], "construction_year": [2000.0]})
        original = X.copy()
        PumpAgeFeature().fit_transform(X)
        pd.testing.assert_frame_equal(X, original)


# ── GeoClusterFeature ──────────────────────────────────────────────────────────


class TestGeoClusterFeature:
    def _coords_df(self, lats, lons) -> pd.DataFrame:
        return pd.DataFrame({"latitude": lats, "longitude": lons})

    def test_assigns_cluster_ids_to_valid_rows(self):
        X = self._coords_df([-6.0, -7.0, -8.0, -9.0], [35.0, 36.0, 37.0, 38.0])
        cfg = GeoClusterConfig(n_clusters=2)
        result = GeoClusterFeature(cfg).fit_transform(X)
        assert result["geo_cluster"].notna().all()
        assert set(result["geo_cluster"].unique()).issubset({0.0, 1.0})

    def test_nan_coords_produce_nan_cluster(self):
        X = self._coords_df([-6.0, np.nan, -8.0], [35.0, np.nan, 37.0])
        cfg = GeoClusterConfig(n_clusters=2)
        result = GeoClusterFeature(cfg).fit_transform(X)
        assert pd.isna(result["geo_cluster"].iloc[1])
        assert pd.notna(result["geo_cluster"].iloc[0])
        assert pd.notna(result["geo_cluster"].iloc[2])

    def test_uses_train_kmeans_on_test_data(self):
        X_train = self._coords_df([-6.0, -7.0, -8.0, -9.0], [35.0, 36.0, 37.0, 38.0])
        X_test = self._coords_df([-6.5, -8.5], [35.5, 37.5])
        cfg = GeoClusterConfig(n_clusters=2)
        geo = GeoClusterFeature(cfg)
        geo.fit(X_train)
        result = geo.transform(X_test)
        assert result["geo_cluster"].notna().all()

    def test_custom_output_column(self):
        X = self._coords_df([-6.0, -7.0, -8.0], [35.0, 36.0, 37.0])
        cfg = GeoClusterConfig(n_clusters=2, output_column="region_cluster")
        result = GeoClusterFeature(cfg).fit_transform(X)
        assert "region_cluster" in result.columns
        assert "geo_cluster" not in result.columns

    def test_does_not_mutate_input(self):
        X = self._coords_df([-6.0, -7.0, -8.0], [35.0, 36.0, 37.0])
        original = X.copy()
        GeoClusterFeature(GeoClusterConfig(n_clusters=2)).fit_transform(X)
        pd.testing.assert_frame_equal(X, original)


# ── FrequencyEncoder ───────────────────────────────────────────────────────────


class TestFrequencyEncoder:
    def test_known_category_maps_to_correct_frequency(self):
        # gov=3/5=0.6, ngo=2/5=0.4
        X = _cat_df(funder=["gov", "gov", "gov", "ngo", "ngo"])
        cfg = FrequencyEncoderConfig(columns=["funder"])
        result = FrequencyEncoder(cfg).fit_transform(X)
        assert pytest.approx(result["funder_freq"].iloc[0]) == 0.6
        assert pytest.approx(result["funder_freq"].iloc[3]) == 0.4

    def test_unknown_category_gets_zero(self):
        X_train = _cat_df(funder=["gov", "gov", "ngo"])
        X_test = _cat_df(funder=["unknown_org"])
        cfg = FrequencyEncoderConfig(columns=["funder"])
        enc = FrequencyEncoder(cfg)
        enc.fit(X_train)
        result = enc.transform(X_test)
        assert result["funder_freq"].iloc[0] == 0.0

    def test_creates_new_column_leaves_original(self):
        X = _cat_df(funder=["gov", "ngo"])
        cfg = FrequencyEncoderConfig(columns=["funder"])
        result = FrequencyEncoder(cfg).fit_transform(X)
        assert "funder" in result.columns  # original preserved
        assert "funder_freq" in result.columns  # new column added

    def test_custom_suffix(self):
        X = _cat_df(funder=["gov", "ngo"])
        cfg = FrequencyEncoderConfig(columns=["funder"], output_suffix="_count_pct")
        result = FrequencyEncoder(cfg).fit_transform(X)
        assert "funder_count_pct" in result.columns

    def test_auto_detects_object_columns(self):
        X = pd.DataFrame(
            {"cat": pd.Series(["a", "a", "b"], dtype="object"), "num": [1.0, 2.0, 3.0]}
        )
        result = FrequencyEncoder().fit_transform(X)
        assert "cat_freq" in result.columns
        assert "num_freq" not in result.columns  # numeric not touched

    def test_explicit_columns_limits_scope(self):
        X = _cat_df(a=["x", "x", "y"], b=["p", "q", "p"])
        cfg = FrequencyEncoderConfig(columns=["a"])
        result = FrequencyEncoder(cfg).fit_transform(X)
        assert "a_freq" in result.columns
        assert "b_freq" not in result.columns

    def test_does_not_mutate_input(self):
        X = _cat_df(funder=["gov", "ngo"])
        original = X.copy()
        FrequencyEncoder(FrequencyEncoderConfig(columns=["funder"])).fit_transform(X)
        pd.testing.assert_frame_equal(X, original)


# ── VarianceThresholdSelector ──────────────────────────────────────────────────


class TestVarianceThresholdSelector:
    def test_drops_zero_variance_column(self):
        X = _num_df(constant=[1, 1, 1, 1], varied=[1.0, 2.0, 3.0, 4.0])
        result = VarianceThresholdSelector().fit_transform(X)
        assert "constant" not in result.columns
        assert "varied" in result.columns

    def test_keeps_high_variance_columns(self):
        X = _num_df(a=[1.0, 2.0, 3.0, 4.0], b=[10.0, 20.0, 30.0, 40.0])
        result = VarianceThresholdSelector().fit_transform(X)
        assert "a" in result.columns
        assert "b" in result.columns

    def test_non_numeric_columns_pass_through(self):
        X = pd.DataFrame(
            {
                "constant_num": [1, 1, 1, 1],
                "varied_num": [1.0, 2.0, 3.0, 4.0],
                "cat": pd.Series(["a", "a", "a", "a"], dtype="object"),
            }
        )
        result = VarianceThresholdSelector().fit_transform(X)
        assert "cat" in result.columns
        assert "constant_num" not in result.columns
        assert "varied_num" in result.columns

    def test_custom_threshold(self):
        # variance of [1,2,3,4] = 1.25 → threshold 2.0 should drop it
        X = _num_df(low_var=[1.0, 2.0, 3.0, 4.0], high_var=[0.0, 10.0, 0.0, 10.0])
        result = VarianceThresholdSelector(VarianceThresholdConfig(threshold=2.0)).fit_transform(X)
        assert "low_var" not in result.columns
        assert "high_var" in result.columns

    def test_does_not_mutate_input(self):
        X = _num_df(a=[1, 2, 3], b=[1, 1, 1])
        original = X.copy()
        VarianceThresholdSelector().fit_transform(X)
        pd.testing.assert_frame_equal(X, original)


# ── CorrelationThresholdSelector ───────────────────────────────────────────────


class TestCorrelationThresholdSelector:
    def test_drops_second_of_perfectly_correlated_pair(self):
        # a and b are identical → correlation = 1.0
        X = _num_df(a=[1.0, 2.0, 3.0, 4.0], b=[1.0, 2.0, 3.0, 4.0], c=[4.0, 3.0, 1.0, 2.0])
        result = CorrelationThresholdSelector(
            CorrelationThresholdConfig(threshold=0.95)
        ).fit_transform(X)
        # a is the first column → a kept, b dropped
        assert "a" in result.columns
        assert "b" not in result.columns
        assert "c" in result.columns

    def test_keeps_uncorrelated_columns(self):
        rng = np.random.default_rng(42)
        X = _num_df(
            a=rng.standard_normal(50).tolist(),
            b=rng.standard_normal(50).tolist(),
        )
        result = CorrelationThresholdSelector(
            CorrelationThresholdConfig(threshold=0.95)
        ).fit_transform(X)
        assert "a" in result.columns
        assert "b" in result.columns

    def test_uses_configured_method(self):
        # monotone but nonlinear: spearman = 1.0, pearson < 1.0
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [v**2 for v in x]
        X = _num_df(a=x, b=y)
        spearman_result = CorrelationThresholdSelector(
            CorrelationThresholdConfig(threshold=0.95, method="spearman")
        ).fit_transform(X)
        assert "b" not in spearman_result.columns  # dropped by spearman

    def test_does_not_mutate_input(self):
        X = _num_df(a=[1.0, 2.0, 3.0], b=[1.0, 2.0, 3.0])
        original = X.copy()
        CorrelationThresholdSelector(CorrelationThresholdConfig(threshold=0.95)).fit_transform(X)
        pd.testing.assert_frame_equal(X, original)


# ── MutualInfoSelector ─────────────────────────────────────────────────────────


class TestMutualInfoSelector:
    def _make_data(self):
        # informative: perfectly separates classes; noise: random
        rng = np.random.default_rng(0)
        n = 60
        informative = [float(i % 3) for i in range(n)]
        noisy = rng.standard_normal(n).tolist()
        X = _num_df(informative=informative, noise=noisy)
        y = pd.Series([i % 3 for i in range(n)])
        return X, y

    def test_keeps_top_k_numeric_features(self):
        X, y = self._make_data()
        result = MutualInfoSelector(MutualInfoSelectorConfig(k=1)).fit(X, y).transform(X)
        assert result.shape[1] == 1
        assert "informative" in result.columns

    def test_non_numeric_passes_through(self):
        X, y = self._make_data()
        X["cat"] = pd.Series(["a"] * len(X), dtype="object")
        result = MutualInfoSelector(MutualInfoSelectorConfig(k=1)).fit(X, y).transform(X)
        assert "cat" in result.columns

    def test_k_larger_than_columns_keeps_all(self):
        X, y = self._make_data()
        result = MutualInfoSelector(MutualInfoSelectorConfig(k=100)).fit(X, y).transform(X)
        assert "informative" in result.columns
        assert "noise" in result.columns

    def test_uses_train_scores_on_test_data(self):
        X_train, y_train = self._make_data()
        X_test = _num_df(informative=[0.0, 1.0, 2.0], noise=[99.0, 99.0, 99.0])
        sel = MutualInfoSelector(MutualInfoSelectorConfig(k=1))
        sel.fit(X_train, y_train)
        result = sel.transform(X_test)
        assert "informative" in result.columns

    def test_does_not_mutate_input(self):
        X, y = self._make_data()
        original = X.copy()
        MutualInfoSelector(MutualInfoSelectorConfig(k=1)).fit(X, y).transform(X)
        pd.testing.assert_frame_equal(X, original)


# ── XGBImportanceSelector ──────────────────────────────────────────────────────


class TestXGBImportanceSelector:
    def _make_data(self, n: int = 80):
        # signal perfectly predicts binary target; noise is irrelevant
        rng = np.random.default_rng(1)
        signal = [float(i % 2) for i in range(n)]
        noise = rng.standard_normal(n).tolist()
        X = _num_df(signal=signal, noise=noise)
        y = pd.Series([i % 2 for i in range(n)])
        return X, y

    def test_retains_important_features(self):
        X, y = self._make_data()
        cfg = XGBImportanceSelectorConfig(percentile_cutoff=50)
        result = XGBImportanceSelector(cfg).fit(X, y).transform(X)
        assert "signal" in result.columns

    def test_non_numeric_passes_through(self):
        X, y = self._make_data()
        X["cat"] = pd.Series(["a"] * len(X), dtype="object")
        cfg = XGBImportanceSelectorConfig(percentile_cutoff=50)
        result = XGBImportanceSelector(cfg).fit(X, y).transform(X)
        assert "cat" in result.columns

    def test_string_target_handled(self):
        X, y = self._make_data()
        y_str = y.map({0: "functional", 1: "non functional"})
        cfg = XGBImportanceSelectorConfig(percentile_cutoff=0)
        result = XGBImportanceSelector(cfg).fit(X, y_str).transform(X)
        assert not result.empty

    def test_uses_train_importance_on_test_data(self):
        X_train, y_train = self._make_data()
        X_test = _num_df(signal=[0.0, 1.0], noise=[99.0, 99.0])
        cfg = XGBImportanceSelectorConfig(percentile_cutoff=50)
        sel = XGBImportanceSelector(cfg)
        sel.fit(X_train, y_train)
        result = sel.transform(X_test)
        assert "signal" in result.columns

    def test_does_not_mutate_input(self):
        X, y = self._make_data()
        original = X.copy()
        XGBImportanceSelector(XGBImportanceSelectorConfig(percentile_cutoff=0)).fit(X, y).transform(
            X
        )
        pd.testing.assert_frame_equal(X, original)


# ── Registry ───────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_all_engineering_transformers_registered(self):
        keys = TRANSFORMERS.keys()
        for name in ("redundant_dropper", "pump_age", "geo_cluster", "frequency_encoder"):
            assert name in keys

    def test_all_selection_selectors_registered(self):
        keys = SELECTORS.keys()
        for name in (
            "variance_threshold",
            "correlation_threshold",
            "mutual_info",
            "xgb_importance",
        ):
            assert name in keys

    def test_get_returns_correct_engineering_classes(self):
        assert TRANSFORMERS.get("redundant_dropper") is RedundantColumnDropper
        assert TRANSFORMERS.get("pump_age") is PumpAgeFeature
        assert TRANSFORMERS.get("geo_cluster") is GeoClusterFeature
        assert TRANSFORMERS.get("frequency_encoder") is FrequencyEncoder

    def test_get_returns_correct_selector_classes(self):
        assert SELECTORS.get("variance_threshold") is VarianceThresholdSelector
        assert SELECTORS.get("correlation_threshold") is CorrelationThresholdSelector
        assert SELECTORS.get("mutual_info") is MutualInfoSelector
        assert SELECTORS.get("xgb_importance") is XGBImportanceSelector


# ── Integration ────────────────────────────────────────────────────────────────


class TestIntegration:
    def test_full_engineering_chain(self):
        """
        redundant_dropper → pump_age → geo_cluster → frequency_encoder
        """
        X = pd.DataFrame(
            {
                "date_recorded": ["2013-01-01", "2015-06-01", "2010-03-15", "2018-09-20"],
                "construction_year": [2000.0, 2005.0, np.nan, 2010.0],
                "latitude": [-6.0, -7.0, -8.0, -9.0],
                "longitude": [35.0, 36.0, 37.0, 38.0],
                "funder": ["gov", "ngo", "gov", "other"],
                "to_drop": [1, 2, 3, 4],
            }
        )

        X = RedundantColumnDropper(
            RedundantColumnDropperConfig(columns_to_drop=["to_drop"])
        ).fit_transform(X)
        assert "to_drop" not in X.columns

        X = PumpAgeFeature().fit_transform(X)
        assert "pump_age" in X.columns
        assert pd.isna(X["pump_age"].iloc[2])  # NaN construction_year

        X = GeoClusterFeature(GeoClusterConfig(n_clusters=2)).fit_transform(X)
        assert "geo_cluster" in X.columns

        X = FrequencyEncoder(FrequencyEncoderConfig(columns=["funder"])).fit_transform(X)
        assert "funder_freq" in X.columns
        assert "funder" in X.columns  # original preserved

    def test_full_selection_chain(self):
        """
        variance_threshold → correlation_threshold → mutual_info
        """
        rng = np.random.default_rng(7)
        n = 60
        X = pd.DataFrame(
            {
                "constant": [1.0] * n,
                "a": rng.standard_normal(n).tolist(),
                "b": rng.standard_normal(n).tolist(),
                "a_copy": None,  # will be set as perfect copy of a
            }
        )
        X["a_copy"] = X["a"]
        y = pd.Series([i % 3 for i in range(n)])

        X = VarianceThresholdSelector().fit_transform(X)
        assert "constant" not in X.columns

        X = (
            CorrelationThresholdSelector(CorrelationThresholdConfig(threshold=0.95))
            .fit(X, y)
            .transform(X)
        )
        # one of a / a_copy dropped
        assert not (("a" in X.columns) and ("a_copy" in X.columns))

        X = MutualInfoSelector(MutualInfoSelectorConfig(k=10)).fit(X, y).transform(X)
        assert X.shape[1] <= 10
