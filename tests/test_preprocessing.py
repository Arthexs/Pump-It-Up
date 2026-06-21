"""Tests for pump.preprocessing."""

import numpy as np
import pandas as pd
import pytest

from pump.configs import (
    CategoricalImputerConfig,
    NumericImputerConfig,
    OrdinalEncoderConfig,
    TargetEncoderConfig,
    ZeroToNanCleanerConfig,
)
from pump.preprocessing import (
    CategoricalImputer,
    NumericImputer,
    OrdinalEncoder,
    TargetEncoder,
    ZeroToNanCleaner,
)
from pump.registry import TRANSFORMERS

# ── Shared helpers ─────────────────────────────────────────────────────────────


def _numeric_df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


def _cat_df(**kwargs) -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(v, dtype="object") for k, v in kwargs.items()})


# ── ZeroToNanCleaner ───────────────────────────────────────────────────────────


class TestZeroToNanCleaner:
    def test_replaces_zeros_with_nan_in_configured_columns(self):
        X = _numeric_df(construction_year=[0, 1990, 0, 2005], gps_height=[0, 100, 200, 0])
        result = ZeroToNanCleaner().fit_transform(X)
        assert result["construction_year"].isna().sum() == 2
        assert result["gps_height"].isna().sum() == 2

    def test_preserves_nonzero_values(self):
        X = _numeric_df(construction_year=[0, 1990, 2005])
        result = ZeroToNanCleaner().fit_transform(X)
        assert result["construction_year"].iloc[1] == 1990
        assert result["construction_year"].iloc[2] == 2005

    def test_preserves_existing_nan(self):
        X = _numeric_df(construction_year=[0, np.nan, 2005])
        result = ZeroToNanCleaner().fit_transform(X)
        assert result["construction_year"].isna().sum() == 2  # zero + original nan

    def test_does_not_touch_unconfigured_columns(self):
        X = _numeric_df(construction_year=[0, 2000], other=[0, 0])
        cfg = ZeroToNanCleanerConfig(columns=["construction_year"])
        result = ZeroToNanCleaner(cfg).fit_transform(X)
        assert result["other"].isna().sum() == 0

    def test_tolerates_column_absent_from_dataframe(self):
        X = _numeric_df(construction_year=[0, 2000])
        # gps_height and population are in default config but not in X — must not raise
        ZeroToNanCleaner().fit_transform(X)

    def test_does_not_mutate_input(self):
        X = _numeric_df(construction_year=[0, 1990])
        original = X.copy()
        ZeroToNanCleaner().fit_transform(X)
        pd.testing.assert_frame_equal(X, original)

    def test_all_three_default_columns_covered(self):
        X = _numeric_df(construction_year=[0], gps_height=[0], population=[0])
        result = ZeroToNanCleaner().fit_transform(X)
        assert result.isna().all(axis=None)


# ── NumericImputer ─────────────────────────────────────────────────────────────


class TestNumericImputer:
    def test_median_fills_nan(self):
        X = _numeric_df(val=[1.0, 3.0, np.nan, 5.0, 7.0])
        result = NumericImputer().fit_transform(X)
        # median of [1, 3, 5, 7] = 4.0
        assert result["val"].iloc[2] == 4.0

    def test_mean_fills_nan(self):
        X = _numeric_df(val=[0.0, 4.0, np.nan])
        result = NumericImputer(NumericImputerConfig(strategy="mean")).fit_transform(X)
        # mean of [0, 4] = 2.0
        assert result["val"].iloc[2] == 2.0

    def test_constant_fills_nan(self):
        X = _numeric_df(val=[1.0, np.nan])
        result = NumericImputer(
            NumericImputerConfig(strategy="constant", fill_value=-1.0)
        ).fit_transform(X)
        assert result["val"].iloc[1] == -1.0

    def test_train_statistic_used_on_unseen_data(self):
        X_train = _numeric_df(val=[1.0, 3.0, np.nan, 5.0, 7.0])
        X_test = _numeric_df(val=[np.nan, 10.0])
        imp = NumericImputer()
        imp.fit(X_train)
        result = imp.transform(X_test)
        assert result["val"].iloc[0] == 4.0  # train median, not test median
        assert result["val"].iloc[1] == 10.0  # non-null unchanged

    def test_auto_detects_numeric_columns(self):
        X = _numeric_df(a=[1.0, np.nan], b=[np.nan, 2.0])
        result = NumericImputer().fit_transform(X)
        assert result.isna().sum().sum() == 0

    def test_explicit_columns_limits_scope(self):
        X = _numeric_df(a=[np.nan, 1.0], b=[np.nan, 2.0])
        result = NumericImputer(NumericImputerConfig(columns=["a"])).fit_transform(X)
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 1  # untouched

    def test_does_not_mutate_input(self):
        X = _numeric_df(val=[1.0, np.nan])
        original = X.copy()
        NumericImputer().fit_transform(X)
        pd.testing.assert_frame_equal(X, original)


# ── CategoricalImputer ─────────────────────────────────────────────────────────


class TestCategoricalImputer:
    def test_most_frequent_fills_nan(self):
        X = _cat_df(col=["a", "a", "b", None])
        result = CategoricalImputer().fit_transform(X)
        assert result["col"].iloc[3] == "a"

    def test_constant_fills_nan(self):
        X = _cat_df(col=["a", None, "b"])
        result = CategoricalImputer(
            CategoricalImputerConfig(strategy="constant", fill_value="unknown")
        ).fit_transform(X)
        assert result["col"].iloc[1] == "unknown"

    def test_auto_detects_object_columns(self):
        X = pd.DataFrame({"cat": pd.Series(["a", None], dtype="object"), "num": [1.0, 2.0]})
        result = CategoricalImputer().fit_transform(X)
        assert result["cat"].isna().sum() == 0
        assert result["num"].iloc[1] == 2.0  # numeric column untouched

    def test_explicit_columns_limits_scope(self):
        X = _cat_df(a=["x", None], b=["y", None])
        result = CategoricalImputer(CategoricalImputerConfig(columns=["a"])).fit_transform(X)
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 1

    def test_does_not_mutate_input(self):
        X = _cat_df(col=["a", None])
        original = X.copy()
        CategoricalImputer().fit_transform(X)
        pd.testing.assert_frame_equal(X, original)


# ── OrdinalEncoder ─────────────────────────────────────────────────────────────


class TestOrdinalEncoder:
    def test_encodes_known_categories_to_integers(self):
        X = _cat_df(col=["bird", "cat", "dog", "cat"])
        result = OrdinalEncoder().fit_transform(X)
        # sklearn encodes alphabetically: bird=0, cat=1, dog=2
        assert set(result["col"].unique()) == {0.0, 1.0, 2.0}
        assert result["col"].iloc[0] == 0.0  # bird
        assert result["col"].iloc[1] == 1.0  # cat

    def test_unknown_category_gets_unknown_value(self):
        X_train = _cat_df(col=["cat", "dog"])
        X_test = _cat_df(col=["fish"])
        enc = OrdinalEncoder()
        enc.fit(X_train)
        result = enc.transform(X_test)
        assert result["col"].iloc[0] == -1.0  # default unknown_value

    def test_error_on_unknown_when_configured(self):
        X_train = _cat_df(col=["cat", "dog"])
        X_test = _cat_df(col=["fish"])
        enc = OrdinalEncoder(OrdinalEncoderConfig(handle_unknown="error"))
        enc.fit(X_train)
        with pytest.raises(ValueError):
            enc.transform(X_test)

    def test_auto_detects_object_columns(self):
        X = pd.DataFrame({"cat": pd.Series(["a", "b"], dtype="object"), "num": [1.0, 2.0]})
        result = OrdinalEncoder().fit_transform(X)
        assert result["cat"].dtype != object
        assert result["num"].tolist() == [1.0, 2.0]  # numeric unchanged

    def test_explicit_columns_limits_scope(self):
        X = _cat_df(a=["x", "y"], b=["p", "q"])
        result = OrdinalEncoder(OrdinalEncoderConfig(columns=["a"])).fit_transform(X)
        assert result["a"].dtype != object
        assert result["b"].tolist() == ["p", "q"]  # untouched

    def test_does_not_mutate_input(self):
        X = _cat_df(col=["a", "b"])
        original = X.copy()
        OrdinalEncoder().fit_transform(X)
        pd.testing.assert_frame_equal(X, original)


# ── TargetEncoder ──────────────────────────────────────────────────────────────


class TestTargetEncoder:
    def _make_data(self):
        X = _cat_df(funder=["A", "A", "A", "B", "B"])
        # numeric target: A group mean=1.0, B group mean=2.0, global mean=1.4
        y = pd.Series([0.0, 2.0, 1.0, 2.0, 2.0])
        return X, y

    def test_smoothed_encoding_formula(self):
        X, y = self._make_data()
        enc = TargetEncoder(TargetEncoderConfig(columns=["funder"], smoothing=1.0))
        enc.fit(X, y)
        result = enc.transform(X)
        # (3 * 1.0 + 1.0 * 1.4) / (3 + 1.0) = 4.4 / 4.0 = 1.1
        assert pytest.approx(result["funder"].iloc[0], abs=1e-9) == 1.1
        # (2 * 2.0 + 1.0 * 1.4) / (2 + 1.0) = 5.4 / 3.0 = 1.8
        assert pytest.approx(result["funder"].iloc[3], abs=1e-9) == 1.8

    def test_unknown_category_gets_global_mean(self):
        X, y = self._make_data()
        enc = TargetEncoder(TargetEncoderConfig(columns=["funder"], smoothing=1.0))
        enc.fit(X, y)
        X_test = _cat_df(funder=["UNKNOWN"])
        result = enc.transform(X_test)
        assert pytest.approx(result["funder"].iloc[0], abs=1e-9) == 1.4  # global mean

    def test_high_smoothing_shrinks_toward_global_mean(self):
        X, y = self._make_data()
        enc = TargetEncoder(TargetEncoderConfig(columns=["funder"], smoothing=1e6))
        enc.fit(X, y)
        result = enc.transform(X)
        # With massive smoothing every category approaches 1.4
        assert pytest.approx(result["funder"].iloc[0], abs=0.01) == 1.4
        assert pytest.approx(result["funder"].iloc[3], abs=0.01) == 1.4

    def test_multiclass_string_target(self):
        X = _cat_df(installer=["gov", "gov", "ngo"])
        # sorted classes: functional=0, functional needs repair=1, non functional=2
        y = pd.Series(["functional", "non functional", "non functional"])
        enc = TargetEncoder(TargetEncoderConfig(columns=["installer"], smoothing=1.0))
        enc.fit(X, y)
        result = enc.transform(X)
        # gov: targets=[0,2], mean=1.0; ngo: targets=[2], mean=2.0
        # global_mean = (0+2+2)/3 ≈ 1.333
        # encoded_gov = (2*1.0 + 1.0*1.333) / (2+1.0) ≈ 1.111
        assert result["installer"].dtype == float
        assert result["installer"].notna().all()

    def test_auto_detects_object_columns(self):
        X = pd.DataFrame(
            {"cat": pd.Series(["a", "b", "a"], dtype="object"), "num": [1.0, 2.0, 3.0]}
        )
        y = pd.Series([0.0, 1.0, 0.0])
        enc = TargetEncoder()
        enc.fit(X, y)
        result = enc.transform(X)
        assert result["cat"].dtype == float
        assert result["num"].tolist() == [1.0, 2.0, 3.0]  # numeric unchanged

    def test_does_not_mutate_input(self):
        X, y = self._make_data()
        original = X.copy()
        enc = TargetEncoder(TargetEncoderConfig(columns=["funder"]))
        enc.fit(X, y)
        enc.transform(X)
        pd.testing.assert_frame_equal(X, original)

    def test_index_alignment_between_x_and_y(self):
        X = _cat_df(col=["a", "b", "c"]).set_index(pd.Index([10, 20, 30]))
        y = pd.Series([0.0, 1.0, 2.0], index=[10, 20, 30])
        enc = TargetEncoder(TargetEncoderConfig(columns=["col"]))
        enc.fit(X, y)
        result = enc.transform(X)
        assert result["col"].notna().all()


# ── Registry ───────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_all_preprocessing_transformers_registered(self):
        keys = TRANSFORMERS.keys()
        for name in (
            "zero_to_nan",
            "numeric_imputer",
            "categorical_imputer",
            "ordinal_encoder",
            "target_encoder",
        ):
            assert name in keys

    def test_get_returns_correct_class(self):
        assert TRANSFORMERS.get("zero_to_nan") is ZeroToNanCleaner
        assert TRANSFORMERS.get("numeric_imputer") is NumericImputer
        assert TRANSFORMERS.get("categorical_imputer") is CategoricalImputer
        assert TRANSFORMERS.get("ordinal_encoder") is OrdinalEncoder
        assert TRANSFORMERS.get("target_encoder") is TargetEncoder


# ── Integration ────────────────────────────────────────────────────────────────


class TestIntegration:
    def test_full_preprocessing_chain(self):
        """
        Simulates the expected pipeline order:
        zero_to_nan → numeric_imputer → categorical_imputer
        → target_encoder → ordinal_encoder
        """
        X = pd.DataFrame(
            {
                "construction_year": [0, 1990, 0, 2005],
                "gps_height": [0.0, 150.0, np.nan, 200.0],
                "funder": ["gov", "ngo", "gov", None],
                "source": ["spring", "river", "spring", "lake"],
            }
        )
        y = pd.Series(["functional", "non functional", "functional", "non functional"])

        X = ZeroToNanCleaner().fit_transform(X)
        X = NumericImputer().fit_transform(X)
        X = CategoricalImputer().fit_transform(X)
        X = TargetEncoder(TargetEncoderConfig(columns=["funder"])).fit(X, y).transform(X)
        X = OrdinalEncoder().fit_transform(X)

        assert X.isna().sum().sum() == 0
        assert X.select_dtypes(include="object").empty
