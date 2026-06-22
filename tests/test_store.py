"""Tests for pump.store."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from pump import store


@pytest.fixture
def fitted_pipe() -> Pipeline:
    pipe = Pipeline([("clf", DummyClassifier(strategy="most_frequent", random_state=0))])
    X = pd.DataFrame({"a": np.zeros(10)})
    y = pd.Series(["functional"] * 10)
    pipe.fit(X, y)
    return pipe


# ── save ───────────────────────────────────────────────────────────────────────


class TestSave:
    def test_returns_existing_path(self, fitted_pipe, tmp_path):
        p = store.save(fitted_pipe, "mymodel", models_dir=tmp_path)
        assert p.exists()
        assert p.suffix == ".joblib"

    def test_stem_matches_naming_convention(self, fitted_pipe, tmp_path):
        import re

        p = store.save(fitted_pipe, "mymodel", models_dir=tmp_path)
        assert re.match(r"mymodel_\d{8}_\d{6}_\d{6}$", p.stem)

    def test_creates_nested_dir(self, fitted_pipe, tmp_path):
        nested = tmp_path / "deep" / "nested"
        store.save(fitted_pipe, "m", models_dir=nested)
        assert nested.exists()

    def test_successive_saves_produce_distinct_files(self, fitted_pipe, tmp_path):
        p1 = store.save(fitted_pipe, "mymodel", models_dir=tmp_path)
        p2 = store.save(fitted_pipe, "mymodel", models_dir=tmp_path)
        assert p1 != p2


# ── load ───────────────────────────────────────────────────────────────────────


class TestLoad:
    def test_roundtrip_returns_pipeline(self, fitted_pipe, tmp_path):
        p = store.save(fitted_pipe, "mymodel", models_dir=tmp_path)
        loaded = store.load(p.stem, models_dir=tmp_path)
        assert isinstance(loaded, Pipeline)

    def test_loaded_pipeline_can_predict(self, fitted_pipe, tmp_path):
        p = store.save(fitted_pipe, "mymodel", models_dir=tmp_path)
        loaded = store.load(p.stem, models_dir=tmp_path)
        X = pd.DataFrame({"a": np.zeros(3)})
        preds = loaded.predict(X)
        assert len(preds) == 3

    def test_missing_stem_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            store.load("doesnotexist_20990101_000000_000000", models_dir=tmp_path)


# ── load_latest ────────────────────────────────────────────────────────────────


class TestLoadLatest:
    def test_returns_pipeline(self, fitted_pipe, tmp_path):
        store.save(fitted_pipe, "mymodel", models_dir=tmp_path)
        loaded = store.load_latest("mymodel", models_dir=tmp_path)
        assert isinstance(loaded, Pipeline)

    def test_returns_most_recent(self, fitted_pipe, tmp_path):
        p1 = store.save(fitted_pipe, "mymodel", models_dir=tmp_path)
        p2 = store.save(fitted_pipe, "mymodel", models_dir=tmp_path)
        loaded = store.load_latest("mymodel", models_dir=tmp_path)
        latest_stem = max(p1.stem, p2.stem)
        assert (tmp_path / f"{latest_stem}.joblib").exists()
        assert isinstance(loaded, Pipeline)

    def test_unknown_name_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            store.load_latest("ghost", models_dir=tmp_path)

    def test_does_not_cross_contaminate_names(self, fitted_pipe, tmp_path):
        store.save(fitted_pipe, "model_a", models_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load_latest("model", models_dir=tmp_path)


# ── list_models ────────────────────────────────────────────────────────────────


class TestListModels:
    def test_nonexistent_dir_returns_empty_df(self, tmp_path):
        df = store.list_models(models_dir=tmp_path / "missing")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_empty_dir_returns_empty_df(self, tmp_path):
        df = store.list_models(models_dir=tmp_path)
        assert len(df) == 0

    def test_expected_columns(self, fitted_pipe, tmp_path):
        store.save(fitted_pipe, "mymodel", models_dir=tmp_path)
        df = store.list_models(models_dir=tmp_path)
        assert set(df.columns) == {"stem", "name", "saved_at", "path"}

    def test_row_per_file(self, fitted_pipe, tmp_path):
        store.save(fitted_pipe, "m1", models_dir=tmp_path)
        store.save(fitted_pipe, "m2", models_dir=tmp_path)
        df = store.list_models(models_dir=tmp_path)
        assert len(df) == 2

    def test_name_parsed_correctly(self, fitted_pipe, tmp_path):
        store.save(fitted_pipe, "xgb_baseline", models_dir=tmp_path)
        df = store.list_models(models_dir=tmp_path)
        assert df["name"].iloc[0] == "xgb_baseline"

    def test_saved_at_is_datetime(self, fitted_pipe, tmp_path):
        from datetime import datetime

        store.save(fitted_pipe, "mymodel", models_dir=tmp_path)
        df = store.list_models(models_dir=tmp_path)
        assert isinstance(df["saved_at"].iloc[0], datetime)

    def test_path_column_is_path(self, fitted_pipe, tmp_path):
        store.save(fitted_pipe, "mymodel", models_dir=tmp_path)
        df = store.list_models(models_dir=tmp_path)
        assert isinstance(df["path"].iloc[0], Path)
