from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import regime_change_utils as rcu


def test_load_or_create_deribit_dataset_reads_existing_csv(tmp_path, raw_market_df):
    csv_path = tmp_path / "existing.csv"
    raw_market_df.reset_index().to_csv(csv_path, index=False)

    df = rcu.load_or_create_deribit_dataset(csv_path=csv_path)

    assert len(df) == len(raw_market_df)
    assert df.index.name == "timestamp"


def test_load_or_create_deribit_dataset_generates_when_missing(tmp_path, monkeypatch, raw_market_df):
    csv_path = tmp_path / "generated.csv"
    called = {}

    def fake_generate(**kwargs):
        called.update(kwargs)
        return raw_market_df

    monkeypatch.setattr(rcu, "generate_merged_deribit_dataset", fake_generate)

    df = rcu.load_or_create_deribit_dataset(
        csv_path=csv_path,
        base_asset="ETH",
        lookback_days=10,
        end_dt=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )

    assert called["base_asset"] == "ETH"
    assert called["csv_path"] == str(csv_path)
    assert csv_path.exists()
    assert len(df) == len(raw_market_df)


def test_add_core_market_features_adds_expected_columns(raw_market_df):
    df = rcu.add_core_market_features(raw_market_df)

    expected = {
        "return_close_spot",
        "return_close_perp",
        "abs_return_close_spot",
        "sq_return_close_perp",
        "log_volume_spot",
        "ma_24h_volume_perp",
        "z_24h_cost_perp",
        "volume_perp_to_spot",
        "cost_perp_per_volume",
    }
    assert expected.issubset(df.columns)
    assert np.isnan(df["return_close_spot"].iloc[0])


def test_add_rolling_volatility_features_adds_windowed_columns(raw_market_df):
    core = rcu.add_core_market_features(raw_market_df)
    df = rcu.add_rolling_volatility_features(core, windows=(3, 5))

    assert "std_3h_return_close_spot" in df.columns
    assert "std_5h_sq_return_close_perp" in df.columns


def test_add_atr_features_adds_atr_columns(raw_market_df):
    df = rcu.add_atr_features(raw_market_df)

    assert {"ATR_24h_spot", "ATR_72h_spot", "ATR_24h_perp_norm", "ATR_72h_perp_norm"}.issubset(df.columns)


def test_engineer_regime_change_features_composes_full_pipeline(raw_market_df):
    df = rcu.engineer_regime_change_features(raw_market_df)

    assert len(df.columns) > len(raw_market_df.columns)
    assert "ATR_24h_perp" in df.columns
    assert "std_72h_return_close_spot" in df.columns


def test_add_binary_high_vol_regime_returns_threshold_and_labels(engineered_df):
    df, threshold = rcu.add_binary_high_vol_regime(engineered_df.dropna().copy())

    assert "high_vol" in df.columns
    assert set(df["high_vol"].unique()).issubset({0, 1})
    assert isinstance(threshold, float)


def test_classify_volatility_regimes_returns_boolean_masks(engineered_df):
    clean = engineered_df.dropna().copy()
    high, low = rcu.classify_volatility_regimes(clean)

    assert high.dtype == bool
    assert low.dtype == bool
    assert len(high) == len(clean)


def test_save_enriched_dataset_creates_parent_directory(tmp_path, engineered_df):
    out_path = tmp_path / "nested" / "enriched.csv"

    saved_path = rcu.save_enriched_dataset(engineered_df, csv_path=out_path)

    assert saved_path == out_path
    assert out_path.exists()


def test_plot_returns_vs_volatility_returns_figure_and_axes(engineered_df):
    clean = engineered_df.dropna().copy()

    fig, axes = rcu.plot_returns_vs_volatility(clean)

    assert fig is not None
    assert len(axes) == 2
    assert axes[0].get_title() == "Returns vs Rolling Volatility"


def test_plot_binary_regime_returns_figure_and_axis(engineered_df):
    clean, _ = rcu.add_binary_high_vol_regime(engineered_df.dropna().copy())

    fig, ax = rcu.plot_binary_regime(clean)

    assert fig is not None
    assert ax.get_ylabel() == "Regime"


def test_plot_volatility_regimes_returns_figure_and_axis(engineered_df):
    clean = engineered_df.dropna().copy()

    fig, ax = rcu.plot_volatility_regimes(clean)

    assert fig is not None
    assert ax.get_title() == "Volatility Regimes (Based on 72h Std)"
