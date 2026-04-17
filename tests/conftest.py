from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest


matplotlib.use("Agg")


@pytest.fixture
def raw_market_df() -> pd.DataFrame:
    periods = 120
    idx = pd.date_range("2024-01-01", periods=periods, freq="H", tz="UTC")
    base = np.linspace(100.0, 160.0, periods)
    close_spot = base + np.sin(np.arange(periods))
    close_perp = base + 1.5 + np.cos(np.arange(periods) / 3)
    index_price = base + 0.5

    df = pd.DataFrame(
        {
            "open_spot": close_spot - 0.5,
            "high_spot": close_spot + 1.0,
            "low_spot": close_spot - 1.0,
            "close_spot": close_spot,
            "volume_spot": np.linspace(10.0, 50.0, periods),
            "cost_spot": np.linspace(1_000.0, 5_000.0, periods),
            "open_perp": close_perp - 0.5,
            "high_perp": close_perp + 1.5,
            "low_perp": close_perp - 1.5,
            "close_perp": close_perp,
            "volume_perp": np.linspace(20.0, 100.0, periods),
            "cost_perp": np.linspace(2_000.0, 10_000.0, periods),
            "index_price": index_price,
            "interest_8h": np.linspace(0.001, 0.003, periods),
            "interest_1h": np.linspace(0.0001, 0.0003, periods),
            "prev_index_price": np.r_[index_price[0], index_price[:-1]],
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def engineered_df(raw_market_df: pd.DataFrame) -> pd.DataFrame:
    from src.regime_change_utils import engineer_regime_change_features

    return engineer_regime_change_features(raw_market_df)


@pytest.fixture
def hmm_ready_df(engineered_df: pd.DataFrame) -> pd.DataFrame:
    df = engineered_df.dropna().copy()
    assert not df.empty
    return df


@pytest.fixture
def tmp_csv_path(tmp_path: Path) -> Path:
    return tmp_path / "sample.csv"
