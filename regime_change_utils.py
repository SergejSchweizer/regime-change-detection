from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deribit_utils import generate_merged_deribit_dataset


EPSILON = 1e-8
WINDOW_24H = 24
WINDOW_72H = 72
DEFAULT_LOOKBACK_DAYS = 365 * 5


@dataclass(frozen=True)
class RegimeThresholds:
    high_quantile: float = 0.75
    low_quantile: float = 0.25


def load_or_create_deribit_dataset(
    csv_path: str | Path = "deribit_data.csv",
    base_asset: str = "BTC",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    end_dt: datetime | None = None,
) -> pd.DataFrame:
    """Load cached Deribit data or fetch it from the API."""
    data_path = Path(csv_path)

    if data_path.exists():
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
        return df.set_index("timestamp")

    effective_end_dt = end_dt or datetime.now(timezone.utc)
    start_dt = effective_end_dt - timedelta(days=lookback_days)

    df = generate_merged_deribit_dataset(
        base_asset=base_asset,
        start_dt=start_dt,
        end_dt=effective_end_dt,
        ohlcv_resolution="60",
        funding_resolution="8h",
        spot_chunk_days=7,
        perp_chunk_days=7,
        funding_chunk_days=30,
        save_csv=True,
        csv_path=str(data_path),
        dropna_subset=["close_spot", "close_perp"],
    )
    df.to_csv(data_path)
    return df


def add_core_market_features(df: pd.DataFrame, eps: float = EPSILON) -> pd.DataFrame:
    """Add return, volume, cost, and cross-market features."""
    return df.assign(
        return_close_spot=lambda x: np.log(x["close_spot"] / x["close_spot"].shift(1)),
        return_close_perp=lambda x: np.log(x["close_perp"] / x["close_perp"].shift(1)),
        return_index_price=lambda x: np.log(x["index_price"] / x["index_price"].shift(1)),
        abs_return_close_spot=lambda x: x["return_close_spot"].abs(),
        abs_return_close_perp=lambda x: x["return_close_perp"].abs(),
        abs_return_index_price=lambda x: x["return_index_price"].abs(),
        sq_return_close_spot=lambda x: x["return_close_spot"] ** 2,
        sq_return_close_perp=lambda x: x["return_close_perp"] ** 2,
        sq_return_index_price=lambda x: x["return_index_price"] ** 2,
        log_volume_spot=lambda x: np.log1p(x["volume_spot"]),
        log_volume_perp=lambda x: np.log1p(x["volume_perp"]),
        ma_24h_volume_spot=lambda x: x["volume_spot"].rolling(WINDOW_24H).mean(),
        ma_72h_volume_spot=lambda x: x["volume_spot"].rolling(WINDOW_72H).mean(),
        ma_24h_volume_perp=lambda x: x["volume_perp"].rolling(WINDOW_24H).mean(),
        ma_72h_volume_perp=lambda x: x["volume_perp"].rolling(WINDOW_72H).mean(),
        std_24h_volume_spot=lambda x: x["volume_spot"].rolling(WINDOW_24H).std(),
        std_72h_volume_spot=lambda x: x["volume_spot"].rolling(WINDOW_72H).std(),
        std_24h_volume_perp=lambda x: x["volume_perp"].rolling(WINDOW_24H).std(),
        std_72h_volume_perp=lambda x: x["volume_perp"].rolling(WINDOW_72H).std(),
        z_24h_volume_spot=lambda x: (
            (x["volume_spot"] - x["volume_spot"].rolling(WINDOW_24H).mean())
            / (x["volume_spot"].rolling(WINDOW_24H).std() + eps)
        ),
        z_24h_volume_perp=lambda x: (
            (x["volume_perp"] - x["volume_perp"].rolling(WINDOW_24H).mean())
            / (x["volume_perp"].rolling(WINDOW_24H).std() + eps)
        ),
        roc_24h_volume_spot=lambda x: x["volume_spot"].pct_change(WINDOW_24H),
        roc_24h_volume_perp=lambda x: x["volume_perp"].pct_change(WINDOW_24H),
        ma_24h_cost_perp=lambda x: x["cost_perp"].rolling(WINDOW_24H).mean(),
        ma_72h_cost_perp=lambda x: x["cost_perp"].rolling(WINDOW_72H).mean(),
        std_24h_cost_perp=lambda x: x["cost_perp"].rolling(WINDOW_24H).std(),
        std_72h_cost_perp=lambda x: x["cost_perp"].rolling(WINDOW_72H).std(),
        z_24h_cost_perp=lambda x: (
            (x["cost_perp"] - x["cost_perp"].rolling(WINDOW_24H).mean())
            / (x["cost_perp"].rolling(WINDOW_24H).std() + eps)
        ),
        diff_1h_cost_perp=lambda x: x["cost_perp"].diff(1),
        diff_8h_cost_perp=lambda x: x["cost_perp"].diff(8),
        abs_cost_perp=lambda x: x["cost_perp"].abs(),
        volume_perp_to_spot=lambda x: x["volume_perp"] / (x["volume_spot"] + eps),
        perp_volume_share=lambda x: x["volume_perp"] / (
            x["volume_perp"] + x["volume_spot"] + eps
        ),
        volume_gap_perp_spot=lambda x: x["volume_perp"] - x["volume_spot"],
        log_volume_gap_perp_spot=lambda x: np.log1p(x["volume_perp"]) - np.log1p(x["volume_spot"]),
        cost_x_volume_perp=lambda x: x["cost_perp"] * x["volume_perp"],
        cost_perp_per_volume=lambda x: x["cost_perp"] / (x["volume_perp"] + eps),
        abs_change_cost_24h=lambda x: x["cost_perp"].diff().abs().rolling(WINDOW_24H).mean(),
        abs_change_volume_perp_24h=lambda x: (
            x["volume_perp"].diff().abs().rolling(WINDOW_24H).mean()
        ),
    )


def add_rolling_volatility_features(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (WINDOW_24H, WINDOW_72H),
) -> pd.DataFrame:
    """Add rolling standard deviations for return-derived columns."""
    return_cols = [
        "return_close_spot",
        "return_close_perp",
        "return_index_price",
        "abs_return_close_spot",
        "abs_return_close_perp",
        "abs_return_index_price",
        "sq_return_close_spot",
        "sq_return_close_perp",
        "sq_return_index_price",
    ]

    feature_map: dict[str, pd.Series] = {}
    for window in windows:
        for col in return_cols:
            feature_map[f"std_{window}h_{col}"] = df[col].rolling(window).std()

    return df.assign(**feature_map)


def add_atr_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add true-range and ATR-based features for spot and perpetual BTC markets."""
    tr_spot = pd.concat(
        [
            df["high_spot"] - df["low_spot"],
            (df["high_spot"] - df["close_spot"].shift(1)).abs(),
            (df["low_spot"] - df["close_spot"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr_perp = pd.concat(
        [
            df["high_perp"] - df["low_perp"],
            (df["high_perp"] - df["close_perp"].shift(1)).abs(),
            (df["low_perp"] - df["close_perp"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return df.assign(
        ATR_24h_spot=tr_spot.rolling(WINDOW_24H).mean(),
        ATR_72h_spot=tr_spot.rolling(WINDOW_72H).mean(),
        ATR_24h_spot_norm=lambda x: x["ATR_24h_spot"] / x["close_spot"],
        ATR_72h_spot_norm=lambda x: x["ATR_72h_spot"] / x["close_spot"],
        ATR_24h_perp=tr_perp.rolling(WINDOW_24H).mean(),
        ATR_72h_perp=tr_perp.rolling(WINDOW_72H).mean(),
        ATR_24h_perp_norm=lambda x: x["ATR_24h_perp"] / x["close_perp"],
        ATR_72h_perp_norm=lambda x: x["ATR_72h_perp"] / x["close_perp"],
    )


def engineer_regime_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete feature-engineering pipeline used by the notebook."""
    enriched = add_core_market_features(df)
    enriched = add_rolling_volatility_features(enriched)
    enriched = add_atr_features(enriched)
    return enriched


def add_binary_high_vol_regime(
    df: pd.DataFrame,
    vol_col: str = "std_24h_return_close_spot",
    regime_col: str = "high_vol",
    high_quantile: float = RegimeThresholds.high_quantile,
) -> tuple[pd.DataFrame, float]:
    """Label high-volatility timestamps using a quantile threshold."""
    threshold = df[vol_col].quantile(high_quantile)
    out = df.copy()
    out[regime_col] = (out[vol_col] > threshold).astype(int)
    return out, float(threshold)


def classify_volatility_regimes(
    df: pd.DataFrame,
    signal_col: str = "std_72h_return_close_spot",
    thresholds: RegimeThresholds = RegimeThresholds(),
) -> tuple[pd.Series, pd.Series]:
    """Return boolean masks for high- and low-volatility regimes."""
    signal = df[signal_col]
    high = signal > signal.quantile(thresholds.high_quantile)
    low = signal < signal.quantile(thresholds.low_quantile)
    return high, low


def save_enriched_dataset(
    df: pd.DataFrame,
    csv_path: str | Path = "deribit_enriched_data.csv",
) -> Path:
    """Persist the engineered dataset to disk."""
    output_path = Path(csv_path)
    df.to_csv(output_path)
    return output_path


def plot_returns_vs_volatility(
    df: pd.DataFrame,
    returns_col: str = "return_close_spot",
    volatility_col: str = "std_24h_return_close_spot",
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot spot returns against rolling volatility."""
    fig, ax1 = plt.subplots(figsize=(14, 6))
    df[returns_col].plot(ax=ax1, alpha=0.3, label="Returns")
    ax1.set_ylabel("Returns")
    ax1.set_title("Returns vs Rolling Volatility")

    ax2 = ax1.twinx()
    df[volatility_col].plot(ax=ax2, color="red", label="Vol (24h std)")
    ax2.set_ylabel("Volatility")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    return fig, (ax1, ax2)


def plot_binary_regime(
    df: pd.DataFrame,
    regime_col: str = "high_vol",
    title: str = "High Volatility Regimes",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a binary regime series as a step chart."""
    fig, ax = plt.subplots(figsize=(14, 4))
    df[regime_col].plot(ax=ax, drawstyle="steps")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Regime")
    fig.tight_layout()
    return fig, ax


def plot_volatility_regimes(
    df: pd.DataFrame,
    volatility_col: str = "std_24h_return_close_spot",
    signal_col: str = "std_72h_return_close_spot",
    thresholds: RegimeThresholds = RegimeThresholds(),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot short-term volatility with high- and low-volatility shading."""
    high, low = classify_volatility_regimes(
        df=df,
        signal_col=signal_col,
        thresholds=thresholds,
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    df[volatility_col].plot(ax=ax, label="24h Vol")
    ax.fill_between(df.index, 0, df[volatility_col], where=high, alpha=0.3, label="High Vol Regime")
    ax.fill_between(df.index, 0, df[volatility_col], where=low, alpha=0.2, label="Low Vol Regime")
    ax.set_title("Volatility Regimes (Based on 72h Std)")
    ax.set_xlabel("")
    ax.set_ylabel("Volatility")
    ax.legend()
    fig.tight_layout()
    return fig, ax


__all__ = [
    "DEFAULT_LOOKBACK_DAYS",
    "EPSILON",
    "RegimeThresholds",
    "WINDOW_24H",
    "WINDOW_72H",
    "add_atr_features",
    "add_binary_high_vol_regime",
    "add_core_market_features",
    "add_rolling_volatility_features",
    "classify_volatility_regimes",
    "engineer_regime_change_features",
    "load_or_create_deribit_dataset",
    "plot_binary_regime",
    "plot_returns_vs_volatility",
    "plot_volatility_regimes",
    "save_enriched_dataset",
]
