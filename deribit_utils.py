from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import Any, Final, Literal, Sequence

import pandas as pd
import requests
from requests import Response, Session
from tqdm import tqdm

MarketType = Literal["spot", "perpetual"]

BASE_URL: Final[str] = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
FUNDING_URL: Final[str] = "https://www.deribit.com/api/v2/public/get_funding_rate_history"


def fetch_deribit_ohlcv(
    base_asset: str,
    market_type: MarketType,
    start_dt: datetime,
    end_dt: datetime,
    resolution: str = "1",
    chunk_days: int = 7,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    """Fetch OHLCV data for Deribit spot or perpetual instruments."""
    # Map the user-facing market type to Deribit's instrument naming scheme.
    if market_type == "spot":
        instrument_name = f"{base_asset}_USDT"
    elif market_type == "perpetual":
        instrument_name = f"{base_asset}-PERPETUAL"
    else:
        raise ValueError("market_type must be 'spot' or 'perpetual'")

    # Keep each request within Deribit's practical candle limit by shrinking long windows.
    max_candles_per_request: int = 7200
    if resolution.endswith("D"):
        candles_per_day: int = 1
    else:
        minutes_per_candle: int = int(resolution)
        candles_per_day = 1440 // minutes_per_candle

    max_chunk_days: int = max(1, max_candles_per_request // candles_per_day)
    optimal_chunk_days: int = min(chunk_days, max_chunk_days)

    all_frames: list[pd.DataFrame] = []
    session: Session = requests.Session()

    total_duration: float = (end_dt - start_dt).total_seconds()
    chunk_duration: int = optimal_chunk_days * 24 * 3600
    total_chunks: int = math.ceil(total_duration / chunk_duration)
    pbar = tqdm(total=total_chunks, desc=f"Fetching {instrument_name} OHLCV")

    chunk_start: datetime = start_dt
    while chunk_start < end_dt:
        chunk_end: datetime = min(chunk_start + timedelta(days=optimal_chunk_days), end_dt)
        params: dict[str, Any] = {
            "instrument_name": instrument_name,
            "start_timestamp": int(chunk_start.timestamp() * 1000),
            "end_timestamp": int(chunk_end.timestamp() * 1000),
            "resolution": resolution,
        }

        response: Response = session.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()

        result: dict[str, Any] = payload.get("result", {})
        if result.get("status") == "ok" and result.get("ticks"):
            all_frames.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(result["ticks"], unit="ms", utc=True),
                        "open": result["open"],
                        "high": result["high"],
                        "low": result["low"],
                        "close": result["close"],
                        "volume": result["volume"],
                        "cost": result["cost"],
                    }
                )
            )

        chunk_start = chunk_end
        time.sleep(sleep_seconds)
        pbar.update(1)

    pbar.close()

    if not all_frames:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "cost"]
        )

    # Combine chunk responses into one time-indexed series without duplicate timestamps.
    return (
        pd.concat(all_frames, ignore_index=True)
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")
    )


def fetch_deribit_funding_rates(
    base_asset: str,
    start_dt: datetime,
    end_dt: datetime,
    resolution: str = "8h",
    chunk_days: int = 30,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    """Fetch Deribit perpetual funding rate history."""
    instrument_name: str = f"{base_asset}-PERPETUAL"
    all_funding: list[pd.DataFrame] = []
    session: Session = requests.Session()

    total_duration: float = (end_dt - start_dt).total_seconds()
    chunk_duration: int = chunk_days * 24 * 3600
    total_chunks: int = math.ceil(total_duration / chunk_duration)
    pbar = tqdm(total=total_chunks, desc=f"Fetching {instrument_name} Funding Rates")

    chunk_start: datetime = start_dt
    while chunk_start < end_dt:
        chunk_end: datetime = min(chunk_start + timedelta(days=chunk_days), end_dt)
        funding_params: dict[str, Any] = {
            "instrument_name": instrument_name,
            "start_timestamp": int(chunk_start.timestamp() * 1000),
            "end_timestamp": int(chunk_end.timestamp() * 1000),
            "resolution": resolution,
        }

        response: Response = session.get(FUNDING_URL, params=funding_params, timeout=30)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()

        result: list[dict[str, Any]] = payload.get("result", [])
        if result:
            funding_df: pd.DataFrame = pd.DataFrame(result)
            if "timestamp" in funding_df.columns:
                funding_df["timestamp"] = pd.to_datetime(
                    funding_df["timestamp"], unit="ms", utc=True
                )
                all_funding.append(funding_df)

        chunk_start = chunk_end
        time.sleep(sleep_seconds)
        pbar.update(1)

    pbar.close()

    if not all_funding:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "funding_rate",
                "index_price",
                "mark_price",
                "interest_8h",
                "interest_1h",
            ]
        )

    # Funding responses are also chunked, so normalize them into one indexed frame.
    return (
        pd.concat(all_funding, ignore_index=True)
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")
    )


def merge_deribit_dataframes(
    spot_df: pd.DataFrame,
    perp_df: pd.DataFrame,
    funding_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Deribit spot, perpetual, and funding rate data into one DataFrame."""
    # Use outer joins so downstream code can decide how aggressively to filter missing data.
    merged_data: pd.DataFrame = pd.merge(
        spot_df,
        perp_df,
        left_index=True,
        right_index=True,
        suffixes=("_spot", "_perp"),
        how="outer",
    )
    merged_data = pd.merge(
        merged_data,
        funding_df,
        left_index=True,
        right_index=True,
        how="outer",
    )
    merged_data.index.name = "timestamp"
    return merged_data


def generate_merged_deribit_dataset(
    base_asset: str,
    start_dt: datetime,
    end_dt: datetime,
    ohlcv_resolution: str = "1",
    funding_resolution: str = "8h",
    spot_chunk_days: int = 7,
    perp_chunk_days: int = 7,
    funding_chunk_days: int = 30,
    save_csv: bool = True,
    csv_path: str | None = None,
    dropna_subset: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Fetch spot, perpetual, and funding data for a base asset and merge it.

    Set ``dropna_subset`` to remove rows where any of the named columns are missing.
    For example, use ``dropna_subset=("open_spot", "open_perp")`` to keep only rows
    where both spot and perpetual OHLCV data are present.
    """
    # Fetch each data source separately so callers can control resolutions and chunk sizes.
    spot_data: pd.DataFrame = fetch_deribit_ohlcv(
        base_asset=base_asset,
        market_type="spot",
        start_dt=start_dt,
        end_dt=end_dt,
        resolution=ohlcv_resolution,
        chunk_days=spot_chunk_days,
    )
    perp_data: pd.DataFrame = fetch_deribit_ohlcv(
        base_asset=base_asset,
        market_type="perpetual",
        start_dt=start_dt,
        end_dt=end_dt,
        resolution=ohlcv_resolution,
        chunk_days=perp_chunk_days,
    )
    funding_rates: pd.DataFrame = fetch_deribit_funding_rates(
        base_asset=base_asset,
        start_dt=start_dt,
        end_dt=end_dt,
        resolution=funding_resolution,
        chunk_days=funding_chunk_days,
    )

    merged_data: pd.DataFrame = merge_deribit_dataframes(
        spot_data, perp_data, funding_rates
    )

    if dropna_subset is not None:
        merged_data = merged_data.dropna(subset=list(dropna_subset))

    if save_csv:
        output_path: str = csv_path or f"merged_{base_asset.lower()}_data.csv"
        merged_data.to_csv(output_path, index_label="timestamp")
        print(f"Merged {base_asset} data saved to '{output_path}'")

    return merged_data


__all__ = [
    "fetch_deribit_ohlcv",
    "fetch_deribit_funding_rates",
    "merge_deribit_dataframes",
    "generate_merged_deribit_dataset",
]
