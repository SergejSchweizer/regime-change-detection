"""
Deribit API utility functions for fetching OHLCV and funding rate data.

This module provides functions to fetch historical OHLCV (Open, High, Low, Close, Volume)
data and funding rates from the Deribit exchange for both spot and perpetual markets.

Usage:
    from deribit_utils import fetch_deribit_ohlcv, fetch_deribit_funding_rates
    from datetime import datetime, timedelta, timezone
    
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=7)
    
    btc_spot = fetch_deribit_ohlcv("BTC", "spot", start_dt, end_dt)
    btc_funding = fetch_deribit_funding_rates("BTC", start_dt, end_dt)
"""

import math
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from tqdm import tqdm

# Deribit API endpoints
BASE_URL = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
FUNDING_URL = "https://www.deribit.com/api/v2/public/get_funding_rate_history"


def fetch_deribit_ohlcv(
    base_asset: str,
    market_type: str,
    start_dt: datetime,
    end_dt: datetime,
    resolution: str = "1",
    chunk_days: int = 7,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Deribit exchange.

    This function retrieves historical OHLCV (Open, High, Low, Close, Volume) data for spot or perpetual markets.

    Parameters:
        base_asset (str): The base asset symbol (e.g., 'BTC', 'ETH').
        market_type (str): Type of market - 'spot' or 'perpetual'.
        start_dt (datetime): Start date for data retrieval.
        end_dt (datetime): End date for data retrieval.
        resolution (str): Time resolution for OHLCV data (e.g., '1' for 1 minute, '60' for 1 hour).
        chunk_days (int): Number of days to fetch in each API request chunk (will be optimized automatically).
        sleep_seconds (float): Sleep time between requests to avoid rate limits.

    Returns:
        pd.DataFrame: DataFrame with OHLCV data indexed by timestamp.

    Raises:
        ValueError: If market_type is not 'spot' or 'perpetual'.
        HTTPError: If API requests fail.

    Examples:
        # Fetch BTC spot OHLCV
        btc_spot = fetch_deribit_ohlcv('BTC', 'spot', start_dt, end_dt)

        # Fetch BTC perpetual OHLCV
        btc_perp = fetch_deribit_ohlcv('BTC', 'perpetual', start_dt, end_dt)
    """
    # Determine instrument name based on market type
    if market_type == "spot":
        instrument_name = f"{base_asset}_USDT"
    elif market_type == "perpetual":
        instrument_name = f"{base_asset}-PERPETUAL"
    else:
        raise ValueError("market_type must be 'spot' or 'perpetual'")

    # Calculate optimal chunk_days based on resolution to maximize data per request
    max_candles_per_request = 7200  # Conservative estimate for Deribit API limit

    # Calculate candles per day based on resolution
    if resolution.endswith('D'):
        candles_per_day = 1
    else:
        minutes_per_candle = int(resolution)
        candles_per_day = 1440 // minutes_per_candle  # 1440 minutes per day

    max_chunk_days = max_candles_per_request // candles_per_day
    # Ensure at least 1 day
    max_chunk_days = max(1, max_chunk_days)

    # Optimize chunk_days: use the smaller of user setting and calculated max
    optimal_chunk_days = min(chunk_days, max_chunk_days)

    # Initialize data containers
    all_frames = []
    session = requests.Session()

    # Calculate total chunks for progress bar using optimal chunk_days
    total_duration = (end_dt - start_dt).total_seconds()
    chunk_duration = optimal_chunk_days * 24 * 3600
    total_chunks = math.ceil(total_duration / chunk_duration)

    # Set up progress bar description
    desc = f"Fetching {instrument_name} OHLCV"
    pbar = tqdm(total=total_chunks, desc=desc)

    # Chunked data fetching loop
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=optimal_chunk_days), end_dt)

        # Fetch OHLCV data
        params = {
            "instrument_name": instrument_name,
            "start_timestamp": int(chunk_start.timestamp() * 1000),
            "end_timestamp": int(chunk_end.timestamp() * 1000),
            "resolution": resolution,
        }

        r = session.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()

        result = payload.get("result", {})
        if result.get("status") == "ok" and result.get("ticks"):
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(result["ticks"], unit="ms", utc=True),
                "open": result["open"],
                "high": result["high"],
                "low": result["low"],
                "close": result["close"],
                "volume": result["volume"],
                "cost": result["cost"],
            })
            all_frames.append(df)

        chunk_start = chunk_end
        time.sleep(sleep_seconds)
        pbar.update(1)

    pbar.close()

    # Process OHLCV data
    if not all_frames:
        ohlcv_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "cost"])
    else:
        ohlcv_df = (
            pd.concat(all_frames, ignore_index=True)
            .drop_duplicates(subset="timestamp")
            .sort_values("timestamp")
            .set_index("timestamp")
        )

    return ohlcv_df


def fetch_deribit_funding_rates(
    base_asset: str,
    start_dt: datetime,
    end_dt: datetime,
    resolution: str = "8h",
    chunk_days: int = 30,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    """
    Fetch funding rate history for perpetual markets from Deribit exchange.

    This function uses the separate Deribit endpoint:
    /public/get_funding_rate_history

    Parameters:
        base_asset (str): The base asset symbol (e.g., 'BTC', 'ETH').
        start_dt (datetime): Start date for data retrieval.
        end_dt (datetime): End date for data retrieval.
        resolution (str): Time resolution for funding rates (e.g., '8h', '1D').
        chunk_days (int): Number of days to fetch in each API request chunk.
        sleep_seconds (float): Sleep time between requests to avoid rate limits.

    Returns:
        pd.DataFrame: DataFrame with funding rate data indexed by timestamp.
                      Columns: funding_rate, index_price, mark_price, interest_8h, interest_1h

    Raises:
        ValueError: If base_asset is invalid.
        HTTPError: If API requests fail.

    Examples:
        # Fetch BTC perpetual funding rates
        btc_funding = fetch_deribit_funding_rates('BTC', start_dt, end_dt)
    """
    # Determine instrument name for perpetual
    instrument_name = f"{base_asset}-PERPETUAL"
    
    # Initialize data containers
    all_funding = []
    session = requests.Session()
    
    # Calculate total chunks for progress bar
    total_duration = (end_dt - start_dt).total_seconds()
    chunk_duration = chunk_days * 24 * 3600
    total_chunks = math.ceil(total_duration / chunk_duration)
    
    # Set up progress bar
    pbar = tqdm(total=total_chunks, desc=f"Fetching {instrument_name} Funding Rates")
    
    # Chunked data fetching loop
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end_dt)

        funding_params = {
            "instrument_name": instrument_name,
            "start_timestamp": int(chunk_start.timestamp() * 1000),
            "end_timestamp": int(chunk_end.timestamp() * 1000),
            "resolution": resolution,
        }

        r_funding = session.get(FUNDING_URL, params=funding_params, timeout=30)
        r_funding.raise_for_status()
        payload_funding = r_funding.json()

        result_funding = payload_funding.get("result", [])
        if result_funding:
            # result_funding is a list of funding rate records
            funding_df = pd.DataFrame(result_funding)
            # Select and rename columns if they exist
            if "timestamp" in funding_df.columns:
                funding_df["timestamp"] = pd.to_datetime(funding_df["timestamp"], unit="ms", utc=True)
                all_funding.append(funding_df)

        chunk_start = chunk_end
        time.sleep(sleep_seconds)
        pbar.update(1)

    pbar.close()

    # Process funding data
    if not all_funding:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "index_price", "mark_price", "interest_8h", "interest_1h"])
    funding_df = (
        pd.concat(all_funding, ignore_index=True)
        .drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    return funding_df
