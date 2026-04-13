# Deribit Data Utilities

Utilities for downloading historical market data from Deribit with pandas-friendly outputs.

The project currently exposes two main functions from [deribit_utils.py](/abs/c:/code/FE/regime_change/deribit_utils.py):

- `fetch_deribit_ohlcv(...)`
- `fetch_deribit_funding_rates(...)`

## Requirements

Install the Python packages used by the module:

```powershell
pip install -r requirements.txt
```

## Functions

### `fetch_deribit_ohlcv`

Fetches historical OHLCV candles from Deribit.

```python
fetch_deribit_ohlcv(
    base_asset: str,
    market_type: str,
    start_dt: datetime,
    end_dt: datetime,
    resolution: str = "1",
    chunk_days: int = 7,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame
```

Parameters:

- `base_asset`: Asset symbol such as `"BTC"` or `"ETH"`.
- `market_type`: `"spot"` or `"perpetual"`.
- `start_dt`: UTC start datetime.
- `end_dt`: UTC end datetime.
- `resolution`: Candle size. Examples: `"1"` for 1 minute, `"5"` for 5 minutes, `"60"` for 1 hour.
- `chunk_days`: Max number of days requested per API call. The function automatically reduces this if needed.
- `sleep_seconds`: Pause between requests to reduce rate-limit risk.

Returns:

- A pandas `DataFrame` indexed by `timestamp`.
- OHLCV columns: `open`, `high`, `low`, `close`, `volume`, `cost`.

Example: BTC perpetual 1-minute candles for the last 7 days

```python
from datetime import datetime, timedelta, timezone
from deribit_utils import fetch_deribit_ohlcv

end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=7)

btc_perp = fetch_deribit_ohlcv(
    base_asset="BTC",
    market_type="perpetual",
    start_dt=start_dt,
    end_dt=end_dt,
    resolution="1",
    chunk_days=7,
)

print(btc_perp.head())
print(btc_perp.tail())
```

Example: BTC spot candles

```python
from datetime import datetime, timedelta, timezone
from deribit_utils import fetch_deribit_ohlcv

end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=30)

btc_spot = fetch_deribit_ohlcv(
    base_asset="BTC",
    market_type="spot",
    start_dt=start_dt,
    end_dt=end_dt,
    resolution="60",
    chunk_days=7,
)
```

Example: save OHLCV to disk

```python
btc_perp.to_csv("btc_perpetual_1m.csv")
btc_perp.to_parquet("btc_perpetual_1m.parquet")
```

### `fetch_deribit_funding_rates`

Fetches historical funding-rate data for perpetual futures.

```python
fetch_deribit_funding_rates(
    base_asset: str,
    start_dt: datetime,
    end_dt: datetime,
    resolution: str = "8h",
    chunk_days: int = 30,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame
```

Parameters:

- `base_asset`: Asset symbol such as `"BTC"` or `"ETH"`.
- `start_dt`: UTC start datetime.
- `end_dt`: UTC end datetime.
- `resolution`: Funding interval, typically `"8h"`.
- `chunk_days`: Days requested per API call.
- `sleep_seconds`: Pause between requests.

Returns:

- A pandas `DataFrame` indexed by `timestamp`.
- Includes funding-related fields returned by Deribit such as `interest_8h`, `interest_1h`, `index_price`, and `mark_price` when present.

Example: BTC funding rates for the last 90 days

```python
from datetime import datetime, timedelta, timezone
from deribit_utils import fetch_deribit_funding_rates

end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=90)

btc_funding = fetch_deribit_funding_rates(
    base_asset="BTC",
    start_dt=start_dt,
    end_dt=end_dt,
    resolution="8h",
    chunk_days=30,
)

print(btc_funding.tail())
```

## Typical Workflow

```python
from datetime import datetime, timedelta, timezone
from deribit_utils import fetch_deribit_ohlcv, fetch_deribit_funding_rates

end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=30)

ohlcv = fetch_deribit_ohlcv("BTC", "perpetual", start_dt, end_dt, resolution="1")
funding = fetch_deribit_funding_rates("BTC", start_dt, end_dt)

merged = ohlcv.join(funding, how="left")
print(merged.tail())
```

## Notes

- Use timezone-aware UTC datetimes to avoid timestamp issues.
- Large 1-minute downloads are automatically chunked.
- Spot and perpetual instruments use different Deribit naming conventions internally.
- If Deribit returns `400 Bad Request`, double-check the market type and asset symbol you passed in.

## Files

- [deribit_utils.py](/abs/c:/code/FE/regime_change/deribit_utils.py): main utility functions
- [get_deribit_data.ipynb](/abs/c:/code/FE/regime_change/get_deribit_data.ipynb): notebook examples
- [tests/README.md](/abs/c:/code/FE/regime_change/tests/README.md): test layout and guidance
