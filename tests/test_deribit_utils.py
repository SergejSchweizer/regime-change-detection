from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src import deribit_utils as du


class DummyTqdm:
    def __init__(self, *args, **kwargs):
        self.updates = []

    def update(self, value: int) -> None:
        self.updates.append(value)

    def close(self) -> None:
        return None


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(self.payloads.pop(0))


def test_fetch_deribit_ohlcv_rejects_unknown_market_type():
    with pytest.raises(ValueError, match="market_type must be 'spot' or 'perpetual'"):
        du.fetch_deribit_ohlcv(
            base_asset="BTC",
            market_type="invalid",  # type: ignore[arg-type]
            start_dt=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_dt=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )


def test_fetch_deribit_ohlcv_merges_chunks_and_deduplicates(monkeypatch):
    payloads = [
        {
            "result": {
                "status": "ok",
                "ticks": [1_700_000_000_000, 1_700_000_060_000],
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.1, 2.1],
                "volume": [10, 20],
                "cost": [100, 200],
            }
        },
        {
            "result": {
                "status": "ok",
                "ticks": [1_700_000_060_000, 1_700_000_120_000],
                "open": [2.0, 3.0],
                "high": [2.5, 3.5],
                "low": [1.5, 2.5],
                "close": [2.1, 3.1],
                "volume": [20, 30],
                "cost": [200, 300],
            }
        },
    ]
    fake_session = FakeSession(payloads)
    monkeypatch.setattr(du.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(du, "tqdm", DummyTqdm)
    monkeypatch.setattr(du.time, "sleep", lambda *_: None)

    df = du.fetch_deribit_ohlcv(
        base_asset="BTC",
        market_type="spot",
        start_dt=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_dt=datetime(2024, 1, 10, tzinfo=timezone.utc),
        resolution="1440",
        chunk_days=5,
        sleep_seconds=0.0,
    )

    assert list(df.columns) == ["open", "high", "low", "close", "volume", "cost"]
    assert len(df) == 3
    assert df.index.is_monotonic_increasing
    assert fake_session.calls[0]["params"]["instrument_name"] == "BTC_USDT"


def test_fetch_deribit_ohlcv_returns_empty_frame_when_no_ticks(monkeypatch):
    fake_session = FakeSession([{"result": {"status": "ok", "ticks": []}}])
    monkeypatch.setattr(du.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(du, "tqdm", DummyTqdm)
    monkeypatch.setattr(du.time, "sleep", lambda *_: None)

    df = du.fetch_deribit_ohlcv(
        base_asset="BTC",
        market_type="perpetual",
        start_dt=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_dt=datetime(2024, 1, 2, tzinfo=timezone.utc),
        resolution="60",
        chunk_days=7,
        sleep_seconds=0.0,
    )

    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume", "cost"]
    assert df.empty


def test_fetch_deribit_funding_rates_builds_indexed_frame(monkeypatch):
    payloads = [
        {
            "result": [
                {
                    "timestamp": 1_700_000_000_000,
                    "funding_rate": 0.01,
                    "index_price": 100.0,
                    "mark_price": 101.0,
                    "interest_8h": 0.001,
                    "interest_1h": 0.0001,
                }
            ]
        }
    ]
    fake_session = FakeSession(payloads)
    monkeypatch.setattr(du.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(du, "tqdm", DummyTqdm)
    monkeypatch.setattr(du.time, "sleep", lambda *_: None)

    df = du.fetch_deribit_funding_rates(
        base_asset="BTC",
        start_dt=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_dt=datetime(2024, 1, 2, tzinfo=timezone.utc),
        sleep_seconds=0.0,
    )

    assert len(df) == 1
    assert "funding_rate" in df.columns
    assert fake_session.calls[0]["params"]["instrument_name"] == "BTC-PERPETUAL"


def test_fetch_deribit_funding_rates_returns_empty_frame_when_no_rows(monkeypatch):
    fake_session = FakeSession([{"result": []}])
    monkeypatch.setattr(du.requests, "Session", lambda: fake_session)
    monkeypatch.setattr(du, "tqdm", DummyTqdm)
    monkeypatch.setattr(du.time, "sleep", lambda *_: None)

    df = du.fetch_deribit_funding_rates(
        base_asset="BTC",
        start_dt=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_dt=datetime(2024, 1, 2, tzinfo=timezone.utc),
        sleep_seconds=0.0,
    )

    assert df.empty
    assert "funding_rate" in df.columns


def test_merge_deribit_dataframes_outer_joins_on_timestamp():
    idx1 = pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"], utc=True)
    idx2 = pd.to_datetime(["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"], utc=True)
    spot_df = pd.DataFrame({"close": [1, 2]}, index=idx1)
    perp_df = pd.DataFrame({"close": [3, 4]}, index=idx2)
    funding_df = pd.DataFrame({"funding_rate": [0.01]}, index=idx2[:1])

    merged = du.merge_deribit_dataframes(spot_df, perp_df, funding_df)

    assert merged.index.name == "timestamp"
    assert set(merged.columns) == {"close_spot", "close_perp", "funding_rate"}
    assert len(merged) == 3


def test_generate_merged_deribit_dataset_fetches_saves_and_filters(tmp_path, monkeypatch):
    idx = pd.date_range("2024-01-01", periods=3, freq="H", tz="UTC")
    spot = pd.DataFrame({"close": [1.0, None, 3.0]}, index=idx)
    perp = pd.DataFrame({"close": [10.0, 11.0, 12.0]}, index=idx)
    funding = pd.DataFrame({"funding_rate": [0.1, 0.2, 0.3]}, index=idx)

    monkeypatch.setattr(du, "fetch_deribit_ohlcv", lambda **kwargs: spot if kwargs["market_type"] == "spot" else perp)
    monkeypatch.setattr(du, "fetch_deribit_funding_rates", lambda **kwargs: funding)

    out_path = tmp_path / "nested" / "merged.csv"
    df = du.generate_merged_deribit_dataset(
        base_asset="BTC",
        start_dt=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_dt=datetime(2024, 1, 2, tzinfo=timezone.utc),
        save_csv=True,
        csv_path=str(out_path),
        dropna_subset=["close_spot", "close_perp"],
    )

    assert len(df) == 2
    assert out_path.exists()
    saved = pd.read_csv(out_path)
    assert "timestamp" in saved.columns
