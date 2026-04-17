# BTC Regime Change Analysis

This repository studies BTC market regimes using Deribit spot, perpetual, and funding-rate data. The workflow is split into two notebooks and three focused utility modules:

- `src/deribit_utils.py` for Deribit data collection and dataset assembly
- `src/regime_change_utils.py` for feature engineering and simple rule-based volatility regime views
- `src/hmm_utils.py` for Hidden Markov Model fitting, feature selection, and latent regime diagnostics

## Project Layout

- `notebooks/00_data_preparation_and_feature_engineering.ipynb`: loads raw Deribit data, engineers market features, and visualizes simple volatility-based regimes
- `notebooks/01_regime_detection_and_hmm_analysis.ipynb`: searches for useful HMM feature subsets, fits a selected HMM, and inspects latent market regimes
- `src/deribit_utils.py`: Deribit API fetching and merged dataset creation
- `src/regime_change_utils.py`: feature engineering, volatility regime labeling, and exploratory plotting
- `src/hmm_utils.py`: HMM training, subset search, regime interpretation, and HMM-specific plotting
- `data/deribit_data.csv`: merged raw Deribit market dataset
- `data/deribit_enriched_data.csv`: feature-engineered dataset used by the HMM notebook
- `data/hmm_feature_selection_summary.csv`: saved summary of top HMM feature-search results
- `requirements.txt`: Python dependencies

## Requirements

Install dependencies with:

```powershell
pip install -r requirements.txt
```

## Module Guide

### `src/deribit_utils.py`

This module handles data ingestion from the Deribit public API.

It includes helpers to:

- fetch spot OHLCV candles with `fetch_deribit_ohlcv(..., market_type="spot")`
- fetch perpetual OHLCV candles with `fetch_deribit_ohlcv(..., market_type="perpetual")`
- fetch perpetual funding-rate history with `fetch_deribit_funding_rates(...)`
- merge spot, perpetual, and funding datasets with `merge_deribit_dataframes(...)`
- generate and optionally save a combined dataset with `generate_merged_deribit_dataset(...)`

Notable behavior:

- large date ranges are requested in chunks to stay within Deribit candle limits
- timestamps are normalized to UTC and used as the dataframe index
- duplicate timestamps are removed after chunked downloads
- merged outputs use outer joins so downstream analysis can decide how to handle missing values

Typical example:

```python
from datetime import datetime, timezone
from src.deribit_utils import generate_merged_deribit_dataset

df = generate_merged_deribit_dataset(
    base_asset="BTC",
    start_dt=datetime(2024, 1, 1, tzinfo=timezone.utc),
    end_dt=datetime(2024, 3, 1, tzinfo=timezone.utc),
    ohlcv_resolution="60",
    funding_resolution="8h",
    csv_path="data/deribit_data.csv",
)
```

### `src/regime_change_utils.py`

This module contains the feature-engineering and rule-based regime tooling used in the exploratory notebook.

It includes helpers to:

- load cached Deribit data or fetch it if missing with `load_or_create_deribit_dataset(...)`
- add core return, volume, cost, and cross-market features with `add_core_market_features(...)`
- add rolling volatility features with `add_rolling_volatility_features(...)`
- add spot and perpetual ATR features with `add_atr_features(...)`
- run the full feature pipeline with `engineer_regime_change_features(...)`
- create a simple binary high-volatility label with `add_binary_high_vol_regime(...)`
- classify high- and low-volatility periods with `classify_volatility_regimes(...)`
- save the enriched dataset with `save_enriched_dataset(...)`
- plot returns vs volatility, binary regimes, and shaded volatility regimes

The feature pipeline keeps downstream column names stable, so the HMM notebook can consume the enriched dataset directly.
The engineered feature families now documented in the notebook include return, absolute-return, squared-return, volume, perp-cost, cross-market structure, rolling-volatility, and ATR features.

Typical example:

```python
from src.regime_change_utils import (
    engineer_regime_change_features,
    load_or_create_deribit_dataset,
    save_enriched_dataset,
)

raw_df = load_or_create_deribit_dataset(csv_path="data/deribit_data.csv", base_asset="BTC")
df = engineer_regime_change_features(raw_df)
save_enriched_dataset(df, csv_path="data/deribit_enriched_data.csv")
```

### `src/hmm_utils.py`

This module is focused on unsupervised HMM-based regime extraction.

It supports:

- dataset loading and feature-block expansion with `load_dataset(...)` and `build_candidate_features(...)`
- chronological train / validation / test splitting with `make_time_splits(...)`
- feature cleaning and correlation filtering with `clean_feature_frame(...)` and `filter_high_correlation_features(...)`
- candidate subset generation with `generate_feature_subsets(...)`
- Gaussian HMM fitting with `fit_hmm(...)`
- HMM-derived regime feature creation with `add_hmm_features(...)`
- automatic HMM feature search with `automatic_hmm_feature_selection(...)`
- result summarization with `extract_best_hmm_feature_subset(...)` and `summarize_hmm_results(...)`
- fitting the best row or any selected row with `fit_best_hmm_from_results(...)` and `fit_hmm_from_results_index(...)`
- regime-sequence diagnostics with `resolve_hmm_columns(...)` and `compute_run_lengths(...)`
- state profiling and semantic labeling with `summarize_state_profile(...)` and `assign_regimes(...)`
- recent-window and full-history regime plotting with `plot_recent_regimes(...)` and `plot_full_regime_overlay(...)`
- regime interpretation and HMM regime plotting

The generated HMM feature frame includes:

- `{prefix}_state`
- `{prefix}_prob_0 ... {prefix}_prob_k`
- `{prefix}_max_prob`
- `{prefix}_entropy`

The current HMM notebook builds candidate inputs from named feature blocks:

- `market_basics`
- `activity`
- `structure`
- `volatility`
- `price_dynamics`
- `stress`
- `funding`

and currently searches 3-state models over subset sizes 2 through 4.

What the HMM is doing:

- Let `x_t in R^d` be the standardized feature vector at time `t` and let `z_t in {1, ..., K}` be the latent regime.
- The model assumes a first-order Markov chain over hidden states:
  `P(z_t = j | z_{t-1} = i) = A_ij`
- The initial state distribution is:
  `P(z_1 = k) = pi_k`
- Conditional on the hidden regime, the observed feature vector is Gaussian:
  `x_t | z_t = k ~ N(mu_k, Sigma_k)`
- With `covariance_type="full"` in the current notebook search, each state has its own full covariance matrix `Sigma_k`.
- Posterior regime probabilities are:
  `gamma_tk = P(z_t = k | x_1, ..., x_T)`
- The exported uncertainty metric is the row-wise posterior entropy:
  `H_t = -sum_k gamma_tk log(gamma_tk + eps)`
  and the notebook reports `avg_entropy = (1 / T) sum_t H_t`.

How model diagnostics are computed:

- `avg_self_transition = (1 / K) sum_k A_kk`
- If the inferred state sequence is `hat(z)_1, ..., hat(z)_T`, state usage is
  `state_fraction_k = (1 / T) sum_t 1[hat(z)_t = k]`
  and the code stores `min_state_fraction = min_k state_fraction_k`.
- Run lengths are consecutive stretches of equal predicted state labels; the selection logic uses the median of those lengths.
- Log-likelihood is `train_loglik = log p(x_1, ..., x_T | theta)`.
- The normalized likelihood term used by the scorer is
  `loglik_per_obs_per_feature = train_loglik / (T d)`.
- The code also reports information criteria using the estimated HMM parameter count `p`:
  `AIC = 2p - 2 train_loglik`
  `BIC = log(T) p - 2 train_loglik`

How feature-subset selection works:

- A candidate model is immediately rejected if it did not converge or if
  `min_state_fraction < 0.05`.
- For eligible models, the main score is
  `selection_score = 3.0 avg_self_transition + 1.5 min_state_fraction - 0.25 median_run_length - 2.5 avg_entropy + 0.05 loglik_per_obs_per_feature`
- So the implementation prefers persistent regimes, non-degenerate state occupancy, low posterior uncertainty, and slightly better normalized fit.
- The negative `median_run_length` term means the primary score penalizes extremely long uninterrupted runs.
- After scoring, successful models are sorted by:
  `eligible desc, selection_score desc, avg_self_transition desc, min_state_fraction desc, median_run_length desc, avg_entropy asc, loglik_per_obs_per_feature desc`
- This means `selection_score` is the main ranking rule, while the later columns act as tie-breakers in the summary table.

Typical example:

```python
from src.hmm_utils import automatic_hmm_feature_selection, summarize_hmm_results

results = automatic_hmm_feature_selection(
    df=df,
    candidate_features=["volume_perp", "volume_spot", "std_72h_sq_return_close_perp"],
    subset_min_size=2,
    subset_max_size=3,
    n_states_list=[3],
)

summary = summarize_hmm_results(results, top_n=10, stringify_features=True)
```

## Notebook Guide

### `notebooks/00_data_preparation_and_feature_engineering.ipynb`

This notebook is the feature-engineering and exploratory entry point.

It is used to:

- load cached Deribit data or regenerate it from the API
- engineer market features used later in regime modeling
- save the enriched dataset to `data/deribit_enriched_data.csv`
- visualize simple rule-based volatility regimes

The regimes in this notebook are rule-based volatility labels, not latent HMM states.
The notebook now also includes:

- a short bridge section explaining how the engineered observed features feed the later HMM objective
- a dedicated "New Features Created In This Notebook" section listing the main engineered feature families and key columns
- short "Insight from this plot" notes explaining what each volatility plot reveals

The three main plot takeaways are:

- returns and short-horizon volatility cluster together during stressed periods
- the binary `high_vol` rule makes the flagged turbulent timestamps explicit
- the slower 72-hour signal highlights that calm and stressed periods tend to persist rather than appear as isolated spikes

### `notebooks/01_regime_detection_and_hmm_analysis.ipynb`

This notebook is the latent-regime analysis notebook.

It is used to:

- load the enriched dataset
- define candidate HMM input feature blocks
- run automatic HMM feature selection
- fit a selected HMM configuration
- inspect state profiles, regime distributions, transition structure, and posterior confidence

The notebook now also includes:

- a "What The HMM Does" section with the state-transition, emission, posterior-probability, and entropy equations
- a "How Selection Works" section with the exact eligibility rule and ranking equation used in `src/hmm_utils.py`
- markdown interpretation notes for each major output block, including the selection summary, chosen configuration, state interpretation, recent regime chart, full-history overlay, regime distribution, run-length statistics, transition matrix, and posterior confidence

## Typical Workflow

1. Run `notebooks/00_data_preparation_and_feature_engineering.ipynb` to fetch BTC Deribit data if needed and produce the enriched feature dataset.
2. Use `data/deribit_enriched_data.csv` as input to `notebooks/01_regime_detection_and_hmm_analysis.ipynb`.
3. Expand the notebook's named feature blocks into candidate HMM inputs, run the HMM feature search, and inspect the top-ranked configurations.
4. Fit a selected HMM row, add `hmm_*` features back to the dataframe, and assign semantic regime labels from state profiles.
5. Analyze latent regimes through recent and full-history overlays, run-length statistics, transition matrices, and posterior probabilities.

## Notes

- The repository assumes time-ordered market data.
- `src/hmm_utils.py` is unsupervised and optimized for regime structure quality, not classifier performance.
- `xgboost` is listed in `requirements.txt`, but there is currently no XGBoost-based workflow in the repository.
- The notebooks currently import directly from `src.regime_change_utils` and `src.hmm_utils`.
