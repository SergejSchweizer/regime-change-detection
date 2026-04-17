from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


warnings.filterwarnings("ignore")


def make_time_splits(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into chronological train, validation, and test sets."""
    n = len(df)

    if n == 0:
        raise ValueError("Input dataframe is empty.")
    if not (0 < train_frac < 1):
        raise ValueError("train_frac must lie in (0, 1).")
    if not (0 < val_frac < 1):
        raise ValueError("val_frac must lie in (0, 1).")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1.")

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    if df_train.empty or df_val.empty or df_test.empty:
        raise ValueError("One of the splits is empty. Adjust split fractions or provide more data.")

    return df_train, df_val, df_test


def generate_feature_subsets(
    features: List[str],
    min_size: int,
    max_size: int
) -> List[List[str]]:
    """Generate all feature combinations between min_size and max_size."""
    if min_size < 1:
        raise ValueError("min_size must be >= 1.")
    if max_size < min_size:
        raise ValueError("max_size must be >= min_size.")
    if max_size > len(features):
        raise ValueError("max_size cannot exceed number of features.")

    subsets: List[List[str]] = []
    for k in range(min_size, max_size + 1):
        subsets.extend([list(c) for c in itertools.combinations(features, k)])
    return subsets


def clean_feature_frame(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """Select feature columns and replace +/-inf with NaN."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df[feature_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def compute_entropy(prob_matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute row-wise entropy for a probability matrix."""
    prob_matrix = np.asarray(prob_matrix)
    if prob_matrix.ndim != 2:
        raise ValueError("prob_matrix must be 2-dimensional.")

    return -(prob_matrix * np.log(prob_matrix + eps)).sum(axis=1)


def filter_high_correlation_features(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.95
) -> List[str]:
    """Remove highly correlated features using train data only."""
    if not feature_cols:
        raise ValueError("feature_cols is empty.")
    if not (0 < threshold <= 1):
        raise ValueError("threshold must lie in (0, 1].")

    x_corr = clean_feature_frame(df_train, feature_cols).dropna()
    if x_corr.empty:
        return feature_cols.copy()

    corr = x_corr.corr().abs()
    to_drop = set()
    cols = list(corr.columns)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr.iloc[i, j] >= threshold:
                to_drop.add(cols[j])

    return [c for c in feature_cols if c not in to_drop]


def _flatten_columns_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    out = df.copy()
    out.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        for col in out.columns
    ]
    return out


def _format_feature_list_for_tqdm(feature_cols: List[str], max_len: int = 60) -> str:
    text = ",".join(feature_cols)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def load_dataset(csv_path: str | Path = "data/deribit_enriched_data.csv") -> pd.DataFrame:
    """Load a time-indexed dataset used for HMM analysis."""
    df = pd.read_csv(csv_path)
    timestamp = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    if timestamp.isna().all():
        timestamp = pd.to_datetime(df["timestamp"])
    return df.assign(timestamp=timestamp).set_index("timestamp").sort_index()


def build_candidate_features(
    df: pd.DataFrame,
    feature_blocks: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Build a deduplicated candidate feature list from named feature blocks."""
    candidate_features: List[str] = []
    missing_by_block: Dict[str, List[str]] = {}

    for block_name, feature_names in feature_blocks.items():
        existing = [feature for feature in feature_names if feature in df.columns]
        missing = [feature for feature in feature_names if feature not in df.columns]

        candidate_features.extend(existing)
        if missing:
            missing_by_block[block_name] = missing

    candidate_features = list(dict.fromkeys(candidate_features))
    return candidate_features, missing_by_block


def resolve_hmm_columns(
    df: pd.DataFrame,
    prefix: str = "hmm"
) -> Tuple[str, List[str]]:
    """Resolve the HMM state column and ordered posterior-probability columns."""
    state_col = f"{prefix}_state"
    prob_cols = [
        col for col in df.columns
        if col.startswith(f"{prefix}_prob_") and col[len(f"{prefix}_prob_"):].isdigit()
    ]
    prob_cols = sorted(prob_cols, key=lambda col: int(col.rsplit("_", 1)[-1]))
    return state_col, prob_cols


def compute_run_lengths(states: pd.Series) -> pd.Series:
    """Compute consecutive run lengths for a state sequence."""
    states = states.dropna().astype(int)
    if states.empty:
        return pd.Series(dtype=float)

    group_id = states.ne(states.shift()).cumsum()
    return states.groupby(group_id).size().astype(float)


def summarize_state_profile(
    df: pd.DataFrame,
    state_col: str,
    feature_cols: List[str]
) -> pd.DataFrame:
    """Summarize mean market features by inferred HMM state."""
    summary_features = [
        "close_perp",
        "abs_return_close_perp",
        "volume_perp",
        *feature_cols,
    ]
    summary_features = [
        feature for feature in dict.fromkeys(summary_features)
        if feature in df.columns
    ]
    return df.groupby(state_col)[summary_features].mean()


from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_recent_regimes(
    df: pd.DataFrame,
    price_col: str,
    state_col: str,
    prob_cols: List[str],
    regime_labels: Dict[int, str],
    regime_colors: Dict[int, str],
    n_points: int = 300
) -> None:
    """Plot recent price action with regime shading, regime labels, and state confidence."""
    plot_df = df[[price_col, state_col, *prob_cols]].dropna().tail(n_points).copy()
    plot_df[state_col] = plot_df[state_col].astype(int)

    x = np.arange(len(plot_df))
    change_points = plot_df[state_col].ne(plot_df[state_col].shift()).cumsum()

    active_prob = pd.Series(
        [plot_df.iloc[i][f"hmm_prob_{state}"] for i, state in enumerate(plot_df[state_col])],
        index=plot_df.index,
    )
    bar_colors = [regime_colors.get(state, "grey") for state in plot_df[state_col]]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        gridspec_kw={"height_ratios": [3, 0.8]},
        sharex=True,
    )

    # -------------------------
    # Top plot: price + regimes
    # -------------------------
    ax1.plot(x, plot_df[price_col].values, lw=1.2, color="black")
    ax1.set_title(f"Last {len(plot_df)} observations with HMM regimes")

    y_min = plot_df[price_col].min()
    y_max = plot_df[price_col].max()
    y_range = y_max - y_min
    label_y = y_max - 0.08 * y_range  # put labels near the top

    for _, block in plot_df.groupby(change_points):
        state = int(block[state_col].iloc[0])
        start_idx = plot_df.index.get_loc(block.index[0])
        end_idx = plot_df.index.get_loc(block.index[-1])

        color = regime_colors.get(state, "grey")
        label = regime_labels.get(state, f"State {state}")

        ax1.axvspan(
            start_idx - 0.5,
            end_idx + 0.5,
            alpha=0.3,
            color=color,
            edgecolor="black" if color == "white" else None,
            linewidth=0.5 if color == "white" else 0,
        )

        # annotate regime label at segment midpoint
        mid_idx = (start_idx + end_idx) / 2
        if end_idx - start_idx >= 2:  # avoid clutter on tiny segments
            ax1.text(
                mid_idx,
                label_y,
                label,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                bbox=dict(
                    facecolor="white",
                    alpha=0.7,
                    edgecolor="none",
                    boxstyle="round,pad=0.2"
                ),
                zorder=5,
            )

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=regime_colors[state],
            edgecolor="black" if regime_colors[state] == "white" else None,
        )
        for state in sorted(regime_labels)
    ]
    labels = [regime_labels[state] for state in sorted(regime_labels)]
    ax1.legend(handles, labels, title="Regimes", loc="upper left")

    ax1.set_ylabel(price_col)
    ax1.grid(True, alpha=0.3)

    # -------------------------
    # Bottom plot: probability of inferred state
    # -------------------------
    ax2.bar(x, active_prob.values, width=0.8, color=bar_colors, alpha=0.9)
    ax2.set_title("Probability of the inferred state")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis="y", alpha=0.3)

    step = max(1, len(plot_df) // 12)
    ax2.set_xticks(x[::step])
    ax2.set_xticklabels([str(idx) for idx in plot_df.index[::step]], rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

def plot_full_regime_overlay(
    df: pd.DataFrame,
    price_col: str,
    state_col: str,
    regime_labels: Dict[int, str],
    regime_colors: Dict[int, str]
) -> None:
    """Plot the full price history overlaid by inferred HMM regimes."""
    plot_df = df[[price_col, state_col]].dropna().copy()
    plot_df[state_col] = plot_df[state_col].astype(int)

    ax = plot_df[price_col].plot(
        figsize=(16, 6),
        lw=1.0,
        color="black",
        alpha=0.5,
        title=f"{price_col} segmented by regime",
    )

    for state in sorted(plot_df[state_col].unique()):
        plot_df[price_col].where(plot_df[state_col] == state).plot(
            ax=ax,
            lw=2,
            label=regime_labels.get(state, f"State {state}"),
            color=regime_colors.get(state, None),
        )

    ax.set_ylabel(price_col)
    ax.legend(title="Regime")
    ax.grid(True, alpha=0.3)
    plt.show()


def fit_hmm(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    n_states: int = 3,
    covariance_type: str = "diag",
    n_iter: int = 200,
    random_state: int = 42
) -> Tuple[GaussianHMM, StandardScaler]:
    """Fit a Gaussian HMM on the training split."""
    if n_states < 2:
        raise ValueError("n_states must be >= 2.")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty.")

    x_df = clean_feature_frame(df_train, feature_cols).dropna()
    if x_df.empty:
        raise ValueError("No valid rows for HMM fitting after dropna().")
    if len(x_df) <= n_states:
        raise ValueError("Not enough valid rows relative to number of states.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_df)

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )
    hmm.fit(x_scaled)

    if hasattr(hmm, "monitor_") and not hmm.monitor_.converged:
        warnings.warn("HMM did not converge.")

    return hmm, scaler


def add_hmm_features(
    df: pd.DataFrame,
    hmm: GaussianHMM,
    scaler: StandardScaler,
    feature_cols: List[str],
    prefix: str = "hmm"
) -> pd.DataFrame:
    """Generate HMM-derived regime features aligned to the dataframe index."""
    x_df = clean_feature_frame(df, feature_cols)
    valid_mask = x_df.notna().all(axis=1)

    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_state"] = np.nan

    for k in range(hmm.n_components):
        out[f"{prefix}_prob_{k}"] = np.nan

    out[f"{prefix}_max_prob"] = np.nan
    out[f"{prefix}_entropy"] = np.nan

    if valid_mask.sum() == 0:
        return out

    x_valid = x_df.loc[valid_mask]
    x_scaled = scaler.transform(x_valid)

    states = hmm.predict(x_scaled)
    probs = hmm.predict_proba(x_scaled)

    out.loc[valid_mask, f"{prefix}_state"] = states
    for k in range(hmm.n_components):
        out.loc[valid_mask, f"{prefix}_prob_{k}"] = probs[:, k]

    out.loc[valid_mask, f"{prefix}_max_prob"] = probs.max(axis=1)
    out.loc[valid_mask, f"{prefix}_entropy"] = compute_entropy(probs)

    return out


def _compute_state_sequence_stats(states: np.ndarray, n_states: int) -> Dict[str, float]:
    states = np.asarray(states).astype(int)
    if len(states) == 0:
        raise ValueError("Empty state sequence.")

    counts = np.bincount(states, minlength=n_states)
    fractions = counts / counts.sum()

    run_lengths = []
    current_run = 1

    for i in range(1, len(states)):
        if states[i] == states[i - 1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)

    return {
        "min_state_fraction": float(fractions.min()),
        "max_state_fraction": float(fractions.max()),
        "median_run_length": float(np.median(run_lengths)),
        "mean_run_length": float(np.mean(run_lengths)),
        "n_runs": float(len(run_lengths)),
    }


def _count_hmm_parameters(n_states: int, n_features: int, covariance_type: str) -> int:
    if covariance_type not in {"diag", "full", "spherical", "tied"}:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    startprob_params = n_states - 1
    transmat_params = n_states * (n_states - 1)
    mean_params = n_states * n_features

    if covariance_type == "diag":
        cov_params = n_states * n_features
    elif covariance_type == "full":
        cov_params = n_states * (n_features * (n_features + 1) // 2)
    elif covariance_type == "spherical":
        cov_params = n_states
    else:
        cov_params = n_features * (n_features + 1) // 2

    return int(startprob_params + transmat_params + mean_params + cov_params)


def _safe_float(value: float, default: float = np.nan) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return default
    except Exception:
        return default


def evaluate_hmm_feature_subset(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    n_states: int = 3,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42
) -> Dict:
    """Fit an HMM on one feature subset and compute regime diagnostics."""
    hmm, scaler = fit_hmm(
        df_train=df_train,
        feature_cols=feature_cols,
        n_states=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )

    x_df = clean_feature_frame(df_train, feature_cols).dropna()
    x_scaled = scaler.transform(x_df)

    states = hmm.predict(x_scaled)
    probs = hmm.predict_proba(x_scaled)

    seq_stats = _compute_state_sequence_stats(states, n_states=n_states)

    avg_self_transition = float(np.mean(np.diag(hmm.transmat_)))
    train_loglik = float(hmm.score(x_scaled))
    avg_entropy = float(compute_entropy(probs).mean())

    converged = True
    if hasattr(hmm, "monitor_") and hasattr(hmm.monitor_, "converged"):
        converged = bool(hmm.monitor_.converged)

    n_obs_used = int(len(x_df))
    n_features = int(len(feature_cols))
    n_params = _count_hmm_parameters(
        n_states=n_states,
        n_features=n_features,
        covariance_type=covariance_type
    )

    loglik_per_obs = train_loglik / n_obs_used
    loglik_per_obs_per_feature = train_loglik / (n_obs_used * n_features)
    aic = 2 * n_params - 2 * train_loglik
    bic = np.log(n_obs_used) * n_params - 2 * train_loglik

    return {
        "feature_cols": feature_cols,
        "n_states": n_states,
        "n_features": n_features,
        "n_obs_used": n_obs_used,
        "converged": converged,
        "train_loglik": train_loglik,
        "loglik_per_obs": float(loglik_per_obs),
        "loglik_per_obs_per_feature": float(loglik_per_obs_per_feature),
        "aic": float(aic),
        "bic": float(bic),
        "n_params": int(n_params),
        "avg_self_transition": avg_self_transition,
        "avg_entropy": avg_entropy,
        "min_state_fraction": seq_stats["min_state_fraction"],
        "max_state_fraction": seq_stats["max_state_fraction"],
        "median_run_length": seq_stats["median_run_length"],
        "mean_run_length": seq_stats["mean_run_length"],
        "n_runs": seq_stats["n_runs"],
        "transition_matrix": hmm.transmat_.copy()
    }


def simple_hmm_selection_score(
    row: pd.Series,
    min_state_fraction_threshold: float = 0.05
) -> float:
    """Regime-quality selection score for HMM feature subsets."""
    if not bool(row["converged"]):
        return -np.inf

    min_frac = _safe_float(row["min_state_fraction"], default=-np.inf)
    if not np.isfinite(min_frac) or min_frac < min_state_fraction_threshold:
        return -np.inf

    avg_self_transition = _safe_float(row["avg_self_transition"], default=-np.inf)
    avg_entropy = _safe_float(row["avg_entropy"], default=np.inf)
    median_run_length = _safe_float(row["median_run_length"], default=-np.inf)
    loglik_per_obs_per_feature = _safe_float(row.get("loglik_per_obs_per_feature", np.nan), default=0.0)

    if not np.isfinite(avg_self_transition) or not np.isfinite(avg_entropy) or not np.isfinite(median_run_length):
        return -np.inf

    score = (
        3.0 * avg_self_transition
        + 1.5 * min_frac
        - 0.25 * median_run_length
        - 2.5 * avg_entropy
        + 0.05 * loglik_per_obs_per_feature
    )
    return float(score)


def automatic_hmm_feature_selection(
    df: pd.DataFrame,
    candidate_features: List[str],
    subset_min_size: int = 1,
    subset_max_size: int = 3,
    n_states_list: List[int] = [2, 3],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    correlation_filter_threshold: Optional[float] = 0.95,
    min_state_fraction_threshold: float = 0.05,
    top_k: Optional[int] = 20,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """Automatically search for strong HMM feature subsets."""
    if not candidate_features:
        raise ValueError("candidate_features is empty.")
    if not n_states_list:
        raise ValueError("n_states_list is empty.")
    if subset_min_size < 1:
        raise ValueError("subset_min_size must be >= 1.")
    if subset_max_size < subset_min_size:
        raise ValueError("subset_max_size must be >= subset_min_size.")
    if not (0.0 <= min_state_fraction_threshold < 1.0):
        raise ValueError("min_state_fraction_threshold must lie in [0, 1).")

    df_train, _, _ = make_time_splits(df, train_frac=train_frac, val_frac=val_frac)

    filtered_features = candidate_features.copy()
    if correlation_filter_threshold is not None:
        filtered_features = filter_high_correlation_features(
            df_train=df_train,
            feature_cols=filtered_features,
            threshold=correlation_filter_threshold
        )

    if not filtered_features:
        raise ValueError("No features left after correlation filtering.")

    max_size_eff = min(subset_max_size, len(filtered_features))
    if subset_min_size > max_size_eff:
        raise ValueError("subset_min_size is larger than available filtered features.")

    subsets = generate_feature_subsets(
        filtered_features,
        subset_min_size,
        max_size_eff
    )

    total = len(subsets) * len(n_states_list)
    progress = tqdm(total=total, disable=not verbose, desc="HMM feature search", unit="fit")

    rows = []
    best_score = -np.inf
    best_desc = "None"

    try:
        for subset in subsets:
            for n_states in n_states_list:
                try:
                    diag = evaluate_hmm_feature_subset(
                        df_train=df_train,
                        feature_cols=subset,
                        n_states=n_states,
                        covariance_type=covariance_type,
                        n_iter=n_iter,
                        random_state=random_state
                    )

                    diag["selection_score"] = simple_hmm_selection_score(
                        pd.Series(diag),
                        min_state_fraction_threshold=min_state_fraction_threshold
                    )
                    diag["eligible"] = bool(np.isfinite(diag["selection_score"]))
                    diag["status"] = "ok"
                    diag["error"] = None
                    rows.append(diag)

                    score = diag["selection_score"]
                    if np.isfinite(score) and score > best_score:
                        best_score = score
                        best_desc = (
                            f"{_format_feature_list_for_tqdm(diag['feature_cols'])} | "
                            f"S={diag['avg_self_transition']:.2f} | "
                            f"R={diag['median_run_length']:.1f} | "
                            f"E={diag['avg_entropy']:.3f}"
                        )

                except Exception as e:
                    rows.append({
                        "feature_cols": subset,
                        "n_states": n_states,
                        "n_features": len(subset),
                        "n_obs_used": np.nan,
                        "converged": False,
                        "train_loglik": np.nan,
                        "loglik_per_obs": np.nan,
                        "loglik_per_obs_per_feature": np.nan,
                        "aic": np.nan,
                        "bic": np.nan,
                        "n_params": np.nan,
                        "avg_self_transition": np.nan,
                        "avg_entropy": np.nan,
                        "min_state_fraction": np.nan,
                        "max_state_fraction": np.nan,
                        "median_run_length": np.nan,
                        "mean_run_length": np.nan,
                        "n_runs": np.nan,
                        "transition_matrix": None,
                        "selection_score": -np.inf,
                        "eligible": False,
                        "status": "skipped",
                        "error": f"{type(e).__name__}: {e}"
                    })

                progress.update(1)
                if verbose:
                    if np.isfinite(best_score):
                        progress.set_postfix_str(f"best={best_score:.3f} | {best_desc}")
                    else:
                        progress.set_postfix_str("best=None")
    finally:
        progress.close()

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    ok_df = out[out["status"] == "ok"].copy()
    bad_df = out[out["status"] != "ok"].copy()

    if not ok_df.empty:
        ok_df = ok_df.sort_values(
            by=[
                "eligible",
                "selection_score",
                "avg_self_transition",
                "min_state_fraction",
                "median_run_length",
                "avg_entropy",
                "loglik_per_obs_per_feature",
            ],
            ascending=[False, False, False, False, False, True, False]
        ).reset_index(drop=True)

    if top_k is not None and top_k > 0:
        ok_df = ok_df.head(top_k)

    out = pd.concat([ok_df, bad_df], axis=0).reset_index(drop=True)
    return out


def extract_best_hmm_feature_subset(
    results_df: pd.DataFrame
) -> pd.DataFrame:
    """Extract the best HMM feature subset and corresponding number of states."""
    if results_df.empty:
        raise ValueError("results_df is empty.")

    ok_df = results_df[results_df["status"] == "ok"].copy()
    if ok_df.empty:
        raise ValueError("No successful rows in results_df.")

    eligible_df = ok_df[ok_df["eligible"] == True].copy()
    if eligible_df.empty:
        raise ValueError("No eligible rows found. All models failed the selection constraints.")

    best_row = eligible_df.iloc[[0]].copy()

    preferred_cols = [
        "feature_cols",
        "n_states",
        "n_features",
        "n_obs_used",
        "selection_score",
        "train_loglik",
        "loglik_per_obs",
        "loglik_per_obs_per_feature",
        "aic",
        "bic",
        "min_state_fraction",
        "avg_self_transition",
        "median_run_length",
        "avg_entropy",
        "eligible",
        "status"
    ]
    cols = [c for c in preferred_cols if c in best_row.columns]

    return best_row[cols].reset_index(drop=True)


def summarize_hmm_results(
    results_df: pd.DataFrame,
    top_n: int = 10,
    stringify_features: bool = False
) -> pd.DataFrame:
    """Return a compact summary view of the top HMM feature-selection results."""
    if results_df.empty:
        raise ValueError("results_df is empty.")

    ok_df = results_df[results_df["status"] == "ok"].copy()
    if ok_df.empty:
        raise ValueError("No successful rows in results_df.")

    cols = [
        "feature_cols",
        "n_states",
        "n_features",
        "n_obs_used",
        "eligible",
        "selection_score",
        "train_loglik",
        "loglik_per_obs",
        "loglik_per_obs_per_feature",
        "min_state_fraction",
        "avg_self_transition",
        "median_run_length",
        "avg_entropy",
        "aic",
        "bic",
    ]
    available_cols = [c for c in cols if c in ok_df.columns]

    summary_df = ok_df[available_cols].head(top_n).reset_index(drop=True)

    if stringify_features:
        summary_df["feature_cols"] = summary_df["feature_cols"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )

    return summary_df


def fit_best_hmm_from_results(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42
) -> Tuple[GaussianHMM, StandardScaler, List[str], int]:
    """Fit the best HMM found by automatic_hmm_feature_selection on the train split."""
    best_df = extract_best_hmm_feature_subset(results_df)
    best_feature_cols = best_df.iloc[0]["feature_cols"]
    best_n_states = int(best_df.iloc[0]["n_states"])

    df_train, _, _ = make_time_splits(df, train_frac=train_frac, val_frac=val_frac)

    hmm, scaler = fit_hmm(
        df_train=df_train,
        feature_cols=best_feature_cols,
        n_states=best_n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )

    return hmm, scaler, best_feature_cols, best_n_states


from typing import Dict, Tuple, Optional, List
import pandas as pd


from typing import Dict, Tuple, Optional, List
import pandas as pd


def assign_regimes(
    df: pd.DataFrame,
    vol_col: Optional[str] = None,
    activity_col: Optional[str] = None,
    trend_col: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """
    Assign semantic labels to HMM states from a state-profile dataframe.

    Works with either:
    - summary columns like 'abs_return_close_perp_mean'
    - raw columns like 'abs_return_close_perp'
    """

    df = _flatten_columns_if_needed(df).copy()

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    def _pick_column(user_col: Optional[str], candidates: List[str], role: str) -> str:
        if user_col is not None:
            if user_col not in df.columns:
                raise ValueError(
                    f"Requested {role} column '{user_col}' not found. "
                    f"Available columns: {list(df.columns)}"
                )
            return user_col

        for col in candidates:
            if col in df.columns:
                return col

        raise ValueError(
            f"Could not infer {role} column. "
            f"Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    vol_col = _pick_column(
        vol_col,
        candidates=[
            "abs_return_close_perp_mean",
            "abs_return_close_perp",
            "std_24h_return_close_perp_mean",
            "std_24h_return_close_perp",
            "std_24h_abs_return_close_perp_mean",
            "std_24h_abs_return_close_perp",
            "ATR_24h_perp_mean",
            "ATR_24h_perp",
            "ATR_72h_perp_mean",
            "ATR_72h_perp",
            "abs_cost_perp_mean",
            "abs_cost_perp",
        ],
        role="volatility",
    )

    activity_col = _pick_column(
        activity_col,
        candidates=[
            "volume_perp_mean",
            "volume_perp",
            "log_volume_perp_mean",
            "log_volume_perp",
            "ma_24h_volume_perp_mean",
            "ma_24h_volume_perp",
            "z_24h_volume_perp_mean",
            "z_24h_volume_perp",
            "cost_x_volume_perp_mean",
            "cost_x_volume_perp",
            "abs_cost_perp_mean",
            "abs_cost_perp",
        ],
        role="activity",
    )

    if trend_col is None:
        for c in [
            "return_close_perp_mean",
            "return_close_perp",
            "diff_1h_cost_perp_mean",
            "diff_1h_cost_perp",
            "close_perp_mean",
            "close_perp",
        ]:
            if c in df.columns:
                trend_col = c
                break

    needed_cols = [vol_col, activity_col]
    if df[needed_cols].isna().any().any():
        raise ValueError(f"Regime interpretation columns contain NaN values: {needed_cols}")

    work = df.copy()
    work["_vol_rank"] = work[vol_col].rank(method="first", ascending=True)
    work["_act_rank"] = work[activity_col].rank(method="first", ascending=True)
    work["_regime_score"] = work["_vol_rank"] + work["_act_rank"]

    ranked_states = work["_regime_score"].sort_values().index.tolist()

    regime_labels: Dict = {}
    regime_colors: Dict = {}
    n = len(ranked_states)

    if n == 1:
        regime_labels[ranked_states[0]] = "Active"
        regime_colors[ranked_states[0]] = "lightgreen"
        return regime_labels, regime_colors

    if n == 2:
        regime_labels[ranked_states[0]] = "Low Activity"
        regime_colors[ranked_states[0]] = "lightgrey"
        regime_labels[ranked_states[1]] = "Stress"
        regime_colors[ranked_states[1]] = "lightcoral"
        return regime_labels, regime_colors

    if n == 3:
        low_state = ranked_states[0]
        mid_state = ranked_states[1]
        high_state = ranked_states[2]

        regime_labels[low_state] = "Low Activity"
        regime_colors[low_state] = "lightgrey"

        regime_labels[high_state] = "Stress"
        regime_colors[high_state] = "lightcoral"

        mid_label = "Active"
        if trend_col is not None and trend_col in work.columns:
            mid_label = "Active / Trend"

        regime_labels[mid_state] = mid_label
        regime_colors[mid_state] = "lightgreen"
        return regime_labels, regime_colors

    for i, state in enumerate(ranked_states):
        if i == 0:
            regime_labels[state] = "Low Activity"
            regime_colors[state] = "lightgrey"
        elif i == n - 1:
            regime_labels[state] = "Stress"
            regime_colors[state] = "lightcoral"
        else:
            regime_labels[state] = f"Active {i}"
            regime_colors[state] = "lightgreen"

    return regime_labels, regime_colors


def fit_hmm_from_results_index(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    selected_idx: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    covariance_type: str = "full",
    n_iter: int = 200,
    random_state: int = 42
) -> Tuple[GaussianHMM, StandardScaler, List[str], int]:
    """Fit an HMM from a user-selected row index in results_df."""
    if results_df.empty:
        raise ValueError("results_df is empty.")

    if selected_idx not in results_df.index:
        raise ValueError(
            f"selected_idx={selected_idx} not found in results_df.index. "
            f"Available indices: {list(results_df.index)}"
        )

    row = results_df.loc[selected_idx]

    if row.get("status", None) != "ok":
        raise ValueError(
            f"Row {selected_idx} has status={row.get('status')} and cannot be fitted."
        )

    feature_cols = row["feature_cols"]
    n_states = int(row["n_states"])

    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise ValueError(f"Row {selected_idx} has invalid feature_cols: {feature_cols}")

    df_train, _, _ = make_time_splits(df, train_frac=train_frac, val_frac=val_frac)

    hmm, scaler = fit_hmm(
        df_train=df_train,
        feature_cols=feature_cols,
        n_states=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )

    return hmm, scaler, feature_cols, n_states


__all__ = [
    "add_hmm_features",
    "assign_regimes",
    "automatic_hmm_feature_selection",
    "build_candidate_features",
    "clean_feature_frame",
    "compute_entropy",
    "compute_run_lengths",
    "evaluate_hmm_feature_subset",
    "extract_best_hmm_feature_subset",
    "filter_high_correlation_features",
    "fit_best_hmm_from_results",
    "fit_hmm",
    "fit_hmm_from_results_index",
    "generate_feature_subsets",
    "load_dataset",
    "make_time_splits",
    "plot_full_regime_overlay",
    "plot_recent_regimes",
    "resolve_hmm_columns",
    "simple_hmm_selection_score",
    "summarize_hmm_results",
    "summarize_state_profile",
]
