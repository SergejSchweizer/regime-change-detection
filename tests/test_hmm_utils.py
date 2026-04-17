from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import hmm_utils as hu


@pytest.fixture
def hmm_feature_cols() -> list[str]:
    return ["close_perp", "volume_perp", "abs_cost_perp"]


def test_make_time_splits_and_validation(hmm_ready_df):
    train, val, test = hu.make_time_splits(hmm_ready_df, train_frac=0.5, val_frac=0.25)

    assert len(train) + len(val) + len(test) == len(hmm_ready_df)
    with pytest.raises(ValueError):
        hu.make_time_splits(hmm_ready_df.iloc[:2], train_frac=0.9, val_frac=0.09)


def test_generate_feature_subsets_and_validation():
    subsets = hu.generate_feature_subsets(["a", "b", "c"], min_size=1, max_size=2)
    assert subsets == [["a"], ["b"], ["c"], ["a", "b"], ["a", "c"], ["b", "c"]]
    with pytest.raises(ValueError):
        hu.generate_feature_subsets(["a"], min_size=2, max_size=1)


def test_clean_feature_frame_replaces_infinities(hmm_ready_df):
    df = hmm_ready_df.copy()
    df.loc[df.index[0], "close_perp"] = np.inf

    cleaned = hu.clean_feature_frame(df, ["close_perp", "volume_perp"])

    assert np.isnan(cleaned.iloc[0, 0])
    with pytest.raises(ValueError):
        hu.clean_feature_frame(df, ["missing"])


def test_compute_entropy_validation_and_values():
    entropy = hu.compute_entropy(np.array([[0.5, 0.5], [1.0, 0.0]]))
    assert entropy[0] > entropy[1]
    with pytest.raises(ValueError):
        hu.compute_entropy(np.array([0.5, 0.5]))


def test_filter_high_correlation_features_removes_redundant_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6], "c": [1, 1, 2]})
    filtered = hu.filter_high_correlation_features(df, ["a", "b", "c"], threshold=0.9)
    assert filtered == ["a", "c"]
    with pytest.raises(ValueError):
        hu.filter_high_correlation_features(df, [], threshold=0.9)


def test_private_helpers_cover_column_flatten_and_format():
    multi = pd.DataFrame([[1, 2]], columns=pd.MultiIndex.from_tuples([("x", "a"), ("x", "b")]))
    flat = hu._flatten_columns_if_needed(multi)
    assert list(flat.columns) == ["x_a", "x_b"]
    assert hu._format_feature_list_for_tqdm(["a", "b"], max_len=10) == "a,b"
    assert hu._format_feature_list_for_tqdm(["abcdef", "ghijkl"], max_len=10).endswith("...")


def test_load_dataset_handles_iso_and_ms_timestamps(tmp_path):
    iso_path = tmp_path / "iso.csv"
    pd.DataFrame({"timestamp": ["2024-01-01T00:00:00Z"], "value": [1]}).to_csv(iso_path, index=False)
    iso_df = hu.load_dataset(iso_path)
    assert iso_df.index.name == "timestamp"

    ms_path = tmp_path / "ms.csv"
    pd.DataFrame({"timestamp": [1_700_000_000_000], "value": [2]}).to_csv(ms_path, index=False)
    ms_df = hu.load_dataset(ms_path)
    assert ms_df.iloc[0]["value"] == 2


def test_build_candidate_features_and_resolve_columns(hmm_ready_df):
    blocks = {"main": ["close_perp", "missing", "volume_perp"], "other": ["close_perp", "abs_cost_perp"]}
    features, missing = hu.build_candidate_features(hmm_ready_df, blocks)

    assert features == ["close_perp", "volume_perp", "abs_cost_perp"]
    assert missing == {"main": ["missing"]}

    df = pd.DataFrame(columns=["hmm_prob_2", "hmm_prob_0", "hmm_state", "hmm_prob_1"])
    state_col, prob_cols = hu.resolve_hmm_columns(df)
    assert state_col == "hmm_state"
    assert prob_cols == ["hmm_prob_0", "hmm_prob_1", "hmm_prob_2"]


def test_compute_run_lengths_and_state_profile(hmm_ready_df):
    states = pd.Series([0, 0, 1, 1, 1, 0])
    run_lengths = hu.compute_run_lengths(states)
    assert run_lengths.tolist() == [2.0, 3.0, 1.0]

    df = hmm_ready_df.iloc[:10].copy()
    df["hmm_state"] = [0, 0, 1, 1, 0, 0, 1, 1, 0, 1]
    profile = hu.summarize_state_profile(df, "hmm_state", ["abs_cost_perp"])
    assert "close_perp" in profile.columns
    assert "abs_cost_perp" in profile.columns


def test_plot_recent_and_full_regime_overlay(monkeypatch, hmm_ready_df):
    monkeypatch.setattr(hu.plt, "show", lambda: None)
    df = hmm_ready_df.iloc[:20].copy()
    df["hmm_state"] = [0, 0, 1, 1, 2] * 4
    df["hmm_prob_0"] = np.where(df["hmm_state"] == 0, 0.9, 0.05)
    df["hmm_prob_1"] = np.where(df["hmm_state"] == 1, 0.9, 0.05)
    df["hmm_prob_2"] = np.where(df["hmm_state"] == 2, 0.9, 0.05)

    hu.plot_recent_regimes(
        df=df,
        price_col="close_perp",
        state_col="hmm_state",
        prob_cols=["hmm_prob_0", "hmm_prob_1", "hmm_prob_2"],
        regime_labels={0: "Low", 1: "Mid", 2: "High"},
        regime_colors={0: "grey", 1: "green", 2: "red"},
        n_points=10,
    )
    hu.plot_full_regime_overlay(
        df=df,
        price_col="close_perp",
        state_col="hmm_state",
        regime_labels={0: "Low", 1: "Mid", 2: "High"},
        regime_colors={0: "grey", 1: "green", 2: "red"},
    )


def test_fit_hmm_and_add_hmm_features(hmm_ready_df, hmm_feature_cols):
    hmm, scaler = hu.fit_hmm(
        hmm_ready_df,
        feature_cols=hmm_feature_cols,
        n_states=2,
        covariance_type="diag",
        n_iter=20,
        random_state=7,
    )
    out = hu.add_hmm_features(hmm_ready_df, hmm, scaler, hmm_feature_cols)

    assert {"hmm_state", "hmm_prob_0", "hmm_prob_1", "hmm_max_prob", "hmm_entropy"}.issubset(out.columns)
    assert out["hmm_state"].notna().any()


def test_private_state_helpers():
    stats = hu._compute_state_sequence_stats(np.array([0, 0, 1, 1, 1]), n_states=2)
    assert stats["median_run_length"] == 2.5
    assert hu._count_hmm_parameters(2, 3, "diag") == 15
    with pytest.raises(ValueError):
        hu._count_hmm_parameters(2, 3, "bad")
    assert hu._safe_float("1.5") == 1.5
    assert np.isnan(hu._safe_float("bad"))


def test_evaluate_hmm_feature_subset_and_score(hmm_ready_df, hmm_feature_cols):
    diag = hu.evaluate_hmm_feature_subset(
        hmm_ready_df,
        feature_cols=hmm_feature_cols,
        n_states=2,
        covariance_type="diag",
        n_iter=20,
        random_state=7,
    )
    assert diag["n_features"] == 3
    assert diag["transition_matrix"].shape == (2, 2)

    row = pd.Series(diag)
    score = hu.simple_hmm_selection_score(row)
    assert np.isfinite(score)
    row["converged"] = False
    assert hu.simple_hmm_selection_score(row) == -np.inf


def test_automatic_hmm_feature_selection_and_result_helpers(hmm_ready_df):
    candidate_features = ["close_perp", "volume_perp", "abs_cost_perp"]
    results = hu.automatic_hmm_feature_selection(
        hmm_ready_df,
        candidate_features=candidate_features,
        subset_min_size=2,
        subset_max_size=2,
        n_states_list=[2],
        correlation_filter_threshold=None,
        top_k=10,
        covariance_type="diag",
        n_iter=20,
        random_state=7,
        verbose=False,
    )

    assert not results.empty
    assert {"feature_cols", "selection_score", "status"}.issubset(results.columns)

    best = hu.extract_best_hmm_feature_subset(results)
    summary = hu.summarize_hmm_results(results, top_n=5, stringify_features=True)
    assert len(best) == 1
    assert isinstance(summary.iloc[0]["feature_cols"], str)


def test_fit_best_hmm_and_fit_from_results_index(hmm_ready_df):
    results = hu.automatic_hmm_feature_selection(
        hmm_ready_df,
        candidate_features=["close_perp", "volume_perp", "abs_cost_perp"],
        subset_min_size=2,
        subset_max_size=2,
        n_states_list=[2],
        correlation_filter_threshold=None,
        top_k=10,
        covariance_type="diag",
        n_iter=20,
        random_state=7,
        verbose=False,
    )

    hmm, scaler, features, states = hu.fit_best_hmm_from_results(
        hmm_ready_df,
        results,
        covariance_type="diag",
        n_iter=20,
        random_state=7,
    )
    assert states == 2
    assert len(features) == 2

    hmm2, scaler2, features2, states2 = hu.fit_hmm_from_results_index(
        hmm_ready_df,
        results,
        selected_idx=0,
        covariance_type="diag",
        n_iter=20,
        random_state=7,
    )
    assert states2 == 2
    assert features2 == features
    assert type(hmm2) is type(hmm)
    assert type(scaler2) is type(scaler)

    with pytest.raises(ValueError):
        hu.fit_hmm_from_results_index(hmm_ready_df, results, selected_idx=999)


def test_assign_regimes_covers_inference_and_edge_cases():
    df3 = pd.DataFrame(
        {
            "abs_return_close_perp": [0.10, 0.02, 0.03],
            "volume_perp": [500, 100, 200],
            "return_close_perp": [0.01, 0.02, 0.03],
        },
        index=[0, 1, 2],
    )
    labels3, colors3 = hu.assign_regimes(df3)
    assert labels3[0] == "Stress"
    assert "Low Activity" in labels3.values()

    df2 = pd.DataFrame({"abs_return_close_perp": [0.01, 0.03], "volume_perp": [10, 50]}, index=[0, 1])
    labels2, colors2 = hu.assign_regimes(df2)
    assert labels2[0] == "Low Activity"
    assert colors2[1] == "lightcoral"

    df1 = pd.DataFrame({"abs_return_close_perp": [0.01], "volume_perp": [10]}, index=[0])
    labels1, _ = hu.assign_regimes(df1)
    assert labels1[0] == "Active"

    with pytest.raises(ValueError):
        hu.assign_regimes(pd.DataFrame())

    with pytest.raises(ValueError):
        hu.assign_regimes(pd.DataFrame({"abs_return_close_perp": [np.nan], "volume_perp": [1]}))
