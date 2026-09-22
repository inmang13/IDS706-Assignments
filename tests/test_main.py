import numpy as np
import pandas as pd
import pytest

from src.main import (
    load_flow_data,
    clean_flow_data,
    summarize_flow,
    create_features,
    split_train_test,
    train_model,
    nash_sutcliffe_efficiency,
    evaluate_model,
    run_pipeline,
    plot_predictions,
    plot_yearly_stats,
    plot_flow_boxplot,
    plot_timeseries_predictions,
    DEFAULT_DATA_PATH,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def raw_flow_df():
    """Small raw frame shaped like the USGS CSV: dupes, a non-Approved
    row, and a string 'value' to exercise clean_flow_data's coercions."""
    return pd.DataFrame({
        "time": ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03"],
        "value": ["10.0", "20.0", "20.0", "30.0"],
        "approval_status": ["Approved", "Approved", "Approved", "Provisional"],
    })


@pytest.fixture
def clean_df():
    """150 days of clean data across a year boundary -- long enough that
    rows survive the rolling_mean_90 warm-up window in create_features."""
    dates = pd.date_range("2018-10-01", periods=150, freq="D")
    values = np.linspace(10, 100, len(dates))
    return pd.DataFrame({"time": dates, "value": values})


# ---------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------
def test_load_flow_data_reads_real_csv():
    df = load_flow_data(DEFAULT_DATA_PATH)
    assert {"time", "value", "approval_status"}.issubset(df.columns)
    assert len(df) > 0


def test_load_flow_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_flow_data("no_such_file.csv")


# ---------------------------------------------------------------------
# 2. Preprocessing / transformation
# ---------------------------------------------------------------------
def test_clean_flow_data_filters_types_and_dedupes(raw_flow_df):
    cleaned = clean_flow_data(raw_flow_df)

    # non-Approved row dropped
    assert len(cleaned) == 2
    # duplicate 2020-01-02 row collapsed to one
    assert cleaned["time"].is_unique
    # types coerced
    assert pd.api.types.is_datetime64_any_dtype(cleaned["time"])
    assert pd.api.types.is_numeric_dtype(cleaned["value"])
    # sorted ascending
    assert cleaned["time"].is_monotonic_increasing


def test_clean_flow_data_empty_input_returns_empty_frame():
    empty = pd.DataFrame(columns=["time", "value", "approval_status"])
    cleaned = clean_flow_data(empty)
    assert len(cleaned) == 0
    assert list(cleaned.columns) == ["time", "value"]


# ---------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------
def test_create_features_target_and_no_leakage(clean_df):
    feat = create_features(clean_df)

    # target_flow on row N should equal value on row N+1 -- check by
    # rejoining on the original clean_df via time
    merged = feat.merge(
        clean_df.rename(columns={"value": "next_day_value"}),
        left_on=feat["time"] + pd.Timedelta(days=1),
        right_on="time",
        suffixes=("", "_next"),
    )
    assert np.allclose(merged["target_flow"], merged["next_day_value"])

    # lag_1 on row N should equal value on row N-1
    merged_lag = feat.merge(
        clean_df.rename(columns={"value": "prev_day_value"}),
        left_on=feat["time"] - pd.Timedelta(days=1),
        right_on="time",
        suffixes=("", "_prev"),
    )
    assert np.allclose(merged_lag["lag_1"], merged_lag["prev_day_value"])

    # rolling_mean_7 must be built on shift(1): it should NOT equal a
    # rolling mean that includes the current day's value
    naive_rolling_mean_7 = clean_df.set_index("time")["value"].rolling(7).mean()
    feat_indexed = feat.set_index("time")["rolling_mean_7"].dropna()
    common_idx = feat_indexed.index.intersection(naive_rolling_mean_7.dropna().index)
    assert not np.allclose(
        feat_indexed.loc[common_idx], naive_rolling_mean_7.loc[common_idx]
    )

    # warm-up NaNs from shifts/rolling windows are dropped
    assert not feat.isna().any().any()


# ---------------------------------------------------------------------
# 4. Model training / prediction / evaluation
# ---------------------------------------------------------------------
def test_split_train_test_no_leakage_and_chronological(clean_df):
    feat = create_features(clean_df)
    split_date = feat["time"].iloc[len(feat) // 2]

    X_train, X_test, y_train, y_test = split_train_test(feat, split_date=split_date)

    # no train timestamp is >= split_date
    assert (X_train.index < split_date).all()
    assert (X_test.index >= split_date).all()
    # leak-prone columns absent from X
    assert "value" not in X_train.columns
    assert "target_flow" not in X_train.columns


def test_split_train_test_split_past_end_gives_empty_test(clean_df):
    feat = create_features(clean_df)
    far_future = feat["time"].max() + pd.Timedelta(days=365)

    _, X_test, _, y_test = split_train_test(feat, split_date=far_future)

    assert len(X_test) == 0
    assert len(y_test) == 0


def test_nash_sutcliffe_efficiency_perfect_and_mean_baseline():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])

    assert nash_sutcliffe_efficiency(y_true, y_true) == pytest.approx(1.0)

    mean_pred = np.full_like(y_true, y_true.mean())
    assert nash_sutcliffe_efficiency(y_true, mean_pred) == pytest.approx(0.0)


def test_evaluate_model_returns_finite_metrics(clean_df):
    feat = create_features(clean_df)
    split_date = feat["time"].iloc[len(feat) // 2]
    X_train, X_test, y_train, y_test = split_train_test(feat, split_date=split_date)

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test)

    metrics = evaluate_model(model, X_test, y_test)
    for key in ("RMSE", "NSE", "NSE_log"):
        assert key in metrics
        assert np.isfinite(metrics[key])


# ---------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------
def test_plot_predictions_saves_file(tmp_path, clean_df):
    feat = create_features(clean_df)
    split_date = feat["time"].iloc[len(feat) // 2]
    X_train, X_test, y_train, y_test = split_train_test(feat, split_date=split_date)
    model = train_model(X_train, y_train)

    out_path = plot_predictions(y_test, model.predict(X_test), output_dir=tmp_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_yearly_stats_saves_file(tmp_path, clean_df):
    flow_stats = summarize_flow(clean_df)

    out_path = plot_yearly_stats(flow_stats["yearly"], output_dir=tmp_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_flow_boxplot_saves_file(tmp_path, clean_df):
    flow_stats = summarize_flow(clean_df)

    out_path = plot_flow_boxplot(flow_stats["calendar_df"], "month", output_dir=tmp_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_timeseries_predictions_saves_file(tmp_path, clean_df):
    feat = create_features(clean_df)
    split_date = feat["time"].iloc[len(feat) // 2]
    X_train, X_test, y_train, y_test = split_train_test(feat, split_date=split_date)
    model = train_model(X_train, y_train)

    out_path = plot_timeseries_predictions(
        y_test.index, y_test, model.predict(X_test), output_dir=tmp_path
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0


# ---------------------------------------------------------------------
# 6. System test
# ---------------------------------------------------------------------
def test_run_pipeline_end_to_end_beats_mean_baseline():
    metrics = run_pipeline()

    assert metrics["NSE"] > 0
    assert np.isfinite(metrics["RMSE"])
    assert np.isfinite(metrics["NSE_log"])
