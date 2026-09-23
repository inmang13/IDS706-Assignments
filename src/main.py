"""
Eno River streamflow analysis pipeline.

USGS monitoring location 02085000 (Eno River at Hillsborough, NC),
daily mean discharge in ft^3/s, 2010-2020.

Pipeline outline
----------------
    load_flow_data      read the raw CSV off disk
    clean_flow_data     filter, type-convert, de-duplicate
    summarize_flow      groupby summaries for the report
    create_features     lags, rolling stats, calendar features
    split_train_test    chronological split (NOT random - time series)
    train_model         fit the regressor
    nash_sutcliffe_efficiency / evaluate_model
    run_pipeline        end-to-end, used by the system test

Each function below is a stub. Fill in the bodies by moving the working
code out of notebooks/basic_analysis.ipynb, then point the notebook at
these functions instead of redefining them inline.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402


# Repo-relative default so this works locally AND in CI.
# (Do not hardcode C:/Users/... - GitHub Actions has no such path.)
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "daily-data-mean.csv"


# ---------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------
def load_flow_data(path=DEFAULT_DATA_PATH):
    """Read the USGS CSV and return a raw DataFrame."""

    flow = pd.read_csv(path)

    return flow


# ---------------------------------------------------------------------
# 2. Preprocessing / transformation
# ---------------------------------------------------------------------
def clean_flow_data(df):
    """Filter to usable rows and coerce types."""

    df = df[df["approval_status"] == "Approved"].copy()
    df["time"] = pd.to_datetime(df["time"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["time", "value"]]

    n_invalid = df["value"].isna().sum()
    n_duplicate_timestamps = df.duplicated(subset=["time"]).sum()

    df = df.dropna(subset=["time", "value"])
    df = df.drop_duplicates(subset=["time"], keep="last")
    df = df.sort_values("time").reset_index(drop=True)

    if df.empty:
        n_missing_days = 0
    else:
        expected_days = pd.date_range(df["time"].min(), df["time"].max(), freq="D")
        n_missing_days = len(expected_days.difference(df["time"]))

    print(
        f"clean_flow_data: {n_invalid} invalid value(s), "
        f"{n_duplicate_timestamps} duplicate timestamp(s), "
        f"{n_missing_days} missing day(s)"
    )

    if n_missing_days:
        raise ValueError(
            "clean_flow_data requires consecutive daily observations; "
            f"found {n_missing_days} missing calendar day(s)"
        )

    return df


def summarize_flow(df):
    """Groupby summaries: mean/max/min by year, by month, by season."""
    df = df.copy()

    start_date = df["time"].min()
    end_date = df["time"].max()
    duration = end_date - start_date
    max_flow = df["value"].max()
    min_flow = df["value"].min()
    mean_flow = df["value"].mean()

    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["season"] = df["month"].map(
        {
            12: "Winter",
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Fall",
            10: "Fall",
            11: "Fall",
        }
    )

    monthly_stats = df.groupby("month")["value"].agg(["mean", "max", "min"])
    seasonal_stats = df.groupby("season")["value"].agg(["mean", "max", "min"])
    yearly_stats = df.groupby("year")["value"].agg(["mean", "max", "min"])

    print(
        f"summarize_flow: {start_date.date()} to {end_date.date()} "
        f"({duration.days} days), mean {mean_flow:.1f}, "
        f"range [{min_flow:.1f}, {max_flow:.1f}] ft^3/s"
    )

    return {
        "overall": {
            "start_date": start_date,
            "end_date": end_date,
            "duration": duration,
            "mean": mean_flow,
            "max": max_flow,
            "min": min_flow,
        },
        "yearly": yearly_stats,
        "monthly": monthly_stats,
        "seasonal": seasonal_stats,
        "calendar_df": df,
    }


# ---------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------
def create_features(df, target_col="value"):
    """Build the model matrix: target, lags, rolling stats, calendar.
    Remember the leakage rule: every rolling stat is computed on
    df[target_col].shift(1), never on the current day.
    """
    np.random.seed(42)
    df = df.set_index("time")
    df_feat = df[[target_col]].copy()

    # Target: Flow 1-day into the future
    df_feat["target_flow"] = df_feat[target_col].shift(-1)
    df_feat["target_time"] = pd.Series(df_feat.index, index=df_feat.index).shift(-1)

    # 2. Lags and rolling statistics (short and seasonal memory)
    lagged = df_feat[target_col].shift(1)
    for lag in (1, 2, 3, 5, 7):
        df_feat[f"lag_{lag}"] = df_feat[target_col].shift(lag)
    df_feat["flow_diff_1"] = df_feat[target_col].diff(1)
    df_feat["rolling_sum_3"] = lagged.rolling(window=3).sum()
    df_feat["rolling_mean_7"] = lagged.rolling(window=7).mean()
    df_feat["rolling_std_7"] = lagged.rolling(window=7).std()
    df_feat["rolling_mean_30"] = lagged.rolling(window=30).mean()
    df_feat["rolling_mean_90"] = lagged.rolling(window=90).mean()
    df_feat["rolling_min_30"] = lagged.rolling(window=30).min()
    df_feat["rolling_max_30"] = lagged.rolling(window=30).max()

    # 3. Calendar features and seasonal cycles
    df_feat["day_of_year"] = df_feat.index.dayofyear
    df_feat["day_of_week"] = df_feat.index.dayofweek
    df_feat["month"] = df_feat.index.month
    df_feat["day_of_year_sin"] = np.sin(2 * np.pi * df_feat["day_of_year"] / 365.25)
    df_feat["day_of_year_cos"] = np.cos(2 * np.pi * df_feat["day_of_year"] / 365.25)

    # Drop rows with NaN values created by shifts/rolling windows
    df_feat = df_feat.dropna()

    return df_feat.reset_index()


# ---------------------------------------------------------------------
# 4. Model training / prediction / evaluation
# ---------------------------------------------------------------------
def split_train_test(df, split_date=None, target_col="target_flow"):
    """Chronological split, ~80/20, snapped to the nearest Jan 1.

    If split_date is not given, pick the Jan 1 closest to the 80% mark
    of the time range so the split lands on a year boundary instead of
    mid-year. Pass split_date explicitly to override.
    """
    df = df.sort_values("time").set_index("time")

    if split_date is None:
        target_idx = int(len(df) * 0.8)
        candidate = df.index[min(target_idx, len(df) - 1)]
        jan1_this_year = pd.Timestamp(year=candidate.year, month=1, day=1)
        jan1_next_year = pd.Timestamp(year=candidate.year + 1, month=1, day=1)
        split_date = (
            jan1_this_year
            if (candidate - jan1_this_year) <= (jan1_next_year - candidate)
            else jan1_next_year
        )
    else:
        split_date = pd.Timestamp(split_date)

    train = df[(df.index < split_date) & (df["target_time"] < split_date)]
    test = df[df.index >= split_date]

    print(
        f"split_train_test: splitting at {split_date.date()} "
        f"({len(train)} train rows, {len(test)} test rows)"
    )

    feature_cols = [
        c for c in df.columns if c not in (target_col, "target_time", "value")
    ]

    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    return X_train, X_test, y_train, y_test


class _Log1pRegressor:
    """Wraps a regressor fit on log1p(y); .predict() inverts back to raw units.

    Streamflow is heavy-tailed (a few flood days dominate raw squared
    error), so fitting on log1p(flow) makes the tree splits optimize for
    proportional error instead of being dominated by flood-day magnitude.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return np.expm1(self.model.predict(X))

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


def train_model(X_train, y_train, random_state=42):
    """Fit an RF on log1p(y_train) and return a wrapper whose .predict()
    returns raw-scale flow. Set the seed so tests are repeatable."""
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    rf_model.fit(X_train, np.log1p(y_train))

    return _Log1pRegressor(rf_model)


def nash_sutcliffe_efficiency(y_true, y_pred):
    """NSE: 1.0 is perfect, 0.0 is no better than predicting the mean."""
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if np.isclose(denominator, 0):
        if np.allclose(y_true, y_pred):
            return 1.0
        raise ValueError("NSE is undefined when observations have zero variance")
    return 1 - (np.sum((y_true - y_pred) ** 2) / denominator)


def evaluate_model(model, X_test, y_test):
    """Predict and return a dict of metrics (RMSE, NSE)."""
    y_pred = model.predict(X_test)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    nse = nash_sutcliffe_efficiency(y_test, y_pred)
    nse_log = nash_sutcliffe_efficiency(np.log1p(y_test), np.log1p(y_pred))

    return {"RMSE": rmse, "NSE": nse, "NSE_log": nse_log}


def feature_importance(model, feature_names):
    """Return the model's feature importances as a Series, most important first."""
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )


# ---------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


def plot_predictions(
    y_true, y_pred, title="Predicted vs Actual Flow", output_dir=OUTPUT_DIR, show=False
):
    """Scatter plot of predicted vs actual flow values.

    Saves the figure to output_dir/predicted_vs_actual.png and returns
    that path. show=False by default so this is safe to call in CI
    (headless runners have no display).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    ax.set_xlabel("Actual Flow (ft^3/s)")
    ax.set_ylabel("Predicted Flow (ft^3/s)")
    ax.set_title(title)
    ax.grid()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "predicted_vs_actual.png"
    fig.savefig(out_path)

    if show:
        plt.show()
    plt.close(fig)

    return out_path


def plot_yearly_stats(
    yearly_stats,
    title="Yearly Streamflow Statistics",
    output_dir=OUTPUT_DIR,
    show=False,
):
    """Line plot of mean/max/min flow by year (log-scaled y).

    yearly_stats: DataFrame indexed by year with 'mean', 'max', 'min'
    columns, e.g. summarize_flow(df)['yearly'].

    Saves the figure to output_dir/yearly_stats.png and returns that path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(yearly_stats.index, yearly_stats["mean"], marker="o", label="Mean")
    ax.plot(yearly_stats.index, yearly_stats["max"], marker="s", label="Max")
    ax.plot(yearly_stats.index, yearly_stats["min"], marker="^", label="Min")
    ax.set_xlabel("Year")
    ax.set_ylabel("Flow (ft^3/s)")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "yearly_stats.png"
    fig.savefig(out_path)

    if show:
        plt.show()
    plt.close(fig)

    return out_path


def plot_flow_boxplot(
    df,
    group_col,
    order=None,
    title=None,
    xlabel=None,
    color="#6baed6",
    edgecolor="#2171b5",
    output_dir=OUTPUT_DIR,
    show=False,
):
    """Boxplot of 'value' grouped by group_col (e.g. 'year', 'month', 'season').

    df must already have group_col as a column (see clean_flow_data /
    summarize_flow for 'month' and 'season'). order controls the x-axis
    order and labels; defaults to the sorted unique group values.

    Saves the figure to output_dir/boxplot_by_<group_col>.png and returns
    that path.
    """
    if order is None:
        order = sorted(df[group_col].dropna().unique())

    data = [df.loc[df[group_col] == group, "value"] for group in order]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(
        data,
        tick_labels=list(order),
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        boxprops={"facecolor": color, "edgecolor": edgecolor},
        whiskerprops={"color": edgecolor},
        capprops={"color": edgecolor},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.4},
    )
    ax.set_yscale("log")
    ax.set_title(
        title or f"Daily Streamflow Distribution by {group_col.title()}",
        fontsize=15,
        weight="bold",
    )
    ax.set_xlabel(xlabel or group_col.title())
    ax.set_ylabel("Flow (ft^3/s)")
    ax.grid(axis="y", linestyle="--", alpha=0.4, which="both")
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"boxplot_by_{group_col}.png"
    fig.savefig(out_path)

    if show:
        plt.show()
    plt.close(fig)

    return out_path


def plot_timeseries_predictions(
    dates,
    y_true,
    y_pred,
    title="Observed vs. Model Predictions",
    output_dir=OUTPUT_DIR,
    show=False,
):
    """Line plot of observed vs. predicted streamflow over time (log-scaled y).

    Saves the figure to output_dir/timeseries_predictions.png and returns
    that path. show=False by default so this is safe to call in CI.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y_true, label="Observed Streamflow", color="black", alpha=0.7)
    ax.plot(
        dates,
        y_pred,
        label="Random Forest Prediction",
        color="royalblue",
        linestyle=":",
        linewidth=2,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow (cfs)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "timeseries_predictions.png"
    fig.savefig(out_path)

    if show:
        plt.show()
    plt.close(fig)

    return out_path


# ---------------------------------------------------------------------
# 6. End-to-end (this is what the system test calls)
# ---------------------------------------------------------------------
def run_pipeline(path=DEFAULT_DATA_PATH, split_date=None):
    """Run load -> clean -> features -> split -> train -> evaluate.

    Return the metrics dict so the system test can assert on it.
    """
    raw_flow_df = load_flow_data(path)
    clean_flow_df = clean_flow_data(raw_flow_df)
    flow_stats = summarize_flow(clean_flow_df)
    features_df = create_features(clean_flow_df)
    X_train, X_test, y_train, y_test = split_train_test(
        features_df, split_date=split_date
    )
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    y_pred = model.predict(X_test)

    calendar_df = flow_stats["calendar_df"]
    plot_yearly_stats(flow_stats["yearly"])
    plot_flow_boxplot(calendar_df, "year")
    plot_flow_boxplot(calendar_df, "month")
    plot_flow_boxplot(
        calendar_df,
        "season",
        order=["Winter", "Spring", "Summer", "Fall"],
        color="#59a14f",
        edgecolor="#2f6b2f",
    )
    plot_predictions(y_test, y_pred)
    plot_timeseries_predictions(y_test.index, y_test, y_pred)

    print(
        f"run_pipeline: RMSE={metrics['RMSE']:.2f}, "
        f"NSE={metrics['NSE']:.3f}, NSE_log={metrics['NSE_log']:.3f}"
    )

    importances = feature_importance(model, X_train.columns)
    print("Top 10 features:")
    print(importances.head(5).to_string())

    return metrics


if __name__ == "__main__":
    metrics = run_pipeline()
    print("Pipeline metrics:", metrics)
