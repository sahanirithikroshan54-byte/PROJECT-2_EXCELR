"""Leakage-safe bike rental demand forecasting pipeline.

The target is total hourly rentals (`cnt`). Casual/registered rider counts,
future weather, and same-hour target values are never model inputs.
"""
from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "cnt"
FORECAST_HOURS_PER_DAY = 24
MAX_FORECAST_DAYS = 30

FEATURE_COLUMNS = [
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
    "dayofyear_sin", "dayofyear_cos",
    "is_weekend", "days_since_start",
    "lag_1", "lag_24", "lag_168",
    "rolling_mean_24", "rolling_mean_168",
]


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and standardize the supplied UCI-style hourly bike rental CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Keep Dataset.csv beside app.py."
        )

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    required = {"dteday", "hr", "cnt"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )

    df = df.replace({"?": np.nan, "": np.nan})
    df["dteday"] = pd.to_datetime(
        df["dteday"], dayfirst=True, errors="coerce"
    )
    for col in ["hr", "cnt", "instant", "weekday", "temp", "atemp",
                "hum", "windspeed", "casual", "registered", "mnth", "yr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["dteday", "hr", "cnt"]).copy()
    df["hr"] = df["hr"].astype(int)
    invalid_hours = ~df["hr"].between(0, 23)
    if invalid_hours.any():
        raise ValueError("The `hr` column must contain values from 0 to 23.")

    df["timestamp"] = (
        df["dteday"].dt.normalize()
        + pd.to_timedelta(df["hr"], unit="h")
    )
    if df["timestamp"].duplicated().any():
        duplicates = int(df["timestamp"].duplicated().sum())
        raise ValueError(
            f"Dataset contains {duplicates} duplicate date-hour records. "
            "Remove duplicates before training."
        )

    # Friendly category labels for the dashboard only; not used as model inputs.
    if "season" in df.columns:
        season_map = {
            "springer": "Spring", "spring": "Spring", "1": "Spring",
            "summer": "Summer", "2": "Summer",
            "fall": "Fall", "autumn": "Fall", "3": "Fall",
            "winter": "Winter", "4": "Winter",
        }
        df["season"] = (
            df["season"].astype(str).str.strip().str.lower()
            .map(season_map).fillna(df["season"].astype(str).str.title())
        )
    if "weathersit" in df.columns:
        weather_map = {
            "clear": "Clear", "clear/few clouds": "Clear",
            "mist": "Mist / Cloudy", "mist + cloudy": "Mist / Cloudy",
            "light snow/rain": "Light Rain / Snow",
            "light rain/snow": "Light Rain / Snow",
            "heavy rain/snow": "Heavy Rain / Snow",
        }
        df["weathersit"] = (
            df["weathersit"].astype(str).str.strip().str.lower()
            .map(weather_map).fillna(df["weathersit"].astype(str).str.title())
        )
    for col in ["holiday", "workingday"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["dteday"] = df["timestamp"].dt.normalize()

    full_index = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq="h",
        name="timestamp",
    )
    observed = df.set_index("timestamp")[TARGET].astype(float).sort_index()
    series = observed.reindex(full_index)
    series.index.name = "timestamp"

    # Add parsed, user-friendly calendar fields for display and analysis.
    df["hour"] = df["timestamp"].dt.hour
    df["weekday_name"] = df["timestamp"].dt.day_name()
    df["month_name"] = df["timestamp"].dt.month_name()
    df["year"] = df["timestamp"].dt.year
    return df, series


def _calendar_features(index: pd.DatetimeIndex, origin: pd.Timestamp) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    hour = idx.hour.to_numpy()
    weekday = idx.dayofweek.to_numpy()
    month = idx.month.to_numpy()
    dayofyear = idx.dayofyear.to_numpy()
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7),
        "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "dayofyear_sin": np.sin(2 * np.pi * (dayofyear - 1) / 365.25),
        "dayofyear_cos": np.cos(2 * np.pi * (dayofyear - 1) / 365.25),
        "is_weekend": (weekday >= 5).astype(int),
        "days_since_start": (idx - origin).total_seconds() / 86400.0,
    }, index=idx)


def make_training_features(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Create strictly past-looking features on a continuous hourly index."""
    if series.empty:
        raise ValueError("Cannot create features from an empty time series.")
    series = series.sort_index().astype(float)
    filled = series.ffill()
    origin = series.index.min()
    X = _calendar_features(series.index, origin)
    X["lag_1"] = filled.shift(1)
    X["lag_24"] = filled.shift(24)
    X["lag_168"] = filled.shift(168)
    prior = filled.shift(1)
    X["rolling_mean_24"] = prior.rolling(24, min_periods=18).mean()
    X["rolling_mean_168"] = prior.rolling(168, min_periods=120).mean()
    X = X[FEATURE_COLUMNS]
    y = series.rename(TARGET)
    return X, y


def _feature_row(history: pd.Series, timestamp: pd.Timestamp,
                 origin: pd.Timestamp) -> pd.DataFrame:
    """Build one future feature row from values strictly before timestamp."""
    history = history.sort_index().astype(float).ffill()
    calendar = _calendar_features(pd.DatetimeIndex([timestamp]), origin)
    def exact_lag(hours: int) -> float:
        key = timestamp - pd.Timedelta(hours=hours)
        if key not in history.index:
            return float(history.iloc[-1]) if len(history) else 0.0
        value = history.loc[key]
        return float(value) if pd.notna(value) else (
            float(history.loc[:key].dropna().iloc[-1])
            if history.loc[:key].notna().any() else 0.0
        )
    past_24 = history.loc[
        (history.index >= timestamp - pd.Timedelta(hours=24))
        & (history.index < timestamp)
    ].tail(24)
    past_168 = history.loc[
        (history.index >= timestamp - pd.Timedelta(hours=168))
        & (history.index < timestamp)
    ].tail(168)
    calendar["lag_1"] = exact_lag(1)
    calendar["lag_24"] = exact_lag(24)
    calendar["lag_168"] = exact_lag(168)
    calendar["rolling_mean_24"] = (
        float(past_24.mean()) if len(past_24) >= 18 else exact_lag(24)
    )
    calendar["rolling_mean_168"] = (
        float(past_168.mean()) if len(past_168) >= 120 else exact_lag(168)
    )
    return calendar[FEATURE_COLUMNS]


def recursive_forecast(
    history: pd.Series,
    model,
    days: int = 30,
    method: str = "model",
    start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generate hourly recursive forecasts; never uses future actual values."""
    if not 1 <= int(days) <= MAX_FORECAST_DAYS:
        raise ValueError("Forecast duration must be between 1 and 30 days.")
    history = history.sort_index().astype(float)
    if history.empty:
        raise ValueError("At least one historical observation is required.")
    history = history.ffill()
    last_timestamp = history.index.max()
    if start is None:
        start = last_timestamp + pd.Timedelta(hours=1)
    future_index = pd.date_range(start=start, periods=int(days) * 24, freq="h")
    working = history.copy()
    origin = history.index.min()
    rows = []

    for timestamp in future_index:
        if method == "daily_naive":
            source = timestamp - pd.Timedelta(hours=24)
            value = float(working.loc[source]) if source in working.index else float(working.iloc[-1])
        elif method == "weekly_naive":
            source = timestamp - pd.Timedelta(hours=168)
            value = float(working.loc[source]) if source in working.index else float(working.iloc[-1])
        else:
            X_one = _feature_row(working, timestamp, origin)
            value = float(np.asarray(model.predict(X_one)).reshape(-1)[0])
        value = max(0.0, value) if np.isfinite(value) else 0.0
        rows.append({"timestamp": timestamp, "predicted_cnt": value})
        working.loc[timestamp] = value

    return pd.DataFrame(rows)


def _metrics(actual: pd.Series, predicted: pd.Series, name: str) -> dict:
    aligned = pd.concat(
        [actual.rename("actual"), predicted.rename("predicted")], axis=1
    ).dropna()
    if aligned.empty:
        return {"Model": name, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}
    y_true = aligned["actual"].to_numpy(dtype=float)
    y_pred = aligned["predicted"].to_numpy(dtype=float)
    nonzero = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
        if nonzero.any() else np.nan
    )
    return {
        "Model": name,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mape,
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def train_and_forecast(data_path: str | Path, output_dir: str | Path | None = None) -> dict:
    """Backtest a 30-day horizon, retrain on all history, and forecast 30 days."""
    df, series = load_dataset(data_path)
    X, y = make_training_features(series)
    valid_X = X.notna().all(axis=1)
    observed_y = y.notna()
    eligible = valid_X & observed_y
    if int(eligible.sum()) < 500:
        raise ValueError(
            "Not enough valid hourly history after creating lag features. "
            "Check that Dataset.csv has a sufficient time span."
        )

    # Last 30 calendar days are held out; training excludes every target in this window.
    cutoff = series.index.max() - pd.Timedelta(hours=FORECAST_HOURS_PER_DAY * 30 - 1)
    train_mask = eligible & (X.index < cutoff)
    test_index = series.index[series.index >= cutoff]
    if train_mask.sum() < 500 or len(test_index) < 24:
        raise ValueError("Not enough pre-test history for a reliable chronological backtest.")

    model_params = dict(
        max_iter=180,
        learning_rate=0.075,
        max_leaf_nodes=31,
        l2_regularization=1.5,
        random_state=42,
        early_stopping=False,
    )
    backtest_model = HistGradientBoostingRegressor(**model_params)
    backtest_model.fit(X.loc[train_mask], y.loc[train_mask])

    pre_test_history = series.loc[series.index < cutoff].ffill()
    observed_end = series.index.max()
    backtest_days = max(1, int((observed_end - cutoff).total_seconds() // 3600 + 1) // 24)
    # Always evaluate the exact final 720 hourly slots.
    backtest_index = pd.date_range(cutoff, periods=720, freq="h")
    # For the rare incomplete final index, the dataset is reindexed hourly already.
    model_backtest = recursive_forecast(
        pre_test_history, backtest_model, days=30, start=backtest_index[0]
    ).set_index("timestamp")["predicted_cnt"]
    daily_naive = recursive_forecast(
        pre_test_history, None, days=30, method="daily_naive", start=backtest_index[0]
    ).set_index("timestamp")["predicted_cnt"]
    weekly_naive = recursive_forecast(
        pre_test_history, None, days=30, method="weekly_naive", start=backtest_index[0]
    ).set_index("timestamp")["predicted_cnt"]

    actual_test = series.reindex(backtest_index)
    comparison = pd.DataFrame([
        _metrics(actual_test, model_backtest, "Gradient Boosting (lag features)"),
        _metrics(actual_test, daily_naive, "Seasonal Naive (24-hour)"),
        _metrics(actual_test, weekly_naive, "Seasonal Naive (168-hour)"),
    ]).sort_values("MAE", na_position="last").reset_index(drop=True)

    # Train final model on all observed targets, still using past-only features.
    final_mask = eligible
    final_model = HistGradientBoostingRegressor(**model_params)
    final_model.fit(X.loc[final_mask], y.loc[final_mask])
    forecast = recursive_forecast(series, final_model, days=30)
    forecast["predicted_cnt"] = forecast["predicted_cnt"].round(2)

    # Permutation importance is calculated on a modest sample of the held-out
    # chronological window and is descriptive, not a causal explanation.
    # Importance is estimated on a sample of the pre-cutoff training period,
    # avoiding any use of held-out target-derived lag values.
    imp_frame = X.loc[train_mask].dropna()
    imp_target = y.reindex(imp_frame.index).dropna()
    imp_frame = imp_frame.loc[imp_target.index]
    importance_df = pd.DataFrame(columns=["Feature", "Importance"])
    if len(imp_frame) >= 30:
        sample_positions = np.linspace(0, len(imp_frame) - 1, min(500, len(imp_frame))).astype(int)
        imp_X = imp_frame.iloc[sample_positions]
        imp_y = imp_target.iloc[sample_positions]
        try:
            result = permutation_importance(
                backtest_model, imp_X, imp_y, n_repeats=2,
                random_state=42, scoring="neg_mean_absolute_error", n_jobs=1
            )
            importance_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Importance": result.importances_mean,
            }).sort_values("Importance", ascending=False).reset_index(drop=True)
        except Exception as exc:
            warnings.warn(f"Could not compute permutation importance: {exc}")

    actual_vs_predicted = pd.DataFrame({
        "timestamp": backtest_index,
        "actual_cnt": actual_test.to_numpy(),
        "predicted_cnt": model_backtest.reindex(backtest_index).to_numpy(),
    })
    actual_vs_predicted = actual_vs_predicted.dropna(subset=["actual_cnt"])

    report = {
        "target": "cnt",
        "best_model": "HistGradientBoostingRegressor",
        "frequency": "Hourly",
        "forecast_horizon_days": 30,
        "forecast_horizon_hours": 720,
        "feature_count": len(FEATURE_COLUMNS),
        "excluded_leakage_columns": ["casual", "registered"],
        "future_weather_used": False,
        "backtest_start": str(backtest_index[0]),
        "backtest_end": str(backtest_index[-1]),
        "backtest_observed_hours": int(actual_test.notna().sum()),
        "missing_hour_slots_in_dataset": int(series.isna().sum()),
        "validation_note": (
            "Chronological 30-day recursive backtest. Predictions are generated "
            "without using actual target values from the held-out period. "
            "Future weather is not modeled because future weather observations "
            "are not supplied."
        ),
        "metrics": comparison.to_dict(orient="records"),
    }

    bundle = {
        "df": df,
        "series": series,
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS.copy(),
        "comparison": comparison,
        "forecast": forecast,
        "report": report,
        "importance": importance_df,
        "actual_vs_predicted": actual_vs_predicted,
    }

    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, output / "gbr_model.pkl")
        joblib.dump(FEATURE_COLUMNS, output / "feature_columns.pkl")
        joblib.dump(report, output / "metadata.pkl")
        comparison.to_csv(output / "model_comparison.csv", index=False)
        forecast.to_csv(output / "next_30_days_forecast.csv", index=False)
        (output / "validation_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        actual_vs_predicted.to_csv(output / "backtest_actual_vs_predicted.csv", index=False)
        importance_df.to_csv(output / "feature_importance.csv", index=False)

    return bundle
