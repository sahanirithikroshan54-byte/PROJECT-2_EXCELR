#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
🏍️ BIKESHARE INTELLIGENCE
Unified Streamlit application for bike-rental analytics and demand forecasting.

Place a compatible hourly bike-sharing CSV beside this file. Supported names:
Dataset.csv, Dataset(4).csv, hour.csv, or bike_sharing_hour.csv.
Optional: model_comparison.csv and next_30_days_forecast.csv.
"""

from pathlib import Path
from datetime import timedelta
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# PAGE AND THEME
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="BikeShare Intelligence",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    "Dataset.csv", "Dataset(4).csv", "hour.csv",
    "bike_sharing_hour.csv", "dataset.csv",
]
FORECAST_CANDIDATES = [
    "next_30_days_forecast.csv",
    "next_30_days_forecast(4).csv",
]
COMPARISON_FILE = BASE_DIR / "model_comparison.csv"

FEATURES = [
    "instant", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
]
NUMERIC_COLUMNS = [
    "instant", "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum", "windspeed",
    "casual", "registered", "cnt",
]
CATEGORY_COLUMNS = ["season", "holiday", "workingday", "weathersit"]

st.markdown(
    """
    <style>
    .stApp {
        background:
          radial-gradient(circle at 85% 5%, rgba(0,153,255,.13), transparent 28%),
          radial-gradient(circle at 5% 75%, rgba(0,214,190,.09), transparent 30%),
          linear-gradient(145deg,#061321 0%,#091d32 48%,#071426 100%);
        color:#edf7ff;
    }
    [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#06101e,#0a2037);
        border-right:1px solid rgba(69,181,255,.24);
    }
    [data-testid="stSidebar"] * {color:#e8f4ff !important;}
    .block-container {max-width:1500px;padding-top:2rem;padding-bottom:3rem;}
    h1,h2,h3 {color:#f7fbff !important;font-weight:820 !important;}
    p,li {color:#c8dced;line-height:1.6;}
    .hero {
        padding:1.35rem 1.6rem;border:1px solid rgba(82,190,255,.28);
        border-radius:20px;background:linear-gradient(120deg,rgba(7,45,75,.94),rgba(8,27,49,.88));
        box-shadow:0 16px 42px rgba(0,0,0,.24);margin-bottom:1.2rem;
    }
    .hero-kicker {color:#5ed7ff;font-size:.76rem;font-weight:800;letter-spacing:2px;}
    .hero-copy {color:#b6cde0;font-size:1rem;margin-bottom:0;}
    [data-testid="stMetric"] {
        background:linear-gradient(145deg,rgba(13,48,78,.95),rgba(5,25,46,.98));
        border:1px solid rgba(86,185,245,.23);border-radius:17px;
        padding:17px;min-height:115px;box-shadow:0 12px 30px rgba(0,0,0,.22);
    }
    [data-testid="stMetricLabel"] {color:#9dc4df !important;font-weight:750 !important;}
    [data-testid="stMetricValue"] {color:#fff !important;font-weight:850 !important;}
    .stButton>button,.stDownloadButton>button {
        border:1px solid rgba(111,211,255,.4);border-radius:12px;
        min-height:45px;font-weight:800;color:white;
        background:linear-gradient(130deg,#079bd7,#075fc0);
    }
    .stButton>button:hover,.stDownloadButton>button:hover {
        border-color:#8ce4ff;background:linear-gradient(130deg,#10b5ed,#0875dc);
    }
    div[data-baseweb="select"]>div,div[data-baseweb="input"]>div {
        background:#0a2945 !important;border-color:rgba(94,183,238,.3) !important;
        border-radius:10px;color:white !important;
    }
    input {color:white !important;}
    [data-testid="stDataFrame"],[data-testid="stVegaLiteChart"],
    [data-testid="stPlotlyChart"] {border:1px solid rgba(86,185,245,.18);border-radius:14px;}
    hr {border-color:rgba(100,190,245,.2);}
    .footer {text-align:center;color:#7595af;font-size:.78rem;padding:2rem 0 .5rem;}
    @media(max-width:800px){.block-container{padding-left:1rem;padding-right:1rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# FILE DISCOVERY AND DATA PREPARATION
# ---------------------------------------------------------------------
def locate_file(candidates):
    for name in candidates:
        path = BASE_DIR / name
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_dataset(path_string):
    data = pd.read_csv(path_string)
    data.columns = [str(c).strip().lower() for c in data.columns]

    aliases = {"date": "dteday", "datetime": "dteday", "hour": "hr",
               "count": "cnt", "rentals": "cnt"}
    for old, new in aliases.items():
        if old in data.columns and new not in data.columns:
            data.rename(columns={old: new}, inplace=True)

    required = ["dteday", "cnt", *FEATURES]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    data = data.replace(["?", "NA", "N/A", "null", "None", ""], np.nan)
    data["dteday"] = pd.to_datetime(data["dteday"], dayfirst=True, errors="coerce")

    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    category_maps = {}
    for col in CATEGORY_COLUMNS:
        if col not in data:
            continue
        if data[col].isna().all():
            data[col] = 0
        elif not pd.api.types.is_numeric_dtype(data[col]):
            values = data[col].astype("string").str.strip()
            mode = values.mode(dropna=True)
            values = values.fillna(mode.iloc[0] if not mode.empty else "Unknown")
            labels = sorted(values.astype(str).unique())
            category_maps[col] = {label: index for index, label in enumerate(labels)}
            data[col] = values.astype(str).map(category_maps[col]).astype(float)

    for col in data.columns:
        if col == "dteday":
            continue
        if not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], errors="coerce")
        median = data[col].median(skipna=True)
        data[col] = data[col].fillna(0 if pd.isna(median) else median)

    data = data.dropna(subset=["dteday", "cnt"]).drop_duplicates()
    data["cnt"] = data["cnt"].clip(lower=0)
    data = data.sort_values(["dteday", "hr", "instant"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid historical observations remain after cleaning.")

    data["Datetime"] = data["dteday"] + pd.to_timedelta(data["hr"].astype(int), unit="h")
    data["Date"] = data["Datetime"].dt.date
    data["Hour"] = data["hr"].astype(int)
    data["WeekdayName"] = data["Datetime"].dt.day_name()
    data["MonthName"] = data["Datetime"].dt.month_name()
    return data, category_maps


@st.cache_data(show_spinner=False)
def load_comparison(path_string):
    try:
        comparison = pd.read_csv(path_string)
        needed = {"Model", "MAE", "RMSE", "MAPE", "R2"}
        return comparison if needed.issubset(comparison.columns) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_supplied_forecast(path_string):
    forecast = pd.read_csv(path_string)
    forecast.columns = [str(c).strip() for c in forecast.columns]
    required = {"Datetime", "Predicted_cnt"}
    if not required.issubset(forecast.columns):
        raise ValueError("Forecast CSV must contain Datetime and Predicted_cnt.")
    forecast["Datetime"] = pd.to_datetime(forecast["Datetime"], errors="coerce")
    forecast["Predicted_cnt"] = pd.to_numeric(forecast["Predicted_cnt"], errors="coerce")
    forecast = forecast.dropna(subset=["Datetime", "Predicted_cnt"])
    if forecast.empty or not np.isfinite(forecast["Predicted_cnt"]).all():
        raise ValueError("The supplied forecast contains no valid predictions.")
    if (forecast["Predicted_cnt"] < 0).any():
        raise ValueError("The supplied forecast contains negative values.")
    forecast["Predicted_cnt"] = forecast["Predicted_cnt"].mask(
        forecast["Predicted_cnt"].eq(0), 1
    )
    return forecast.sort_values("Datetime").reset_index(drop=True)


# ---------------------------------------------------------------------
# MODEL TRAINING AND EVALUATION
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_model(path_string):
    data, _ = load_dataset(path_string)
    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = data["cnt"].astype(float)

    split = max(1, int(len(data) * 0.8))
    if split >= len(data):
        split = max(1, len(data) - 1)

    if len(data) >= 10 and split < len(data):
        evaluator = GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            subsample=0.8, max_features=0.8, random_state=50,
        )
        evaluator.fit(X.iloc[:split], y.iloc[:split])
        test_pred = np.maximum(0, evaluator.predict(X.iloc[split:]))
        actual = y.iloc[split:]
        mae = mean_absolute_error(actual, test_pred)
        rmse = np.sqrt(mean_squared_error(actual, test_pred))
        r2 = r2_score(actual, test_pred) if len(actual) > 1 else np.nan
        nonzero = actual != 0
        mape = (np.abs((actual[nonzero] - test_pred[nonzero]) /
                       actual[nonzero]).mean() * 100) if nonzero.any() else np.nan
        evaluation = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2,
                      "Train rows": split, "Test rows": len(actual)}
    else:
        evaluation = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan,
                      "R2": np.nan, "Train rows": len(data), "Test rows": 0}

    model = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        subsample=0.8, max_features=0.8, random_state=50,
    )
    model.fit(X, y)
    return model, evaluation


def build_future_features(history, timestamp):
    last = history.iloc[-1]
    row = {feature: float(last[feature]) for feature in FEATURES}
    row["instant"] = float(last["instant"]) + 1
    row["yr"] = float(timestamp.year > history["dteday"].dt.year.min())
    row["mnth"] = float(timestamp.month)
    row["hr"] = float(timestamp.hour)
    row["weekday"] = float((timestamp.dayofweek + 1) % 7)
    # Use the most recently observed encoded holiday/working-day/weather
    # and season values when a future external feed is unavailable.
    return pd.DataFrame([{key: row[key] for key in FEATURES}]), row


def generate_forecast(data, days, model):
    if not 1 <= int(days) <= 30:
        raise ValueError("Choose a forecast horizon from 1 to 30 days.")
    history = data.sort_values(["dteday", "hr"]).reset_index(drop=True).copy()
    last = history.iloc[-1]
    stamp = pd.Timestamp(last["dteday"]).normalize() + pd.Timedelta(hours=int(last["hr"]))
    results = []

    for _ in range(int(days) * 24):
        stamp += pd.Timedelta(hours=1)
        features, feature_row = build_future_features(history, stamp)
        prediction = float(np.asarray(model.predict(features)).reshape(-1)[0])
        if not np.isfinite(prediction):
            raise ValueError(f"Invalid prediction generated for {stamp}.")
        prediction = max(0.0, prediction)
        # Business display floor is applied silently to the prediction series.
        prediction = 1.0 if prediction == 0 else prediction
        results.append({"Datetime": stamp, "Predicted_cnt": prediction})

        new_row = {column: np.nan for column in history.columns}
        new_row.update(feature_row)
        new_row["dteday"] = stamp.normalize()
        new_row["hr"] = stamp.hour
        new_row["cnt"] = prediction
        for column in FEATURES:
            if pd.isna(new_row.get(column)):
                new_row[column] = feature_row[column]
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    return pd.DataFrame(results)


def daily_rollup(frame, date_col, value_col):
    work = frame.copy()
    work["RentalDate"] = pd.to_datetime(work[date_col]).dt.date
    return work.groupby("RentalDate", as_index=False).agg(
        Predicted_Rentals=(value_col, "sum"),
        Average_Hourly_Rentals=(value_col, "mean"),
        Peak_Hourly_Rentals=(value_col, "max"),
        Minimum_Hourly_Rentals=(value_col, "min"),
    )


def csv_download(label, frame, filename):
    st.download_button(
        label, frame.to_csv(index=False).encode("utf-8"),
        filename, "text/csv", use_container_width=True,
    )


# ---------------------------------------------------------------------
# INITIALIZE DATA AND MODEL
# ---------------------------------------------------------------------
data_path = locate_file(DATA_CANDIDATES)
forecast_path = locate_file(FORECAST_CANDIDATES)

if data_path is None:
    st.error("📁 Historical bike-sharing dataset was not found.")
    st.info("Place Dataset.csv or Dataset(4).csv in the same folder as this application.")
    st.stop()

try:
    with st.spinner("Preparing rental data and training the forecasting engine..."):
        df, category_maps = load_dataset(str(data_path))
        model, evaluation = train_model(str(data_path))
        comparison = load_comparison(str(COMPARISON_FILE)) if COMPARISON_FILE.exists() else pd.DataFrame()
except Exception as exc:
    st.error(f"Unable to initialize the application: {exc}")
    st.stop()

# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------
st.sidebar.markdown("## 🏍️ BikeShare AI")
st.sidebar.caption("RENTAL ANALYTICS • DEMAND INTELLIGENCE")
st.sidebar.divider()

pages = [
    "📌 Dashboard Overview",
    "🔎 Exploratory Analytics",
    "🕒 Rental Patterns",
    "🤖 Model Performance",
    "🔮 Demand Forecast",
    "🗂️ Dataset Explorer",
]
page = st.sidebar.radio("🧭 Navigation", pages)

st.sidebar.divider()
st.sidebar.subheader("🟢 Platform Status")
st.sidebar.success("Historical data ready")
st.sidebar.success("Prediction engine ready")
st.sidebar.caption(f"📚 {len(df):,} valid hourly observations")
st.sidebar.caption(f"📅 {df['Datetime'].min():%d %b %Y} — {df['Datetime'].max():%d %b %Y}")

with st.sidebar.expander("⚙️ Forecast configuration"):
    st.write("**Algorithm:** Gradient Boosting Regressor")
    st.write("**Estimators:** 500")
    st.write("**Learning rate:** 0.05")
    st.write("**Maximum depth:** 7")
    st.write("**Forecast frequency:** Hourly")
    st.write("**Maximum horizon:** 30 days")

with st.sidebar.expander("📦 Dataset summary"):
    st.metric("Historical records", f"{len(df):,}")
    st.metric("Features", f"{len(df.columns):,}")
    st.metric("Total recorded rentals", f"{df['cnt'].sum():,.0f}")

# ---------------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">SMART MOBILITY • DATA-DRIVEN OPERATIONS</div>
      <h1>🏍️ Bike Sharing Rental Intelligence</h1>
      <p class="hero-copy">Understand demand, discover rental patterns and plan fleet capacity
      with historical analytics and machine-learning forecasts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# DASHBOARD OVERVIEW
# ---------------------------------------------------------------------
if page == "📌 Dashboard Overview":
    st.header("📊 Rental Performance Overview")
    latest = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚲 Total rentals", f"{df['cnt'].sum():,.0f}")
    c2.metric("📈 Average per hour", f"{df['cnt'].mean():,.1f}")
    c3.metric("🏆 Peak hourly demand", f"{df['cnt'].max():,.0f}")
    c4.metric("🕒 Latest observation", latest["Datetime"].strftime("%d %b %Y"))

    daily = df.groupby("Date")["cnt"].sum()
    st.subheader("📈 Daily Rental Activity")
    st.line_chart(daily, height=340)

    left, right = st.columns(2)
    with left:
        st.subheader("⏰ Demand by Hour")
        st.bar_chart(df.groupby("Hour")["cnt"].mean(), height=300)
    with right:
        st.subheader("📆 Demand by Weekday")
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
        weekday = df.groupby("WeekdayName")["cnt"].mean().reindex(weekday_order)
        st.bar_chart(weekday, height=300)

    left, right = st.columns(2)
    with left:
        st.subheader("🌦️ Demand by Weather")
        st.bar_chart(df.groupby("weathersit")["cnt"].mean(), height=280)
    with right:
        st.subheader("👥 Customer Mix")
        if {"casual", "registered"}.issubset(df.columns):
            st.bar_chart(df[["casual", "registered"]].sum(), height=280)
        else:
            st.info("Customer-type columns are not available in this dataset.")

    st.subheader("🔍 Recent Rental Records")
    st.dataframe(df.tail(15), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# EXPLORATORY ANALYTICS
# ---------------------------------------------------------------------
elif page == "🔎 Exploratory Analytics":
    st.header("🔎 Exploratory Data Analysis")
    st.caption("Explore the cleaned historical dataset and its statistical characteristics.")
    q1, q2, q3 = st.columns(3)
    q1.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    q2.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    q3.metric("Numeric features", f"{len(df.select_dtypes(include=np.number).columns):,}")

    st.subheader("📚 Descriptive Statistics")
    st.dataframe(df.select_dtypes(include=np.number).describe().T.round(3),
                 use_container_width=True)

    numeric_options = [column for column in
                       ["cnt", "casual", "registered", "temp", "atemp", "hum", "windspeed"]
                       if column in df.columns]
    selected = st.selectbox("📊 Select a feature distribution", numeric_options)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.hist(df[selected].dropna(), bins=35)
    ax.set_title(f"Distribution of {selected}")
    ax.set_xlabel(selected)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    corr_columns = [column for column in
                    ["cnt", "casual", "registered", "temp", "atemp", "hum",
                     "windspeed", "hr", "mnth"] if column in df.columns]
    st.subheader("🔥 Feature Correlation")
    fig, ax = plt.subplots(figsize=(10, 5))
    matrix = df[corr_columns].corr()
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(corr_columns)), corr_columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_columns)), corr_columns)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------
# RENTAL PATTERNS
# ---------------------------------------------------------------------
elif page == "🕒 Rental Patterns":
    st.header("🕒 Understand Rental Behaviour")
    left, right = st.columns(2)
    with left:
        st.subheader("⏰ Hourly Demand Profile")
        st.bar_chart(df.groupby("Hour")["cnt"].mean(), height=320)
        st.subheader("📅 Weekday Demand")
        order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
        st.bar_chart(df.groupby("WeekdayName")["cnt"].mean().reindex(order), height=300)
    with right:
        st.subheader("📈 Daily Rentals")
        st.line_chart(df.groupby("Date")["cnt"].sum(), height=320)
        st.subheader("🌡️ Temperature and Demand")
        st.scatter_chart(df, x="temp", y="cnt", color="weathersit", height=300)

    st.subheader("🗓️ Hourly Demand by Weekday")
    pivot = df.pivot_table(index="Hour", columns="WeekdayName",
                           values="cnt", aggfunc="mean")
    st.dataframe(pivot.round(1), use_container_width=True)
    if {"casual", "registered"}.issubset(df.columns):
        st.subheader("👥 Customer Type by Hour")
        st.bar_chart(df.groupby("Hour")[["casual", "registered"]].mean(), height=320)

# ---------------------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------------------
elif page == "🤖 Model Performance":
    st.header("🤖 Prediction Model & Performance")
    st.caption("Metrics below are calculated from a chronological holdout set.")
    a, b, c, d = st.columns(4)
    a.metric("MAE", f"{evaluation['MAE']:,.2f}" if np.isfinite(evaluation["MAE"]) else "N/A")
    b.metric("RMSE", f"{evaluation['RMSE']:,.2f}" if np.isfinite(evaluation["RMSE"]) else "N/A")
    c.metric("MAPE", f"{evaluation['MAPE']:,.2f}%" if np.isfinite(evaluation["MAPE"]) else "N/A")
    d.metric("R² score", f"{evaluation['R2']:.4f}" if np.isfinite(evaluation["R2"]) else "N/A")

    st.write(f"**Training observations:** {evaluation['Train rows']:,}")
    st.write(f"**Testing observations:** {evaluation['Test rows']:,}")

    if not comparison.empty:
        st.subheader("🏁 Notebook Model Comparison")
        st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)
        metric_choices = [col for col in ["MAE", "RMSE", "MAPE", "R2"] if col in comparison]
        if metric_choices:
            metric = st.selectbox("Compare models using", metric_choices,
                                  index=metric_choices.index("RMSE") if "RMSE" in metric_choices else 0)
            st.bar_chart(comparison.set_index("Model")[metric], height=330)
        csv_download("⬇️ Download model comparison", comparison, "model_comparison.csv")

    st.subheader("🌟 Gradient Boosting Feature Importance")
    importance = pd.DataFrame({
        "Feature": FEATURES, "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    st.bar_chart(importance.set_index("Feature")["Importance"], height=340)
    st.dataframe(importance.round(5), use_container_width=True, hide_index=True)

    st.info("The model excludes casual and registered counts from its inputs to prevent target leakage.")

# ---------------------------------------------------------------------
# DEMAND FORECAST
# ---------------------------------------------------------------------
elif page == "🔮 Demand Forecast":
    st.header("🔮 Future Bike Rental Demand")
    st.write("Generate hourly demand estimates using the trained Gradient Boosting model.")
    days = int(st.slider("📅 Forecast horizon", min_value=1, max_value=30, value=30))
    st.caption(f"Forecast output: **{days * 24:,} hourly estimates**")

    if forecast_path is not None:
        st.info("A supplied forecast file is available. Generate a fresh model forecast below, or load the supplied forecast.")
        use_supplied = st.checkbox("Use supplied forecast file", value=False)
    else:
        use_supplied = False
        st.caption("Forecasts are generated directly by the trained model.")

    run_col, clear_col = st.columns([3, 1])
    with run_col:
        generate = st.button("🏍️ Generate Demand Forecast", type="primary", use_container_width=True)
    with clear_col:
        clear = st.button("↻ Clear results", use_container_width=True)

    if clear:
        st.session_state.pop("bike_forecast_result", None)
        st.session_state.pop("bike_forecast_horizon", None)

    if generate:
        with st.spinner("Analysing demand patterns and preparing predictions..."):
            try:
                if use_supplied and forecast_path is not None:
                    forecast = load_supplied_forecast(str(forecast_path)).copy()
                    forecast = forecast.head(days * 24)
                else:
                    forecast = generate_forecast(df, days, model)
                st.session_state["bike_forecast_result"] = forecast
                st.session_state["bike_forecast_horizon"] = days
            except Exception as exc:
                st.error(f"Forecast generation failed: {exc}")

    if "bike_forecast_result" in st.session_state:
        forecast = st.session_state["bike_forecast_result"].copy()
        forecast = forecast.sort_values("Datetime").reset_index(drop=True)
        daily = daily_rollup(forecast, "Datetime", "Predicted_cnt")

        st.divider()
        st.subheader("📊 Forecast at a Glance")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("🕒 Forecast hours", f"{len(forecast):,}")
        f2.metric("📈 Average hourly demand", f"{forecast['Predicted_cnt'].mean():,.1f}")
        f3.metric("🏆 Peak hourly demand", f"{forecast['Predicted_cnt'].max():,.1f}")
        f4.metric("🚲 Expected rentals", f"{forecast['Predicted_cnt'].sum():,.0f}")

        peak = forecast.loc[forecast["Predicted_cnt"].idxmax()]
        st.caption(f"🏆 Highest expected demand: {peak['Datetime']:%d %b %Y, %H:%M}")

        st.subheader("📈 Hourly Demand Forecast")
        st.line_chart(forecast.set_index("Datetime")["Predicted_cnt"], height=390)

        st.subheader("📆 Daily Demand Forecast")
        st.line_chart(daily.set_index("RentalDate")["Predicted_Rentals"], height=340)

        st.subheader("🕒 Hourly Forecast Details")
        hourly_display = forecast.copy()
        hourly_display["Datetime"] = hourly_display["Datetime"].dt.strftime("%d-%b-%Y %H:%M")
        hourly_display["Predicted_cnt"] = hourly_display["Predicted_cnt"].round(2)
        st.dataframe(hourly_display, use_container_width=True, height=420, hide_index=True)

        st.subheader("📅 Daily Forecast Summary")
        daily_display = daily.copy()
        daily_display["RentalDate"] = pd.to_datetime(daily_display["RentalDate"]).dt.strftime("%d-%b-%Y")
        numeric = [col for col in daily_display if col != "RentalDate"]
        daily_display[numeric] = daily_display[numeric].round(2)
        st.dataframe(daily_display, use_container_width=True, hide_index=True)

        e1, e2 = st.columns(2)
        with e1:
            csv_download("⬇️ Download hourly forecast", forecast, "bike_rental_hourly_forecast.csv")
        with e2:
            csv_download("📅 Download daily summary", daily, "bike_rental_daily_forecast.csv")

# ---------------------------------------------------------------------
# DATASET EXPLORER
# ---------------------------------------------------------------------
elif page == "🗂️ Dataset Explorer":
    st.header("🗂️ Historical Data Explorer")
    minimum = df["Datetime"].min().date()
    maximum = df["Datetime"].max().date()
    left, right = st.columns(2)
    with left:
        start_date = st.date_input("📅 Start date", value=minimum,
                                   min_value=minimum, max_value=maximum)
    with right:
        end_date = st.date_input("📅 End date", value=maximum,
                                 min_value=minimum, max_value=maximum)

    if start_date > end_date:
        st.warning("The start date must be on or before the end date.")
    else:
        filtered = df[df["Datetime"].dt.date.between(start_date, end_date)].copy()
        st.metric("🔎 Matching observations", f"{len(filtered):,}")
        st.metric("🚲 Rentals in selected period", f"{filtered['cnt'].sum():,.0f}")
        st.dataframe(filtered, use_container_width=True, height=470, hide_index=True)
        csv_download("⬇️ Download selected historical records",
                     filtered, "bike_rental_historical_records.csv")

    st.divider()
    st.subheader("📚 Dataset Structure")
    schema = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[col].dtype) for col in df.columns],
        "Available values": [int(df[col].notna().sum()) for col in df.columns],
    })
    st.dataframe(schema, use_container_width=True, hide_index=True)
    csv_download("⬇️ Download cleaned dataset", df, "bike_rental_cleaned_dataset.csv")

# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
      🏍️ BikeShare Intelligence &nbsp;•&nbsp; Smart Mobility Analytics
      &nbsp;•&nbsp; Gradient Boosting Demand Forecasting
    </div>
    """,
    unsafe_allow_html=True,
)
