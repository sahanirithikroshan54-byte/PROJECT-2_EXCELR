# -*- coding: utf-8 -*-
"""BikeShare AI — hourly bike rental demand forecasting dashboard."""
from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset.csv"
MODEL_PATH = BASE_DIR / "gbr_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_columns.pkl"

st.set_page_config(
    page_title="BikeShare AI | Demand Forecast",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bike-sharing background and a high-contrast dashboard theme.
st.markdown(
    """
    <style>
    .stApp {
      background: linear-gradient(rgba(2,12,24,.88),rgba(3,18,31,.94)),
      url("https://images.unsplash.com/photo-1485965120184-e220f721d03e?auto=format&fit=crop&w=2600&q=90");
      background-size: cover; background-position: center; background-attachment: fixed;
      color: #eef8ff;
    }
    .stApp:before {content:"";position:fixed;inset:0;pointer-events:none;
      background:radial-gradient(circle at 80% 10%,rgba(0,220,170,.12),transparent 35%),
      radial-gradient(circle at 5% 75%,rgba(0,145,255,.13),transparent 38%);z-index:0}
    .main{position:relative;z-index:1}
    .block-container{max-width:1500px;padding:2.2rem 2.5rem 3rem}
    section[data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(2,18,31,.99),rgba(4,35,49,.99));border-right:1px solid rgba(63,225,190,.28)}
    section[data-testid="stSidebar"] *{color:#edfaff!important}
    h1{color:#fff!important;font-size:2.65rem!important;font-weight:850!important;letter-spacing:-.8px;text-shadow:0 4px 20px #000}
    h2{color:#f5ffff!important;font-weight:820!important;text-shadow:0 3px 14px #000}
    h3{color:#e5fff8!important;font-weight:780!important}
    p,li{color:#d2e8ef;line-height:1.6}
    .intro{color:#b5d3df;font-size:1.02rem;line-height:1.7;margin-bottom:1.2rem;text-shadow:0 2px 8px #000}
    div[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(8,55,67,.95),rgba(3,29,43,.97));border:1px solid rgba(67,225,192,.25);border-radius:17px;padding:18px;min-height:120px;box-shadow:0 14px 38px rgba(0,0,0,.3)}
    div[data-testid="stMetricLabel"]{color:#9bc9d3!important;font-weight:750!important;text-transform:uppercase;font-size:.75rem!important}
    div[data-testid="stMetricValue"]{color:#fff!important;font-weight:850!important}
    .stButton>button,.stDownloadButton>button{min-height:48px;border-radius:12px;font-weight:800;color:#fff!important;background:linear-gradient(135deg,#00b894,#087f8c);border:1px solid rgba(155,255,230,.35);box-shadow:0 9px 25px rgba(0,130,120,.25)}
    .stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-2px);background:linear-gradient(135deg,#08d6ad,#0798a6)}
    div[data-baseweb="select"]>div,div[data-baseweb="input"]>div{background:rgba(5,35,48,.96)!important;border:1px solid rgba(80,205,190,.3)!important;border-radius:11px;color:white!important}
    input{color:#fff!important}
    .streamlit-expanderHeader{background:rgba(5,40,53,.96);border:1px solid rgba(80,205,190,.22);border-radius:12px;color:white!important;font-weight:750}
    div[data-testid="stDataFrame"],div[data-testid="stPlotlyChart"],div[data-testid="stVegaLiteChart"]{border-radius:14px;overflow:hidden;border:1px solid rgba(80,205,190,.2)}
    hr{border:none;height:1px;background:linear-gradient(90deg,transparent,rgba(80,220,200,.35),transparent)}
    .footer{text-align:center;color:#80aab8;font-size:.78rem;padding-top:1.8rem}
    @media(max-width:900px){.block-container{padding-left:1rem;padding-right:1rem}h1{font-size:2rem!important}}
    </style>
    """,
    unsafe_allow_html=True,
)

# These maps reproduce sklearn LabelEncoder's alphabetical ordering used in the notebook.
CATEGORY_MAPS = {
    "season": {"fall": 0, "springer": 1, "summer": 2, "winter": 3},
    "holiday": {"No": 0, "Yes": 1},
    "workingday": {"No work": 0, "Working day": 1},
    "weathersit": {"Clear": 0, "Heavy Rain": 1, "Light Rain": 2, "Mist": 3},
}
NUMERIC_COLUMNS = [
    "instant", "yr", "mnth", "hr", "weekday", "temp", "atemp", "hum",
    "windspeed", "cnt", "casual", "registered",
]


@st.cache_resource
def load_assets():
    missing = [str(path.name) for path in (MODEL_PATH, FEATURE_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))
    # Loading pickle/joblib files executes Python deserialization; use only trusted project files.
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        model = joblib.load(MODEL_PATH)
        feature_cols = list(joblib.load(FEATURE_PATH))
    if not feature_cols:
        raise ValueError("feature_columns.pkl contains no model features.")
    model_features = getattr(model, "n_features_in_", len(feature_cols))
    if int(model_features) != len(feature_cols):
        raise ValueError(
            f"Model expects {model_features} features, but feature_columns.pkl contains {len(feature_cols)}."
        )
    return model, feature_cols


@st.cache_data
def load_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH.name}")
    df = pd.read_csv(DATA_PATH)
    required = {
        "instant", "dteday", "season", "yr", "mnth", "hr", "holiday",
        "weekday", "workingday", "weathersit", "temp", "atemp", "hum",
        "windspeed", "cnt",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))

    # Dataset dates are stored as DD-MM-YYYY. Parse explicitly to avoid day/month ambiguity.
    df["dteday"] = pd.to_datetime(df["dteday"], dayfirst=True, errors="coerce")
    for col, mapping in CATEGORY_MAPS.items():
        if df[col].dtype == object:
            df[col] = df[col].map(mapping)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["dteday", "cnt", "hr"]).copy()
    df = df[(df["hr"] >= 0) & (df["hr"] <= 23)]
    df = df.sort_values(["dteday", "hr"]).drop_duplicates(["dteday", "hr"], keep="last")
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid hourly observations remain after data cleaning.")

    # Interpolate numeric weather values; fail clearly if a required feature remains unavailable.
    for col in ["temp", "atemp", "hum", "windspeed"]:
        df[col] = df[col].interpolate(limit_direction="both")
        if df[col].isna().all():
            raise ValueError(f"Column '{col}' has no usable numeric values.")
    for col in ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit", "instant"]:
        if df[col].isna().any():
            df[col] = df[col].ffill().bfill()
    return df


def timestamp_for_row(row: pd.Series) -> pd.Timestamp:
    """Build a full timestamp from the dataset's date-only dteday and hour field."""
    return pd.Timestamp(row["dteday"]).normalize() + pd.Timedelta(hours=int(row["hr"]))


def season_code(month: int) -> int:
    # Matches the dataset's season labels and the training LabelEncoder.
    if month in (1, 2, 3):
        return 1  # springer
    if month in (4, 5, 6):
        return 2  # summer
    if month in (7, 8, 9):
        return 0  # fall
    return 3  # winter


def future_row(history: pd.DataFrame) -> tuple[dict, pd.Timestamp]:
    last = history.iloc[-1]
    last_timestamp = timestamp_for_row(last)
    current = last_timestamp + pd.Timedelta(hours=1)
    row = {
        "instant": float(last["instant"]) + 1.0,
        "season": season_code(current.month),
        "yr": 1 if current.year >= 2012 else 0,
        "mnth": current.month,
        "hr": current.hour,
        # The source notebook does not include a future holiday calendar; default to non-holiday.
        "holiday": 0,
        "weekday": current.dayofweek,
        "workingday": 0 if current.dayofweek >= 5 else 1,
        # No future weather forecast is included with the supplied project, so persist latest values.
        "weathersit": int(last["weathersit"]),
        "temp": float(last["temp"]),
        "atemp": float(last["atemp"]),
        "hum": float(last["hum"]),
        "windspeed": float(last["windspeed"]),
    }
    return row, current


def forecast_demand(data: pd.DataFrame, model, features: list[str], days: int) -> pd.DataFrame:
    if not isinstance(days, (int, np.integer)) or days < 1 or days > 365:
        raise ValueError("Forecast days must be an integer between 1 and 365.")
    if len(data) < 1:
        raise ValueError("The dataset contains no historical observations.")
    history = data.copy().reset_index(drop=True)
    results = []
    for _ in range(days * 24):
        row, timestamp = future_row(history)
        missing = [column for column in features if column not in row]
        if missing:
            raise ValueError(f"Missing required model feature(s): {missing}")
        x = pd.DataFrame([{column: row[column] for column in features}], columns=features)
        x = x.apply(pd.to_numeric, errors="coerce")
        if x.isna().any().any():
            bad = x.columns[x.isna().any()].tolist()
            raise ValueError(f"Invalid values in model feature(s): {bad}")
        prediction = float(np.asarray(model.predict(x)).reshape(-1)[0])
        if not np.isfinite(prediction):
            raise ValueError(f"The model returned a non-finite prediction at {timestamp}.")
        prediction = max(0.0, prediction)
        results.append({"timestamp": timestamp, "hour": timestamp.hour, "Predicted_cnt": prediction})

        # Carry the generated timestamp and forecast forward without shifting the hour twice.
        next_row = dict(row)
        next_row.update({"dteday": timestamp.normalize(), "hr": timestamp.hour, "cnt": prediction})
        for col in history.columns:
            if col not in next_row:
                next_row[col] = np.nan
        history = pd.concat([history, pd.DataFrame([next_row], columns=history.columns)], ignore_index=True)
    result = pd.DataFrame(results)
    if len(result) != days * 24 or result["timestamp"].isna().any():
        raise RuntimeError("Forecast validation failed: output length or timestamps are invalid.")
    return result


try:
    model, feature_columns = load_assets()
    df = load_data()
    missing_feature_columns = [c for c in feature_columns if c not in df.columns and c not in {
        "instant", "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"
    }]
    if missing_feature_columns:
        raise ValueError("Model features not found in dataset or forecast logic: " + ", ".join(missing_feature_columns))
except Exception as exc:
    st.error(f"Application setup failed: {exc}")
    st.info("Keep application.py, gbr_model.pkl, feature_columns.pkl, and Dataset.csv in the same folder. Install dependencies from requirements.txt.")
    st.stop()

# Sidebar controls and status
st.sidebar.title("🚲 BikeShare AI")
st.sidebar.caption("BIKE RENTAL DEMAND FORECASTING")
st.sidebar.divider()
st.sidebar.subheader("🎯 Forecast Settings")
forecast_days = st.sidebar.number_input("Forecast Horizon (Days)", min_value=1, max_value=365, value=30, step=1)
st.sidebar.caption(f"Hourly predictions: {int(forecast_days) * 24:,}")
st.sidebar.divider()
st.sidebar.subheader("🟢 System Status")
st.sidebar.success("Model loaded")
st.sidebar.success("Dataset loaded")
st.sidebar.success("Forecast engine ready")
with st.sidebar.expander("🤖 Model Information"):
    st.write("**Model:** Gradient Boosting Regressor")
    st.write("**Target Variable:** `cnt` (total bike rentals)")
    st.write("**Forecast Frequency:** Hourly")
    st.write("**Forecast Type:** Recursive multi-step forecast")
    st.write("**Maximum Forecast Horizon:** 365 days")
    st.write("**Features:** Time, season, weather, temperature, humidity, windspeed and calendar indicators.")
with st.sidebar.expander("📊 Dataset Information"):
    st.metric("Observations", f"{len(df):,}")
    st.metric("Model Features", f"{len(feature_columns)}")
    st.write(f"**Data Start:** {df['dteday'].min():%d %b %Y}")
    st.write(f"**Data End:** {df['dteday'].max():%d %b %Y}")
with st.sidebar.expander("⚙️ Model Details"):
    st.write("**Algorithm:** Gradient Boosting Regressor")
    st.write("**Frequency:** Hourly")
    st.write("**Target:** Total hourly bike rentals")
    st.write("**Weather assumption:** Latest observed weather conditions are carried forward.")
    st.write("**Holiday assumption:** Future holiday flag defaults to 0 because no holiday calendar is supplied.")
st.sidebar.divider()
st.sidebar.subheader("📡 Data Status")
st.sidebar.success("Bike sharing dataset loaded successfully")
st.sidebar.caption(f"{len(df):,} hourly observations available")

# Main dashboard
st.title("🚲 BikeShare Demand Intelligence")
st.markdown('<p class="intro">Machine-learning-powered bike rental demand forecasting using the supplied Gradient Boosting model. Explore historical usage, forecast future hourly rentals, and export your predictions.</p>', unsafe_allow_html=True)
st.header("📈 Historical Overview")
latest = float(df["cnt"].iloc[-1])
avg = float(df["cnt"].mean())
peak = float(df["cnt"].max())
latest_time = timestamp_for_row(df.iloc[-1])
m1, m2, m3, m4 = st.columns(4)
m1.metric("🚲 Latest Rentals", f"{latest:,.0f}")
m2.metric("📊 Historical Average", f"{avg:,.0f}")
m3.metric("🔥 Peak Hourly Rentals", f"{peak:,.0f}")
m4.metric("🕒 Last Observation", latest_time.strftime("%d %b %Y %H:%M"))
st.subheader("Historical Bike Rental Demand — Last 7 Days")
st.caption("Hourly rental counts across the latest week in the supplied dataset.")
hist = df.tail(24 * 7).copy()
hist["timestamp"] = hist.apply(timestamp_for_row, axis=1)
st.line_chart(hist.set_index("timestamp")[["cnt"]], height=390)
c1, c2 = st.columns(2)
with c1:
    st.subheader("🌦️ Rentals by Weather")
    weather = df.groupby("weathersit", dropna=True)["cnt"].mean().rename(index={0: "Clear", 1: "Heavy Rain", 2: "Light Rain", 3: "Mist"})
    st.bar_chart(weather)
with c2:
    st.subheader("🕓 Average Rentals by Hour")
    st.bar_chart(df.groupby("hr")["cnt"].mean())

st.header("🔮 Future Bike Rental Forecast")
st.markdown(f'<p class="intro">Generate <strong style="color:white">{int(forecast_days) * 24:,} hourly predictions</strong> for the next <strong style="color:white">{int(forecast_days):,} day(s)</strong>.</p>', unsafe_allow_html=True)
if st.button("🚲 Generate Forecast", type="primary", use_container_width=True):
    with st.spinner(f"Generating {int(forecast_days)}-day bike rental forecast..."):
        try:
            forecast_result = forecast_demand(df, model, feature_columns, int(forecast_days))
            st.session_state["bike_forecast_df"] = forecast_result
            st.session_state["bike_forecast_days"] = int(forecast_days)
            st.success("Forecast generated and validated successfully.")
        except Exception as exc:
            st.session_state.pop("bike_forecast_df", None)
            st.error(f"Forecast generation failed: {exc}")

if "bike_forecast_df" in st.session_state:
    forecast_df = st.session_state["bike_forecast_df"].copy()
    st.divider()
    st.header("📊 Forecast Results")
    mean_fc = float(forecast_df["Predicted_cnt"].mean())
    max_fc = float(forecast_df["Predicted_cnt"].max())
    min_fc = float(forecast_df["Predicted_cnt"].min())
    peak_row = forecast_df.loc[forecast_df["Predicted_cnt"].idxmax()]
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("📈 Average Forecast", f"{mean_fc:,.0f} rentals/hr")
    f2.metric("🔥 Forecast Peak", f"{max_fc:,.0f}")
    f3.metric("📉 Forecast Minimum", f"{min_fc:,.0f}")
    f4.metric("🕒 Expected Peak Time", peak_row["timestamp"].strftime("%d %b %H:%M"))
    st.subheader("🔮 Forecasted Bike Rental Demand")
    st.caption("Hourly predicted total rentals under the assumptions shown in Model Details.")
    st.line_chart(forecast_df.set_index("timestamp")[["Predicted_cnt"]], height=420)
    s1, s2, s3 = st.columns(3)
    s1.metric("🧮 Forecast Hours", f"{len(forecast_df):,}")
    s2.metric("📅 Forecast Period", f"{forecast_df['timestamp'].min():%d %b %Y %H:%M} — {forecast_df['timestamp'].max():%d %b %Y %H:%M}")
    s3.metric("🚲 Total Predicted Rentals", f"{forecast_df['Predicted_cnt'].sum():,.0f}")
    st.subheader("🗂️ Hourly Forecast Data")
    display = forecast_df.copy()
    display["timestamp"] = display["timestamp"].dt.strftime("%d-%b-%Y %H:%M")
    display["Predicted_cnt"] = display["Predicted_cnt"].round(2)
    st.dataframe(display, use_container_width=True, height=430, hide_index=True)
    st.subheader("⬇️ Export Forecast")
    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Forecast CSV",
        data=csv_bytes,
        file_name=f"bike_sharing_{st.session_state.get('bike_forecast_days', forecast_days)}_day_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown('<p class="footer">🚲 Bike Sharing Demand Forecasting &nbsp;•&nbsp; Gradient Boosting Regression &nbsp;•&nbsp; Streamlit</p>', unsafe_allow_html=True)
