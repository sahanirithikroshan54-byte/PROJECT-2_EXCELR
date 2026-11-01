#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
============================================================
🏍️ BIKE SHARING RENTAL INTELLIGENCE
============================================================
Historical bike rental analytics and 30-day hourly forecasting.

Forecast source:
    next_30_days_forecast(4).csv

Historical source:
    Dataset(4).csv

IMPORTANT:
    The supplied forecast is used without retraining.
    Only exact zero predictions are replaced with 1.
    All predictions greater than 1 remain unchanged.
============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="🏍️ Bike Sharing Rental Intelligence",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "Dataset(4).csv"

FORECAST_FILE = (
    BASE_DIR / "next_30_days_forecast(4).csv"
)


# ============================================================
# MOTORCYCLE BACKGROUND AND PROFESSIONAL THEME
# ============================================================

st.markdown(
    """
    <style>

    .stApp {

        background:
            linear-gradient(
                rgba(1, 8, 18, 0.88),
                rgba(2, 12, 25, 0.93)
            ),
            url(
                "https://images.unsplash.com/photo-1558981806-ec527fa84c39?auto=format&fit=crop&w=2600&q=90"
            );

        background-size: cover;

        background-position: center;

        background-attachment: fixed;

        color: #edf7ff;
    }


    .main {

        position: relative;

        z-index: 1;
    }


    .block-container {

        max-width: 1500px;

        padding-top: 2.2rem;

        padding-bottom: 3rem;

        padding-left: 2.5rem;

        padding-right: 2.5rem;
    }


    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                rgba(2, 10, 24, 0.99),
                rgba(4, 24, 45, 0.99)
            );

        border-right:
            1px solid
            rgba(255, 153, 51, 0.25);
    }


    section[data-testid="stSidebar"] * {

        color: #edf7ff !important;
    }


    h1 {

        color: #ffffff !important;

        font-size: 2.65rem !important;

        font-weight: 850 !important;

        letter-spacing: -0.8px;

        text-shadow:
            0 4px 20px
            rgba(0,0,0,0.75);
    }


    h2 {

        color: #ffffff !important;

        font-weight: 820 !important;

        margin-top: 1.5rem !important;
    }


    h3 {

        color: #e8f4ff !important;

        font-weight: 780 !important;
    }


    p, li {

        color: #d4e6f6;

        line-height: 1.6;
    }


    .intro-text {

        color: #b5cde2;

        font-size: 1.02rem;

        line-height: 1.7;

        margin-bottom: 1.4rem;
    }


    div[data-testid="stMetric"] {

        background:
            linear-gradient(
                145deg,
                rgba(15, 43, 70, 0.96),
                rgba(4, 20, 39, 0.98)
            );

        border:
            1px solid
            rgba(255, 153, 51, 0.30);

        border-radius: 17px;

        padding: 19px;

        min-height: 120px;

        box-shadow:
            0 12px 30px
            rgba(0,0,0,0.30);
    }


    div[data-testid="stMetricLabel"] {

        color: #a9c7df !important;

        font-weight: 750 !important;

        font-size: 0.76rem !important;

        text-transform: uppercase;
    }


    div[data-testid="stMetricValue"] {

        color: #ffffff !important;

        font-size: 1.75rem !important;

        font-weight: 850 !important;
    }


    .stButton > button,
    .stDownloadButton > button {

        min-height: 46px;

        border-radius: 12px;

        font-weight: 800;

        color: #ffffff !important;

        background:
            linear-gradient(
                135deg,
                #f28c28,
                #d95716
            );

        border:
            1px solid
            rgba(255, 190, 100, 0.40);

        transition: all 0.2s ease;
    }


    .stButton > button:hover,
    .stDownloadButton > button:hover {

        transform: translateY(-2px);

        background:
            linear-gradient(
                135deg,
                #ffab45,
                #ed6c25
            );
    }


    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {

        background:
            rgba(7, 31, 54, 0.96) !important;

        border:
            1px solid
            rgba(255, 153, 51, 0.25) !important;

        border-radius: 11px;

        color: #ffffff !important;
    }


    .streamlit-expanderHeader {

        background:
            rgba(8, 32, 55, 0.96);

        border:
            1px solid
            rgba(255, 153, 51, 0.20);

        border-radius: 12px;

        color: #ffffff !important;

        font-weight: 750;
    }


    div[data-testid="stDataFrame"] {

        border-radius: 14px;

        overflow: hidden;

        border:
            1px solid
            rgba(255, 153, 51, 0.20);
    }


    div[data-testid="stAlert"] {

        border-radius: 12px;
    }


    hr {

        border: none;

        height: 1px;

        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(255, 153, 51, 0.45),
                transparent
            );

        margin: 1.4rem 0;
    }


    .footer-text {

        color: #91abc2;

        text-align: center;

        font-size: 0.78rem;

        padding: 1.8rem 0 0.5rem 0;
    }


    @media (max-width: 900px) {

        .block-container {

            padding-left: 1rem;

            padding-right: 1rem;
        }

        h1 {

            font-size: 2rem !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_historical_data():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Historical dataset not found: {DATA_FILE.name}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "dteday",
        "hr",
        "cnt",
        "casual",
        "registered"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing historical columns: {missing}"
        )

    # --------------------------------------------------------
    # Parse historical date
    # --------------------------------------------------------

    df["Datetime"] = pd.to_datetime(
        df["dteday"],
        dayfirst=True,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Parse numerical columns
    # --------------------------------------------------------

    numeric_columns = [
        "instant",
        "yr",
        "mnth",
        "hr",
        "weekday",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "casual",
        "registered",
        "cnt"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col].replace("?", np.nan),
                errors="coerce"
            )

    # --------------------------------------------------------
    # Remove invalid target/date observations
    # --------------------------------------------------------

    df = df.dropna(
        subset=["Datetime", "cnt"]
    )

    df = df.sort_values(
        "Datetime"
    ).reset_index(drop=True)

    if df.empty:

        raise ValueError(
            "Historical dataset contains no valid observations."
        )

    return df


# ============================================================
# FORECAST DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_forecast_data():

    if not FORECAST_FILE.exists():

        raise FileNotFoundError(
            f"Forecast file not found: {FORECAST_FILE.name}"
        )

    forecast = pd.read_csv(
        FORECAST_FILE
    )

    required_columns = [
        "Datetime",
        "Predicted_cnt"
    ]

    missing = [
        col for col in required_columns
        if col not in forecast.columns
    ]

    if missing:

        raise ValueError(
            f"Missing forecast columns: {missing}"
        )

    # --------------------------------------------------------
    # Parse forecast datetime
    # --------------------------------------------------------

    forecast["Datetime"] = pd.to_datetime(
        forecast["Datetime"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Parse forecast values
    # --------------------------------------------------------

    forecast["Predicted_cnt"] = pd.to_numeric(
        forecast["Predicted_cnt"],
        errors="coerce"
    )

    forecast = forecast.dropna(
        subset=[
            "Datetime",
            "Predicted_cnt"
        ]
    ).copy()

    forecast = forecast.sort_values(
        "Datetime"
    ).reset_index(drop=True)

    if forecast.empty:

        raise ValueError(
            "The forecast file contains no valid predictions."
        )

    if len(forecast) != 720:

        raise ValueError(
            f"Expected 720 hourly predictions, "
            f"but found {len(forecast)}."
        )

    if not np.isfinite(
        forecast["Predicted_cnt"].to_numpy()
    ).all():

        raise ValueError(
            "Forecast contains non-finite values."
        )

    if (forecast["Predicted_cnt"] < 0).any():

        raise ValueError(
            "Forecast contains negative rental predictions."
        )

    # --------------------------------------------------------
    # CRITICAL FORECAST RULE
    #
    # Replace exact zeros with 1.
    # Keep every other prediction unchanged.
    # --------------------------------------------------------

    forecast["Predicted_cnt"] = (
        forecast["Predicted_cnt"].mask(
            forecast["Predicted_cnt"].eq(0),
            1
        )
    )

    return forecast


# ============================================================
# LOAD APPLICATION DATA
# ============================================================

try:

    df = load_historical_data()

    forecast_df = load_forecast_data()

except Exception as error:

    st.error(
        f"Application setup failed: {error}"
    )

    st.info(
        "Keep Dataset(4).csv and "
        "next_30_days_forecast(4).csv "
        "in the same folder as application.py."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🏍️ Bike Rental AI"
)

st.sidebar.caption(
    "BIKE SHARING RENTAL INTELLIGENCE"
)

st.sidebar.divider()


# ============================================================
# FORECAST SETTINGS
# ============================================================

st.sidebar.subheader(
    "🎯 Forecast Settings"
)

forecast_days = st.sidebar.number_input(
    "Forecast Horizon (Days)",
    min_value=30,
    max_value=30,
    value=30,
    step=1
)

st.sidebar.caption(
    "720 hourly predictions"
)

st.sidebar.caption(
    "30 days × 24 hours"
)

st.sidebar.divider()


# ============================================================
# SYSTEM STATUS
# ============================================================

st.sidebar.subheader(
    "🟢 System Status"
)

st.sidebar.success(
    "Historical dataset loaded"
)

st.sidebar.success(
    "Forecast CSV loaded"
)

st.sidebar.success(
    "30-day forecast ready"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.sidebar.expander(
    "🤖 Model Information",
    expanded=False
):

    st.write(
        "**Algorithm:** Gradient Boosting Regressor"
    )

    st.write(
        "**Target Variable:** cnt"
    )

    st.write(
        "**Forecast Frequency:** Hourly"
    )

    st.write(
        "**Forecast Horizon:** 30 days"
    )

    st.write(
        "**Forecast Type:** Recursive Multi-Step"
    )

    st.write(
        "**Estimators:** 500"
    )

    st.write(
        "**Learning Rate:** 0.05"
    )

    st.write(
        "**Maximum Depth:** 7"
    )

    st.write(
        "**Random State:** 50"
    )

    st.caption(
        "Predictions are loaded from the supplied "
        "forecast CSV. The model is not retrained "
        "inside this dashboard."
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

with st.sidebar.expander(
    "📊 Dataset Information",
    expanded=False
):

    st.metric(
        "Historical Records",
        f"{len(df):,}"
    )

    st.metric(
        "Historical Features",
        f"{len(df.columns):,}"
    )

    st.metric(
        "Forecast Records",
        f"{len(forecast_df):,}"
    )

    st.write(
        "**Historical Start:** "
        + df["Datetime"].min().strftime(
            "%d %b %Y"
        )
    )

    st.write(
        "**Historical End:** "
        + df["Datetime"].max().strftime(
            "%d %b %Y"
        )
    )


# ============================================================
# DATA STATUS
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "📡 Data Status"
)

st.sidebar.success(
    "Bike rental data available"
)

st.sidebar.caption(
    f"{len(df):,} hourly observations"
)

st.sidebar.caption(
    f"{len(forecast_df):,} forecast observations"
)


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🏍️ Bike Sharing Rental Intelligence"
)

st.markdown(
    """
    <p class="intro-text">
        Machine learning powered bike rental analytics
        and 30-day hourly rental forecasting using the
        supplied Gradient Boosting forecast.
        Monitor rental activity, historical performance,
        expected demand and future rental patterns.
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HISTORICAL OVERVIEW
# ============================================================

st.header(
    "📊 Historical Rental Overview"
)


# ============================================================
# HISTORICAL KPI VALUES
# ============================================================

latest_value = float(
    df["cnt"].iloc[-1]
)

average_value = float(
    df["cnt"].mean()
)

peak_value = float(
    df["cnt"].max()
)

minimum_value = float(
    df["cnt"].min()
)

latest_timestamp = (
    df["Datetime"].iloc[-1]
)


# ============================================================
# HISTORICAL KPI METRICS
# ============================================================

m1, m2, m3, m4 = st.columns(4)


with m1:

    st.metric(
        "🏍️ Latest Rentals",
        f"{latest_value:,.0f}"
    )


with m2:

    st.metric(
        "📈 Average Rentals",
        f"{average_value:,.1f}"
    )


with m3:

    st.metric(
        "🔥 Peak Rentals",
        f"{peak_value:,.0f}"
    )


with m4:

    st.metric(
        "📅 Last Observation",
        latest_timestamp.strftime(
            "%d %b %Y"
        )
    )


# ============================================================
# HISTORICAL RENTAL CHART
# ============================================================

st.subheader(
    "📉 Historical Bike Rentals"
)

st.caption(
    "Hourly bike rental activity across the historical dataset."
)

historical_chart = (
    df[
        ["Datetime", "cnt"]
    ]
    .set_index("Datetime")
)

st.line_chart(
    historical_chart,
    height=400
)


# ============================================================
# HISTORICAL DAILY SUMMARY
# ============================================================

st.subheader(
    "📆 Historical Daily Rental Summary"
)

daily_history = (
    df.assign(
        RentalDate=df["Datetime"].dt.date
    )
    .groupby("RentalDate", as_index=False)
    .agg(
        Total_Rentals=("cnt", "sum"),
        Average_Hourly_Rentals=("cnt", "mean"),
        Peak_Hourly_Rentals=("cnt", "max")
    )
)

daily_history["RentalDate"] = pd.to_datetime(
    daily_history["RentalDate"]
)

daily_history = daily_history.sort_values(
    "RentalDate"
)

st.line_chart(
    daily_history.set_index("RentalDate")[
        ["Total_Rentals"]
    ],
    height=350
)


# ============================================================
# HISTORICAL RENTAL DISTRIBUTION
# ============================================================

with st.expander(
    "📊 Historical Rental Statistics",
    expanded=False
):

    h1, h2, h3 = st.columns(3)

    with h1:

        st.metric(
            "Minimum Rentals",
            f"{minimum_value:,.0f}"
        )

    with h2:

        st.metric(
            "Median Rentals",
            f"{df['cnt'].median():,.1f}"
        )

    with h3:

        st.metric(
            "Total Historical Rentals",
            f"{df['cnt'].sum():,.0f}"
        )

    st.dataframe(
        daily_history.tail(30),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FORECAST SECTION
# ============================================================

st.divider()

st.header(
    "🔮 Future Bike Rental Forecast"
)

st.markdown(
    """
    <p class="intro-text">
        The next 30 days of hourly bike rental predictions
        are loaded from the forecast generated in your
        notebook. The original predicted values are
        preserved, except that exact zero values are
        displayed as one rental.
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FORECAST BUTTON
# ============================================================

generate_forecast_button = st.button(
    "🏍️ Generate 30-Day Forecast",
    type="primary",
    use_container_width=True
)


# ============================================================
# FORECAST SESSION STATE
# ============================================================

if "bike_forecast_ready" not in st.session_state:

    st.session_state[
        "bike_forecast_ready"
    ] = False


if generate_forecast_button:

    st.session_state[
        "bike_forecast_ready"
    ] = True

    st.success(
        "30-day bike rental forecast loaded successfully."
    )


# ============================================================
# FORECAST RESULTS
# ============================================================

if st.session_state["bike_forecast_ready"]:

    st.divider()

    st.header(
        "📈 Forecast Results"
    )


    # ========================================================
    # FORECAST SUMMARY
    # ========================================================

    average_forecast = float(
        forecast_df["Predicted_cnt"].mean()
    )

    maximum_forecast = float(
        forecast_df["Predicted_cnt"].max()
    )

    minimum_forecast = float(
        forecast_df["Predicted_cnt"].min()
    )

    total_forecast = float(
        forecast_df["Predicted_cnt"].sum()
    )

    peak_row = forecast_df.loc[
        forecast_df["Predicted_cnt"].idxmax()
    ]

    peak_time = peak_row["Datetime"]


    # ========================================================
    # FORECAST KPI METRICS
    # ========================================================

    f1, f2, f3, f4 = st.columns(4)


    with f1:

        st.metric(
            "📊 Average Hourly Rentals",
            f"{average_forecast:,.2f}"
        )


    with f2:

        st.metric(
            "🔥 Forecast Peak",
            f"{maximum_forecast:,.2f}"
        )


    with f3:

        st.metric(
            "⬇️ Forecast Minimum",
            f"{minimum_forecast:,.2f}"
        )


    with f4:

        st.metric(
            "🏆 Expected Peak Time",
            peak_time.strftime(
                "%d %b %H:%M"
            )
        )


    # ========================================================
    # TOTAL RENTAL FORECAST
    # ========================================================

    st.subheader(
        "🏍️ Expected Total Rentals"
    )

    st.metric(
        "Total Predicted Rentals — 30 Days",
        f"{total_forecast:,.0f}"
    )


    # ========================================================
    # HOURLY FORECAST CHART
    # ========================================================

    st.subheader(
        "📈 Hourly Bike Rental Forecast"
    )

    st.caption(
        "All 720 hourly predictions from the supplied CSV."
    )

    forecast_chart = (
        forecast_df[
            ["Datetime", "Predicted_cnt"]
        ]
        .set_index("Datetime")
    )

    st.line_chart(
        forecast_chart,
        height=450
    )


    # ========================================================
    # DAILY FORECAST SUMMARY
    # ========================================================

    st.subheader(
        "📆 Daily Rental Forecast"
    )

    daily_forecast = (
        forecast_df.assign(
            ForecastDate=forecast_df["Datetime"].dt.date
        )
        .groupby("ForecastDate", as_index=False)
        .agg(
            Predicted_Rentals=(
                "Predicted_cnt",
                "sum"
            ),
            Average_Hourly_Rentals=(
                "Predicted_cnt",
                "mean"
            ),
            Peak_Hourly_Rentals=(
                "Predicted_cnt",
                "max"
            ),
            Minimum_Hourly_Rentals=(
                "Predicted_cnt",
                "min"
            )
        )
    )

    daily_forecast["ForecastDate"] = pd.to_datetime(
        daily_forecast["ForecastDate"]
    )

    st.line_chart(
        daily_forecast.set_index(
            "ForecastDate"
        )[["Predicted_Rentals"]],
        height=400
    )


    # ========================================================
    # FORECAST STATISTICS
    # ========================================================

    st.subheader(
        "📊 Forecast Statistics"
    )

    s1, s2, s3 = st.columns(3)


    with s1:

        st.metric(
            "Forecast Hours",
            f"{len(forecast_df):,}"
        )


    with s2:

        forecast_period = (
            forecast_df["Datetime"].min().strftime(
                "%d %b %Y"
            )
            + " — "
            + forecast_df["Datetime"].max().strftime(
                "%d %b %Y"
            )
        )

        st.metric(
            "Forecast Period",
            forecast_period
        )


    with s3:

        st.metric(
            "Forecast Days",
            f"{daily_forecast.shape[0]:,}"
        )


    # ========================================================
    # HOURLY FORECAST TABLE
    # ========================================================

    st.subheader(
        "🕒 Hourly Forecast Data"
    )

    display_df = forecast_df.copy()

    display_df["Datetime"] = (
        display_df["Datetime"].dt.strftime(
            "%d-%b-%Y %H:%M"
        )
    )

    display_df["Predicted_cnt"] = (
        display_df["Predicted_cnt"].round(6)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        height=450,
        hide_index=True
    )


    # ========================================================
    # DAILY FORECAST TABLE
    # ========================================================

    st.subheader(
        "📅 Daily Forecast Details"
    )

    daily_display = daily_forecast.copy()

    daily_display["ForecastDate"] = (
        daily_display["ForecastDate"].dt.strftime(
            "%d-%b-%Y"
        )
    )

    numeric_daily_columns = [
        "Predicted_Rentals",
        "Average_Hourly_Rentals",
        "Peak_Hourly_Rentals",
        "Minimum_Hourly_Rentals"
    ]

    daily_display[numeric_daily_columns] = (
        daily_display[numeric_daily_columns].round(2)
    )

    st.dataframe(
        daily_display,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # EXPORT FORECAST
    # ========================================================

    st.subheader(
        "⬇️ Export Forecast"
    )

    csv_data = forecast_df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download 30-Day Forecast CSV",
        data=csv_data,
        file_name="bike_rental_30_day_forecast.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ========================================================
    # EXPORT DAILY SUMMARY
    # ========================================================

    daily_csv = daily_forecast.to_csv(
        index=False
    )

    st.download_button(
        label="📅 Download Daily Forecast Summary",
        data=daily_csv,
        file_name="bike_rental_daily_forecast.csv",
        mime="text/csv",
        use_container_width=True
    )


    # ========================================================
    # FORECAST VALIDATION
    # ========================================================

    st.subheader(
        "🛡️ Forecast Validation"
    )

    zero_replacements = int(
        (
            forecast_df["Predicted_cnt"] == 1
        ).sum()
    )

    st.success(
        "All forecast values are non-negative "
        "and the forecast contains 720 hourly records."
    )

    st.info(
        "Exact zero values in the supplied forecast "
        "were replaced with 1. Values above 1 were "
        "not recalculated or modified."
    )


# ============================================================
# HISTORICAL DATA EXPLORER
# ============================================================

st.divider()

st.header(
    "🔎 Historical Data Explorer"
)

with st.expander(
    "Explore Historical Bike Rental Records",
    expanded=False
):

    search_start = st.date_input(
        "Start Date",
        value=df["Datetime"].min().date(),
        min_value=df["Datetime"].min().date(),
        max_value=df["Datetime"].max().date(),
        key="history_start"
    )

    search_end = st.date_input(
        "End Date",
        value=df["Datetime"].max().date(),
        min_value=df["Datetime"].min().date(),
        max_value=df["Datetime"].max().date(),
        key="history_end"
    )

    if search_start > search_end:

        st.warning(
            "Start date must be before or equal to end date."
        )

    else:

        filtered_history = df.loc[
            df["Datetime"].dt.date.between(
                search_start,
                search_end
            )
        ].copy()

        st.metric(
            "Filtered Records",
            f"{len(filtered_history):,}"
        )

        st.dataframe(
            filtered_history,
            use_container_width=True,
            height=400,
            hide_index=True
        )

        historical_csv = filtered_history.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Historical Data",
            data=historical_csv,
            file_name="bike_rental_historical_data.csv",
            mime="text/csv"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <p class="footer-text">
        🏍️ Bike Sharing Rental Intelligence
        &nbsp; • &nbsp;
        Gradient Boosting Regression
        &nbsp; • &nbsp;
        30-Day Hourly Forecast
        &nbsp; • &nbsp;
        Streamlit
    </p>
    """,
    unsafe_allow_html=True
)

