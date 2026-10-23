#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
🚲 Bike Sharing Intelligence
Streamlit dashboard based on the supplied BIKE_SHARING_RENTAL.ipynb and project files.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Keep app.py, Dataset.csv, gbr_model.pkl, feature_columns.pkl,
metadata.pkl, model_comparison.csv in the same directory.
"""
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Bike Sharing Intelligence",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset.csv"
MODEL_PATH = BASE_DIR / "gbr_model.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.pkl"
METADATA_PATH = BASE_DIR / "metadata.pkl"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"
FORECAST_PATH = BASE_DIR / "next_30_days_forecast.csv"

# ============================================================
# VISUAL THEME
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
          radial-gradient(circle at 85% 5%, rgba(37, 190, 135, .14), transparent 30%),
          radial-gradient(circle at 5% 75%, rgba(27, 140, 118, .12), transparent 35%),
          linear-gradient(135deg, #071a1b 0%, #0b2426 48%, #07191c 100%);
        color: #edfdf7;
    }
    .block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 3rem; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(5, 24, 25, .99), rgba(8, 39, 37, .99));
        border-right: 1px solid rgba(88, 220, 171, .24);
    }
    section[data-testid="stSidebar"] * { color: #e9fff6; }
    h1, h2, h3 { color: #f5fff9 !important; letter-spacing: -.35px; }
    p, li, label { color: #c8e5da; }
    .intro {
        color: #b6d8ca; font-size: 1.04rem; line-height: 1.7;
        padding: .2rem 0 1rem 0;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(18, 63, 56, .95), rgba(8, 39, 38, .98));
        border: 1px solid rgba(98, 225, 174, .23);
        border-radius: 17px; padding: 18px; min-height: 115px;
        box-shadow: 0 12px 30px rgba(0,0,0,.18);
    }
    div[data-testid="stMetricLabel"] { color: #a9d7c4 !important; font-weight: 700 !important; }
    div[data-testid="stMetricValue"] { color: #f3fff8 !important; font-weight: 800 !important; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 12px; min-height: 45px; font-weight: 750;
        color: #06231c; background: linear-gradient(135deg, #71efb2, #36c98c);
        border: 1px solid rgba(190,255,223,.5);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        color: #041c17; border-color: #b5ffda; filter: brightness(1.04);
    }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background: rgba(7, 35, 33, .96) !important;
        border-color: rgba(104, 211, 168, .3) !important; border-radius: 10px;
    }
    div[data-testid="stDataFrame"], div[data-testid="stVegaLiteChart"],
    div[data-testid="stPlotlyChart"] {
        border: 1px solid rgba(98, 225, 174, .16); border-radius: 14px; overflow: hidden;
    }
    .footer { color: #7fae9c; text-align: center; font-size: .8rem; padding-top: 1.6rem; }
    hr { border-color: rgba(98, 225, 174, .2); }
    </style>
    """,
    unsafe_allow_html=True,
)

