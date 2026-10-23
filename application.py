#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""🏍️ BikeShare Intelligence — Streamlit deployment application.

Place Dataset.csv (optional) and any exported model artifacts beside this file,
or upload a compatible bike-sharing CSV from the sidebar.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent
TARGET = "cnt"
DATE_COL = "dteday"
DROP_FEATURES = {"cnt", "casual", "registered", DATE_COL}

st.set_page_config(
    page_title="BikeShare Intelligence",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 🎨 PREMIUM DARK BIKE-SHARING THEME (same overall structure)
# ============================================================
st.markdown("""
<style>
.stApp {
  background: linear-gradient(rgba(5,16,22,.91),rgba(5,16,22,.95)),
    radial-gradient(circle at 80% 5%,rgba(0,200,150,.20),transparent 32%),
    linear-gradient(135deg,#071b20,#102d32);
  color:#effcf8;
}
.block-container {max-width:1500px;padding-top:2rem;padding-bottom:3rem}
section[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#071b20,#0b292c,#081f24);
  border-right:1px solid rgba(65,220,175,.25)
}
section[data-testid="stSidebar"] * {color:#eafaf5}
h1,h2,h3 {color:#f5fffc!important;letter-spacing:-.35px}
h1 {font-weight:850!important}
p,li {color:#d0e8e1;line-height:1.6}
.intro-text {color:#b9d9d0;font-size:1.03rem;line-height:1.7}
div[data-testid="stMetric"] {
  background:linear-gradient(145deg,rgba(15,62,61,.94),rgba(8,36,40,.97));
  border:1px solid rgba(70,220,175,.24);border-radius:17px;
  padding:18px;min-height:112px;box-shadow:0 12px 32px rgba(0,0,0,.23)
}
div[data-testid="stMetricLabel"] {color:#a9d7c9!important;font-weight:750!important}
div[data-testid="stMetricValue"] {color:#fff!important;font-weight:850!important}
.stButton>button,.stDownloadButton>button {
  border-radius:12px;font-weight:800;min-height:44px;
  background:linear-gradient(135deg,#00a985,#087d76);color:white;border:1px solid #43d9b0
}
div[data-baseweb="select"]>div,div[data-baseweb="input"]>div {
  background:rgba(8,36,40,.95);border-radius:10px
}
div[data-testid="stDataFrame"],div[data-testid="stPlotlyChart"],div[data-testid="stVegaLiteChart"] {
  border-radius:14px;overflow:hidden
}
hr {border-color:rgba(93,204,173,.25)}
.footer-text {text-align:center;color:#7fa99e;font-size:.8rem;padding-top:1.5rem}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 📦 DATA PREPARATION
# ============================================================
REQUIRED_COLUMNS = {DATE_COL, TARGET}
DEFAULT_FEATURES = [
    "season", "yr", "mnth", "hr", "holiday", "weekday",
    "workingday", "weathersit", "temp", "atemp", "hum",
    "windspeed", "instant",
]

def normalize_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the original Bike Sharing dataset."""
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    aliases = {
        "date": DATE_COL,
        "datetime": DATE_COL,
        "count": TARGET,
        "total": TARGET,
        "total_rentals": TARGET,

