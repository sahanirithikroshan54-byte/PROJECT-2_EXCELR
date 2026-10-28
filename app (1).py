# -*- coding: utf-8 -*-
"""Bike Sharing Intelligence — Streamlit dashboard.

Run locally with: streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline import load_dataset, train_and_forecast, recursive_forecast

st.set_page_config(
    page_title="Bike Sharing Intelligence",
    page_icon="🏍️",
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

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL APPLICATION BACKGROUND
       ======================================================== */

    .stApp {

        background:
            linear-gradient(rgba(1, 8, 20, 0.78), rgba(2, 14, 31, 0.90)),
            url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxOTIwIiBoZWlnaHQ9IjEwODAiIHZpZXdCb3g9IjAgMCAxOTIwIDEwODAiPgo8ZGVmcz4KIDxsaW5lYXJHcmFkaWVudCBpZD0ic2t5IiB4MT0iMCIgeTE9IjAiIHgyPSIxIiB5Mj0iMSI+PHN0b3Agb2Zmc2V0PSIwIiBzdG9wLWNvbG9yPSIjMDQxNTJiIi8+PHN0b3Agb2Zmc2V0PSIuNTUiIHN0b3AtY29sb3I9IiMwODJhNGEiLz48c3RvcCBvZmZzZXQ9IjEiIHN0b3AtY29sb3I9IiMwNjEzMjYiLz48L2xpbmVhckdyYWRpZW50PgogPGxpbmVhckdyYWRpZW50IGlkPSJyb2FkIiB4MT0iMCIgeTE9IjAiIHgyPSIwIiB5Mj0iMSI+PHN0b3Agc3RvcC1jb2xvcj0iIzA5Mjk0NSIvPjxzdG9wIG9mZnNldD0iMSIgc3RvcC1jb2xvcj0iIzAyMDkxNSIvPjwvbGluZWFyR3JhZGllbnQ+CiA8cmFkaWFsR3JhZGllbnQgaWQ9Imdsb3ciPjxzdG9wIHN0b3AtY29sb3I9IiMwMGNmZmYiIHN0b3Atb3BhY2l0eT0iLjM2Ii8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMDBjZmZmIiBzdG9wLW9wYWNpdHk9IjAiLz48L3JhZGlhbEdyYWRpZW50PgogPGxpbmVhckdyYWRpZW50IGlkPSJzdW4iIHgxPSIwIiB5MT0iMCIgeDI9IjEiIHkyPSIxIj48c3RvcCBzdG9wLWNvbG9yPSIjNTVlOGZmIiBzdG9wLW9wYWNpdHk9Ii45Ii8+PHN0b3Agb2Zmc2V0PSIxIiBzdG9wLWNvbG9yPSIjMGE2N2I1IiBzdG9wLW9wYWNpdHk9Ii4xIi8+PC9saW5lYXJHcmFkaWVudD4KPC9kZWZzPgo8cmVjdCB3aWR0aD0iMTkyMCIgaGVpZ2h0PSIxMDgwIiBmaWxsPSJ1cmwoI3NreSkiLz4KPGNpcmNsZSBjeD0iMTQ4MCIgY3k9IjIxMCIgcj0iNDIwIiBmaWxsPSJ1cmwoI2dsb3cpIi8+CjxjaXJjbGUgY3g9IjI5MCIgY3k9IjczMCIgcj0iMzQwIiBmaWxsPSJ1cmwoI2dsb3cpIiBvcGFjaXR5PSIuNDUiLz4KPCEtLSBkaXN0YW50IHNreWxpbmUgLS0+CjxnIGZpbGw9IiMwNzFjMzQiIHN0cm9rZT0iIzBiM2M2MCIgc3Ryb2tlLXdpZHRoPSIzIiBvcGFjaXR5PSIuOTYiPgogPHBhdGggZD0iTTAgNjEwVjM5MGg4MHYtNzVoNjV2Mjk1eiIvPjxwYXRoIGQ9Ik0xMTUgNjEwVjQzMGg3NXYtOTVoNTJ2Mjc1eiIvPgogPHBhdGggZD0iTTIyNSA2MTBWMzYwaDkwdi05MGg0NnYzNDB6Ii8+PHBhdGggZD0iTTMzMCA2MTBWNDIwaDYydi0xMjVoNzV2MzE1eiIvPgogPHBhdGggZD0iTTQ0MCA2MTBWMzc1aDgydi02NWg1NXYzMDB6Ii8+PHBhdGggZD0iTTU0MCA2MTBWNDUwaDgwdi0xMDBoNTV2MjYweiIvPgogPHBhdGggZD0iTTY1NSA2MTBWMzg1aDU0di05MGg4NXYzMTV6Ii8+PHBhdGggZD0iTTc4MCA2MTBWNDM1aDk1di0xNTVoNTB2MzMweiIvPgogPHBhdGggZD0iTTkwMCA2MTBWMzU1aDcydi05NWg2MnYzNTB6Ii8+PHBhdGggZD0iTTEwMTAgNjEwVjQyNWg3NXYtMTE1aDU0djMwMHoiLz4KIDxwYXRoIGQ9Ik0xMTI1IDYxMFYzODBoOTV2LTcwaDU1djMwMHoiLz48cGF0aCBkPSJNMTI0NSA2MTBWNDQ1aDcwdi0xNDVoODB2MzEweiIvPgogPHBhdGggZD0iTTEzODUgNjEwVjM1MGg3MHYtODBoODJ2MzQweiIvPjxwYXRoIGQ9Ik0xNTMwIDYxMFY0MjBoODB2LTEyMGg2MHYzMTB6Ii8+CiA8cGF0aCBkPSJNMTY3MCA2MTBWMzkwaDg4di02NWg1OHYyODV6Ii8+PHBhdGggZD0iTTE3OTAgNjEwVjQ1NWg3MHYtMTA1aDYwdjI2MHoiLz4KPC9nPgo8IS0tIGdsb3dpbmcgd2luZG93cyAtLT4KPGcgZmlsbD0iIzI4Y2ZmZiIgb3BhY2l0eT0iLjI1Ij4KIDxwYXRoIGQ9Ik0zMCA0MjVoOXYxMmgtOXptMjQgMGg5djEyaC05em0yNCAwaDl2MTJoLTl6TTI1NSAzMjBoMTB2MTNoLTEwem0yNCAwaDEwdjEzaC0xMHptMCAzNGgxMHYxM2gtMTB6TTQ3MCA0MDJoOXYxMmgtOXptMjUgMGg5djEyaC05ek05MzAgMzg4aDEwdjEyaC0xMHptMjUgMGgxMHYxMmgtMTB6bTAgMzJoMTB2MTJoLTEwek0xNDEwIDM5MGgxMHYxMmgtMTB6bTI1IDBoMTB2MTJoLTEwek0xNzEwIDQzMGgxMHYxMmgtMTB6bTI1IDBoMTB2MTJoLTEweiIvPgo8L2c+CjwhLS0gcm9hZCBhbmQgYmlrZSBsYW5lIC0tPgo8cGF0aCBkPSJNMCA2MTBRNjAwIDU2NSA5NjAgNjMwVDE5MjAgNjAwVjEwODBIMFoiIGZpbGw9InVybCgjcm9hZCkiLz4KPHBhdGggZD0iTTAgNzQwUTU1MCA2ODAgOTYwIDczMFQxOTIwIDY5MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMGI0YjcyIiBzdHJva2Utd2lkdGg9IjQiIG9wYWNpdHk9Ii44Ii8+CjxwYXRoIGQ9Ik0wIDkxMFE1NTAgODI1IDk2MCA4ODBUMTkyMCA4MjUiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzBlNjY4ZiIgc3Ryb2tlLXdpZHRoPSIzIiBvcGFjaXR5PSIuNTUiLz4KPHBhdGggZD0iTTAgMTAxMFE1NTAgOTE1IDk2MCA5ODBUMTkyMCA5MjAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzBlNjY4ZiIgc3Ryb2tlLXdpZHRoPSIyIiBvcGFjaXR5PSIuNCIvPgo8cGF0aCBkPSJNMjMwIDEwODBMODgwIDYzME0xNjgwIDEwODBMMTExMCA2MzAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzBlNGU3OCIgc3Ryb2tlLXdpZHRoPSIzIiBvcGFjaXR5PSIuNDUiLz4KPHBhdGggZD0iTTkxMCAxMDgwbDM4LTE4MGgyMGwtMTAgMTgwIiBmaWxsPSIjMWJjOWY0IiBvcGFjaXR5PSIuMTYiLz4KPCEtLSBzdHlsaXplZCBiaWN5Y2xlIHNpbGhvdWV0dGUgLS0+CjxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEwOTAgNDU1KSIgZmlsbD0ibm9uZSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj4KIDxjaXJjbGUgY3g9IjAiIGN5PSIxMDUiIHI9Ijc4IiBzdHJva2U9IiMwNWM5ZmYiIHN0cm9rZS13aWR0aD0iOCIgb3BhY2l0eT0iLjkiLz4KIDxjaXJjbGUgY3g9IjI3MCIgY3k9IjEwNSIgcj0iNzgiIHN0cm9rZT0iIzA1YzlmZiIgc3Ryb2tlLXdpZHRoPSI4IiBvcGFjaXR5PSIuOSIvPgogPHBhdGggZD0iTTAgMTA1TDc4IDE1TDE0NSAxMDVIMEwxMDUgMTA1TDE3MCAyNUwyNzAgMTA1TDIwNSAxMDUiIHN0cm9rZT0iIzE3YzhlZiIgc3Ryb2tlLXdpZHRoPSI5Ii8+CiA8cGF0aCBkPSJNNzggMTVoNDhNMTcwIDI1bC0xOC0yOGgtMzBNMTQ1IDEwNWwyMi03OCIgc3Ryb2tlPSIjNDVlNGZmIiBzdHJva2Utd2lkdGg9IjgiLz4KIDxwYXRoIGQ9Ik0xNDAgLThoNDhNMTYwIC04bC0xNCAyMyIgc3Ryb2tlPSIjNDVlNGZmIiBzdHJva2Utd2lkdGg9IjciLz4KIDxjaXJjbGUgY3g9IjE0NSIgY3k9IjEwNSIgcj0iMTIiIHN0cm9rZT0iIzQ1ZTRmZiIgc3Ryb2tlLXdpZHRoPSI3Ii8+CiA8cGF0aCBkPSJNMTQ0IDEwNWwtNDIgMzVNMTQ0IDEwNWwzNSAzMiIgc3Ryb2tlPSIjNDVlNGZmIiBzdHJva2Utd2lkdGg9IjYiLz4KPC9nPgo8IS0tIHRlYWwgbGlnaHQgdHJhaWwgLS0+CjxwYXRoIGQ9Ik0wIDg2MFE1MjAgNzkwIDk4MCA4NTBUMTkyMCA4MDAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzAwY2ZmZiIgc3Ryb2tlLXdpZHRoPSI1IiBvcGFjaXR5PSIuMTUiLz4KPHBhdGggZD0iTTAgODcwUTUyMCA4MDAgOTgwIDg2MFQxOTIwIDgxMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMDBjZmZmIiBzdHJva2Utd2lkdGg9IjEuNSIgb3BhY2l0eT0iLjUiLz4KPCEtLSBkYXJrIG92ZXJsYXkgZm9yIHJlYWRhYmxlIGFwcCBjYXJkcyAtLT4KPHJlY3Qgd2lkdGg9IjE5MjAiIGhlaWdodD0iMTA4MCIgZmlsbD0iIzAyMGIxOCIgb3BhY2l0eT0iLjM2Ii8+Cjwvc3ZnPg==");


        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
        background-repeat: no-repeat;

        color: #edf7ff;
    }


    /* ========================================================
       PREMIUM BACKGROUND OVERLAY
       ======================================================== */

    .stApp::before {

        content: "";

        position: fixed;

        inset: 0;

        background:

            radial-gradient(
                circle at 82% 12%,
                rgba(0, 174, 255, 0.13),
                transparent 32%
            ),

            radial-gradient(
                circle at 8% 75%,
                rgba(0, 115, 220, 0.11),
                transparent 38%
            ),

            linear-gradient(
                180deg,
                rgba(2, 10, 24, 0.08),
                rgba(2, 10, 24, 0.30)
            );

        pointer-events: none;

        z-index: 0;
    }


    /* ========================================================
       MAIN CONTENT
       ======================================================== */

    .main {

        position: relative;

        z-index: 1;
    }


    .block-container {

        max-width: 1500px;

        padding-top: 2.4rem;
        padding-bottom: 3rem;

        padding-left: 2.8rem;
        padding-right: 2.8rem;
    }


    /* ========================================================
       SIDEBAR - PREMIUM CONTROL PANEL
       ======================================================== */

    section[data-testid="stSidebar"] {

        background:

            linear-gradient(
                180deg,
                rgba(2, 12, 27, 0.985),
                rgba(4, 23, 46, 0.985),
                rgba(2, 17, 35, 0.985)
            );

        border-right:
            1px solid
            rgba(76, 182, 255, 0.28);

        box-shadow:

            12px 0 40px
            rgba(0, 0, 0, 0.50),

            inset -1px 0
            rgba(103, 196, 255, 0.08);
    }


    /* ========================================================
       SIDEBAR TEXT
       ======================================================== */

    section[data-testid="stSidebar"] * {

        color: #edf7ff !important;
    }


    section[data-testid="stSidebar"] h1 {

        color: #ffffff !important;

        font-size: 1.55rem !important;

        font-weight: 850 !important;

        letter-spacing: -0.4px;

        margin-bottom: 0.15rem !important;
    }


    section[data-testid="stSidebar"] h2 {

        color: #ffffff !important;

        font-size: 1.18rem !important;

        font-weight: 800 !important;
    }


    section[data-testid="stSidebar"] h3 {

        color: #f4faff !important;

        font-size: 1.02rem !important;

        font-weight: 780 !important;
    }


    section[data-testid="stSidebar"] p {

        color: #a9c5df !important;

        line-height: 1.55;

        font-size: 0.88rem;
    }


    section[data-testid="stSidebar"] strong {

        color: #ffffff !important;

        font-weight: 800 !important;
    }


    /* ========================================================
       SIDEBAR CAPTION
       ======================================================== */

    section[data-testid="stSidebar"] .stCaption {

        color: #74b8e8 !important;

        font-size: 0.75rem !important;

        font-weight: 700 !important;

        letter-spacing: 1.1px;

        text-transform: uppercase;
    }


    /* ========================================================
       SIDEBAR DIVIDERS
       ======================================================== */

    section[data-testid="stSidebar"] hr {

        border: none;

        height: 1px;

        background:

            linear-gradient(
                90deg,
                transparent,
                rgba(90, 190, 255, 0.28),
                transparent
            );

        margin: 1.15rem 0;
    }


    /* ========================================================
       SIDEBAR INPUT
       ======================================================== */

    section[data-testid="stSidebar"]
    div[data-baseweb="input"] > div {

        background:

            linear-gradient(
                145deg,
                rgba(9, 34, 61, 0.98),
                rgba(5, 24, 47, 0.98)
            ) !important;

        border:
            1px solid
            rgba(92, 190, 250, 0.28) !important;

        border-radius: 13px;

        min-height: 44px;

        box-shadow:

            inset 0 1px 0
            rgba(255,255,255,0.035),

            0 5px 16px
            rgba(0,0,0,0.18);
    }


    section[data-testid="stSidebar"] input {

        color: #ffffff !important;

        font-weight: 700 !important;
    }


    /* ========================================================
       SIDEBAR EXPANDERS
       ======================================================== */

    section[data-testid="stSidebar"]
    .streamlit-expanderHeader {

        background:

            linear-gradient(
                135deg,
                rgba(9, 37, 67, 0.98),
                rgba(5, 25, 48, 0.98)
            );

        border:
            1px solid
            rgba(96, 190, 246, 0.22);

        border-radius: 13px;

        color: #ffffff !important;

        font-weight: 760;

        padding: 0.82rem 0.9rem;

        box-shadow:

            0 7px 20px
            rgba(0,0,0,0.16);

        transition: all 0.2s ease;
    }


    section[data-testid="stSidebar"]
    .streamlit-expanderHeader:hover {

        border-color:
            rgba(104, 207, 255, 0.46);

        background:

            linear-gradient(
                135deg,
                rgba(12, 48, 84, 0.99),
                rgba(6, 29, 55, 0.99)
            );
    }


    section[data-testid="stSidebar"]
    .streamlit-expanderContent {

        background:

            linear-gradient(
                180deg,
                rgba(4, 22, 42, 0.98),
                rgba(3, 17, 34, 0.98)
            );

        border:
            1px solid
            rgba(81, 170, 230, 0.14);

        border-top: none;

        border-radius:
            0 0 13px 13px;

        color: #d7e9f8 !important;

        padding-top: 0.65rem;
    }


    /* ========================================================
       SIDEBAR METRICS
       ======================================================== */

    section[data-testid="stSidebar"]
    div[data-testid="stMetric"] {

        background:

            linear-gradient(
                145deg,
                rgba(9, 38, 68, 0.90),
                rgba(4, 25, 49, 0.92)
            );

        border:
            1px solid
            rgba(91, 188, 247, 0.17);

        border-radius: 12px;

        padding: 10px 12px;

        margin-bottom: 8px;
    }


    section[data-testid="stSidebar"]
    div[data-testid="stMetricLabel"] {

        color: #8fb5d7 !important;

        font-size: 0.72rem !important;

        font-weight: 750 !important;

        text-transform: uppercase;
    }


    section[data-testid="stSidebar"]
    div[data-testid="stMetricValue"] {

        color: #ffffff !important;

        font-size: 1.18rem !important;

        font-weight: 820 !important;
    }


    /* ========================================================
       MAIN TITLE
       ======================================================== */

    h1 {

        color: #ffffff !important;

        font-size: 2.75rem !important;

        font-weight: 850 !important;

        letter-spacing: -1.1px;

        line-height: 1.12;

        margin-bottom: 0.35rem !important;

        text-shadow:
            0 4px 20px
            rgba(0,0,0,0.72);
    }


    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    h2 {

        color: #f5fbff !important;

        font-size: 1.65rem !important;

        font-weight: 820 !important;

        letter-spacing: -0.35px;

        margin-top: 1.8rem !important;

        margin-bottom: 0.8rem !important;

        text-shadow:
            0 3px 14px
            rgba(0,0,0,0.62);
    }


    h3 {

        color: #e7f6ff !important;

        font-size: 1.22rem !important;

        font-weight: 780 !important;

        margin-top: 1.25rem !important;

        margin-bottom: 0.65rem !important;
    }


    /* ========================================================
       NORMAL TEXT
       ======================================================== */

    p,
    li {

        color: #d2e6f7;

        line-height: 1.6;
    }


    /* ========================================================
       INTRO TEXT
       ======================================================== */

    .intro-text {

        color: #a9c7df;

        font-size: 1.02rem;

        line-height: 1.7;

        margin-bottom: 1.4rem;

        text-shadow:
            0 2px 8px
            rgba(0,0,0,0.60);
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {

        background:

            linear-gradient(
                145deg,
                rgba(12, 44, 78, 0.94),
                rgba(4, 24, 47, 0.97)
            );

        border:
            1px solid
            rgba(99, 194, 255, 0.25);

        border-radius: 18px;

        padding: 20px;

        min-height: 125px;

        box-shadow:

            0 14px 38px
            rgba(0,0,0,0.34),

            inset 0 1px 0
            rgba(255,255,255,0.045);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }


    div[data-testid="stMetric"]:hover {

        transform:
            translateY(-3px);

        box-shadow:

            0 18px 42px
            rgba(0,0,0,0.43),

            0 0 24px
            rgba(0,157,255,0.13);
    }


    div[data-testid="stMetric"] label {

        color: #91b8d9 !important;

        font-size: 0.74rem !important;

        font-weight: 760 !important;

        letter-spacing: 0.75px;

        text-transform: uppercase;
    }


    div[data-testid="stMetricValue"] {

        color: #ffffff !important;

        font-size: 1.80rem !important;

        font-weight: 850 !important;

        letter-spacing: -0.5px;

        text-shadow:
            0 2px 11px
            rgba(0,0,0,0.42);
    }


    div[data-testid="stMetricDelta"] {

        color: #72d9ff !important;

        font-weight: 700 !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button,
    .stDownloadButton > button {

        min-height: 48px;

        border-radius: 13px;

        font-size: 0.94rem;

        font-weight: 800;

        letter-spacing: 0.25px;

        color: #ffffff !important;

        background:

            linear-gradient(
                135deg,
                #00a0e8,
                #006bd6
            );

        border:
            1px solid
            rgba(133, 221, 255, 0.34);

        box-shadow:

            0 9px 25px
            rgba(0, 103, 205, 0.28);

        transition:
            all 0.2s ease;
    }


    .stButton > button:hover,
    .stDownloadButton > button:hover {

        transform:
            translateY(-2px);

        background:

            linear-gradient(
                135deg,
                #08b1f2,
                #0879e3
            );

        box-shadow:

            0 13px 31px
            rgba(0, 123, 235, 0.38);
    }


    /* ========================================================
       SELECTBOXES / INPUTS
       ======================================================== */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {

        background:

            rgba(7, 31, 57, 0.94) !important;

        border:
            1px solid
            rgba(96, 189, 245, 0.24) !important;

        border-radius: 11px;

        color: #ffffff !important;
    }


    input {

        color: #ffffff !important;
    }


    /* ========================================================
       MAIN EXPANDERS
       ======================================================== */

    .streamlit-expanderHeader {

        background:

            linear-gradient(
                135deg,
                rgba(9, 37, 67, 0.96),
                rgba(6, 26, 49, 0.96)
            );

        border:
            1px solid
            rgba(95, 185, 240, 0.20);

        border-radius: 12px;

        color: #f2f8ff !important;

        font-weight: 750;

        padding: 0.8rem 1rem;
    }


    .streamlit-expanderContent {

        background:
            rgba(5, 22, 42, 0.94);

        border:
            1px solid
            rgba(100, 180, 240, 0.12);

        border-top: none;

        border-radius:
            0 0 12px 12px;

        color: #d8e9f8 !important;
    }


    /* ========================================================
       CHART CONTAINERS
       ======================================================== */

    div[data-testid="stVegaLiteChart"],
    div[data-testid="stArrowVegaLiteChart"],
    div[data-testid="stPlotlyChart"] {

        background:

            linear-gradient(
                145deg,
                rgba(5, 27, 51, 0.92),
                rgba(3, 19, 38, 0.96)
            );

        border:
            1px solid
            rgba(91, 184, 244, 0.17);

        border-radius: 17px;

        padding: 11px;

        box-shadow:
            0 13px 32px
            rgba(0,0,0,0.25);
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {

        border-radius: 15px;

        overflow: hidden;

        border:
            1px solid
            rgba(103, 190, 245, 0.18);

        box-shadow:
            0 13px 30px
            rgba(0,0,0,0.28);

        background:
            rgba(4, 21, 40, 0.94);
    }


    /* ========================================================
       ALERT / INFO BOXES
       ======================================================== */

    div[data-testid="stAlert"] {

        border-radius: 12px;

        background:
            rgba(6, 29, 54, 0.94);

        border:
            1px solid
            rgba(97, 190, 242, 0.20);
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {

        border: none;

        height: 1px;

        background:

            linear-gradient(
                90deg,
                transparent,
                rgba(110, 196, 248, 0.32),
                transparent
            );

        margin: 1.45rem 0;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer-text {

        color: #6f91af;

        text-align: center;

        font-size: 0.76rem;

        padding: 1.8rem 0 0.5rem 0;

        letter-spacing: 0.25px;
    }


    /* ========================================================
       RESPONSIVE
       ======================================================== */

    @media (max-width: 900px) {

        .block-container {

            padding-left: 1rem;
            padding-right: 1rem;
        }

        h1 {

            font-size: 2rem !important;
        }

        h2 {

            font-size: 1.4rem !important;
        }

        div[data-testid="stMetric"] {

            min-height: 105px;

            padding: 15px;
        }

        div[data-testid="stMetricValue"] {

            font-size: 1.4rem !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)




# ============================================================
# LOAD DATA, TRAIN A LEAKAGE-SAFE MODEL, AND BUILD FORECAST
# ============================================================

@st.cache_resource(show_spinner=False)
def get_bundle(data_path: str, modified_time: float):
    # modified_time is part of the cache key: replacing Dataset.csv retrains the model.
    return train_and_forecast(data_path, BASE_DIR)


if not DATA_PATH.exists():
    st.error("⚠️ Dataset.csv was not found. Place the supplied bike rental CSV beside app.py.")
    st.stop()

with st.sidebar:
    st.title("🏍️ BikeShare AI")
    st.caption("BIKE RENTAL ANALYTICS & FORECASTING")
    st.divider()
    page = st.radio(
        "🧭 Navigate",
        [
            "📌 Dashboard Overview",
            "🔎 Exploratory Data Analysis",
            "🕒 Rental Patterns",
            "🤖 Model Evaluation",
            "🔮 Demand Forecast",
            "🗂️ Dataset & Model Information",
        ],
    )
    st.divider()
    st.subheader("🟢 System Status")
    st.success("Dataset file found")
    if st.button("🔄 Retrain & refresh model", use_container_width=True):
        get_bundle.clear()
        st.rerun()
    st.caption("Retraining runs a chronological 30-day backtest, then fits the final model on all observed history.")
    with st.expander("🧠 Forecast assumptions", expanded=False):
        st.write("**Target:** `cnt` — total rentals per hour")
        st.write("**Model:** HistGradientBoostingRegressor")
        st.write("**Forecast horizon:** Up to 30 days (720 hours)")
        st.write("**Leakage control:** `casual` and `registered` are excluded.")
        st.write("**Future weather:** Not modeled; no future weather feed is included.")
    try:
        with st.spinner("Preparing data, validating the model, and generating the forecast…"):
            bundle = get_bundle(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    except Exception as exc:
        st.error(f"⚠️ The model pipeline could not start: {exc}")
        st.info("Check that Dataset.csv contains dteday, hr, and cnt, and that requirements.txt is installed.")
        st.stop()

df = bundle["df"].copy()
series = bundle["series"].copy()
model = bundle["model"]
comparison = bundle["comparison"].copy()
forecast = bundle["forecast"].copy()
report = bundle["report"]
feature_columns = bundle["feature_columns"]
importance = bundle["importance"].copy()
actual_vs_predicted = bundle["actual_vs_predicted"].copy()

# Sidebar data-quality details
with st.sidebar.expander("📊 Dataset Information", expanded=False):
    st.metric("Observations", f"{len(df):,}")
    st.metric("Columns", f"{len(df.columns):,}")
    st.write(f"**Start date:** {df['timestamp'].min():%d %b %Y}")
    st.write(f"**End date:** {df['timestamp'].max():%d %b %Y}")
with st.sidebar.expander("🧹 Data Quality", expanded=False):
    st.write(f"**Duplicate rows:** {int(df['timestamp'].duplicated().sum()):,}")
    st.write(f"**Missing hourly slots:** {int(report['missing_hour_slots_in_dataset']):,}")
    st.write(f"**Missing values in source rows:** {int(df.isna().sum().sum()):,}")

# ============================================================
# MAIN HEADER
# ============================================================

st.title("🏍️ Bike Sharing Intelligence")
st.markdown(
    """
    <p class="intro-text">
        Machine-learning powered bike rental analytics, demand exploration,
        model evaluation, and leakage-safe recursive hourly demand forecasting.
        <br><b>🚲 Predict • Plan • Promote sustainable mobility</b>
    </p>
    """,
    unsafe_allow_html=True,
)

# Derived tables used by multiple pages
total_rentals = float(df["cnt"].sum())
average_hourly = float(df["cnt"].mean())
peak_rentals = float(df["cnt"].max())
daily = df.groupby("dteday", as_index=False)["cnt"].sum().sort_values("dteday")
hourly = df.groupby("hour", as_index=False)["cnt"].mean().sort_values("hour")
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_average = (
    df.groupby("weekday_name", as_index=False)["cnt"].mean()
    .set_index("weekday_name").reindex(weekday_order).dropna().reset_index()
)
season_average = (
    df.groupby("season", as_index=False)["cnt"].mean()
    if "season" in df.columns else pd.DataFrame()
)
weather_average = (
    df.groupby("weathersit", as_index=False)["cnt"].mean()
    if "weathersit" in df.columns else pd.DataFrame()
)

# ============================================================
# PAGE 1 — DASHBOARD OVERVIEW
# ============================================================

if page == "📌 Dashboard Overview":
    st.header("📊 Historical Rental Overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏍️ Total Rentals", f"{total_rentals:,.0f}")
    m2.metric("📈 Average per Hour", f"{average_hourly:,.1f}")
    m3.metric("🏆 Peak Hourly Rentals", f"{peak_rentals:,.0f}")
    m4.metric("🗓️ Historical Records", f"{len(df):,}")
    st.caption(
        f"Dataset coverage: **{df['timestamp'].min():%d %b %Y} – "
        f"{df['timestamp'].max():%d %b %Y}**. "
        "The current forecast is a statistical projection, not a guarantee."
    )
    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("📈 Daily Bike Rentals")
        st.line_chart(daily.set_index("dteday")["cnt"], height=330)
    with right:
        st.subheader("🕒 Average Rentals by Hour")
        st.bar_chart(hourly.set_index("hour")["cnt"], height=330)
    left, right = st.columns(2)
    with left:
        st.subheader("🌦️ Average Rentals by Weather")
        if not weather_average.empty:
            st.bar_chart(weather_average.set_index("weathersit")["cnt"], height=300)
        else:
            st.info("Weather labels are unavailable in the supplied data.")
    with right:
        st.subheader("🍂 Average Rentals by Season")
        if not season_average.empty:
            st.bar_chart(season_average.set_index("season")["cnt"], height=300)
        else:
            st.info("Season labels are unavailable in the supplied data.")
    st.subheader("👥 Casual vs Registered Rentals")
    if {"casual", "registered"}.issubset(df.columns):
        type_totals = pd.DataFrame({
            "Rental type": ["Casual", "Registered"],
            "Total rentals": [df["casual"].sum(), df["registered"].sum()],
        }).set_index("Rental type")
        st.bar_chart(type_totals, height=260)
        st.caption("These columns are used for descriptive analysis only and are never model inputs.")

# ============================================================
# PAGE 2 — EXPLORATORY DATA ANALYSIS
# ============================================================

elif page == "🔎 Exploratory Data Analysis":
    st.header("🔎 Exploratory Data Analysis")
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(30), use_container_width=True, hide_index=True)
    st.subheader("🧱 Dataset Columns and Data Types")
    schema = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Non-null values": [int(df[c].notna().sum()) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
    })
    st.dataframe(schema, use_container_width=True, hide_index=True)
    st.subheader("📋 Descriptive Statistics")
    st.dataframe(df.select_dtypes(include=np.number).describe().T.round(3), use_container_width=True)
    st.subheader("🧹 Missing Values by Column")
    missing = df.isna().sum().rename("Missing values").to_frame()
    missing = missing[missing["Missing values"] > 0].sort_values("Missing values", ascending=False)
    if missing.empty:
        st.success("No missing values were found in the displayed source rows.")
    else:
        st.dataframe(missing, use_container_width=True)
        st.bar_chart(missing)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.subheader("📊 Numeric Feature Distributions")
    selected_col = st.selectbox("Select a numeric feature", numeric_cols, index=numeric_cols.index("cnt") if "cnt" in numeric_cols else 0)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df[selected_col].dropna(), bins=35, kde=True, ax=ax, color="#27b8f5")
    ax.set_title(f"Distribution of {selected_col}")
    ax.set_xlabel(selected_col)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.subheader("📦 Numeric Feature Outlier Inspection")
    box_col = st.selectbox("Choose a feature for boxplot", numeric_cols, index=numeric_cols.index("cnt") if "cnt" in numeric_cols else 0, key="eda_box_col")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.boxplot(x=df[box_col].dropna(), ax=ax, color="#24c7b8")
    ax.set_title(f"Outlier inspection: {box_col}")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.subheader("🔥 Correlation Between Numeric Features")
    corr_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 1]
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, max(5, len(corr_cols) * 0.48)))
    sns.heatmap(corr, cmap="YlGnBu", center=0, ax=ax, annot=len(corr_cols) <= 10, fmt=".2f")
    ax.set_title("Numeric feature correlation")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption("Correlation is descriptive. In particular, casual + registered equals cnt, so those fields are excluded from predictive modeling.")

# ============================================================
# PAGE 3 — RENTAL PATTERNS
# ============================================================

elif page == "🕒 Rental Patterns":
    st.header("🕒 Bike Rental Patterns")
    st.subheader("⏰ Average Bike Rentals by Hour")
    st.line_chart(hourly.set_index("hour")["cnt"], height=320)
    st.subheader("📅 Daily Rentals Over Time")
    st.line_chart(daily.set_index("dteday")["cnt"], height=330)
    st.subheader("📆 Average Rentals by Weekday")
    st.bar_chart(weekday_average.set_index("weekday_name")["cnt"], height=280)
    st.subheader("🌡️ Weather Variables vs Total Rentals")
    weather_cols = [c for c in ["temp", "atemp", "hum", "windspeed"] if c in df.columns and df[c].notna().any()]
    if weather_cols:
        weather_feature = st.selectbox("Select a weather variable", weather_cols)
        plot_df = df[[weather_feature, "cnt"]].dropna()
        fig, ax = plt.subplots(figsize=(11, 4.5))
        sns.scatterplot(data=plot_df, x=weather_feature, y="cnt", alpha=0.28, s=18, color="#2D8CFF", ax=ax)
        ax.set_title(f"{weather_feature} vs Total Rentals")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    st.subheader("🗓️ Average Rentals by Hour and Weekday")
    pivot = df.pivot_table(values="cnt", index="hour", columns="weekday_name", aggfunc="mean")
    pivot = pivot.reindex(columns=[c for c in weekday_order if c in pivot.columns])
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.heatmap(pivot, cmap="YlGnBu", ax=ax)
    ax.set_title("Average Rentals: Hour vs Weekday")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Hour of day")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.subheader("👥 Casual and Registered Rentals")
    if {"casual", "registered"}.issubset(df.columns):
        type_totals = pd.DataFrame({
            "Rental type": ["Casual", "Registered"],
            "Total rentals": [df["casual"].sum(), df["registered"].sum()],
        })
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=type_totals, x="Rental type", y="Total rentals", color="#2D8CFF", ax=ax)
        ax.set_title("Casual vs Registered Rental Totals")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Casual and registered counts are not both available.")

# ============================================================
# PAGE 4 — MODEL EVALUATION
# ============================================================

elif page == "🤖 Model Evaluation":
    st.header("🤖 Model Building and Evaluation")
    st.markdown(
        "All metrics below are calculated by a chronological, recursive 30-day backtest. "
        "The held-out period is not used as actual input while generating its predictions."
    )
    st.subheader("🏁 Model Comparison Results")
    st.dataframe(comparison.round(3), use_container_width=True, hide_index=True)
    if not comparison.empty and comparison["MAE"].notna().any():
        best = comparison.loc[comparison["MAE"].idxmin()]
        st.info(f"📌 Best model by MAE on this backtest: **{best['Model']}** (MAE {best['MAE']:.2f}).")
    a, b, c, d = st.columns(4)
    primary = comparison[comparison["Model"].str.startswith("Gradient Boosting")]
    if not primary.empty:
        row = primary.iloc[0]
        a.metric("MAE", f"{row['MAE']:.2f}")
        b.metric("RMSE", f"{row['RMSE']:.2f}")
        c.metric("MAPE", f"{row['MAPE']:.2f}%")
        d.metric("R²", f"{row['R2']:.3f}")
    st.caption(
        f"Backtest window: {report['backtest_start']} to {report['backtest_end']}. "
        f"Actual target values were available for {report['backtest_observed_hours']} hourly slots."
    )
    st.subheader("🎯 Actual vs Predicted Rentals")
    if not actual_vs_predicted.empty:
        chart_df = actual_vs_predicted.set_index("timestamp")[["actual_cnt", "predicted_cnt"]]
        st.line_chart(chart_df, height=360)
        st.dataframe(actual_vs_predicted.head(200).round(2), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download backtest predictions",
            actual_vs_predicted.to_csv(index=False).encode("utf-8"),
            "backtest_actual_vs_predicted.csv",
            "text/csv",
            use_container_width=True,
        )
    st.subheader("🌟 Model Feature Importance")
    if not importance.empty:
        st.bar_chart(importance.set_index("Feature")["Importance"], height=360)
        st.dataframe(importance.round(5), use_container_width=True, hide_index=True)
    else:
        st.info("Feature importance could not be calculated for this run.")
    st.warning("The test score is one historical backtest, not a guarantee of future performance. Future weather and special-event effects are not included.")

# ============================================================
# PAGE 5 — DEMAND FORECAST
# ============================================================

elif page == "🔮 Demand Forecast":
    st.header("🔮 Bike Rental Demand Forecast")
    st.markdown(
        "Generate a recursive hourly forecast using calendar patterns and historical rental lags. "
        "The app deliberately does not use `casual` or `registered` as predictors."
    )
    forecast_days = int(st.number_input(
        "🏍️ Enter forecast duration (days)",
        min_value=1, max_value=30, value=7, step=1,
        help="Generate 1–30 days of hourly rental-demand estimates."
    ))
    st.caption(f"Forecast size: **{forecast_days * 24:,} hourly predictions**")
    st.caption(
        "Future weather is not available in the supplied resources, so this model does not attempt to predict weather."
    )
    st.info(
        f"📅 The supplied history ends at {series.index.max():%d %b %Y, %H:%M}. "
        f"This forecast therefore begins at {(series.index.max() + pd.Timedelta(hours=1)):%d %b %Y, %H:%M}. "
        "Upload a more recent dataset if you need a forecast for the current period."
    )
    if st.button("🏍️ Generate Forecast", type="primary", use_container_width=True):
        try:
            with st.spinner(f"Generating {forecast_days * 24:,} hourly predictions…"):
                generated = recursive_forecast(series, model, days=forecast_days)
                st.session_state["bike_forecast_df"] = generated
                st.session_state["bike_forecast_days"] = forecast_days
        except Exception as exc:
            st.error(f"Forecast generation failed: {exc}")

    if (
        "bike_forecast_df" not in st.session_state
        or st.session_state.get("bike_forecast_days") != forecast_days
    ):
        try:
            st.session_state["bike_forecast_df"] = recursive_forecast(
                series, model, days=forecast_days
            )
            st.session_state["bike_forecast_days"] = forecast_days
        except Exception as exc:
            st.error(f"Could not prepare the selected forecast: {exc}")
            st.stop()

    generated_forecast = st.session_state["bike_forecast_df"].copy()
    if not generated_forecast.empty:
        st.subheader("📊 Generated Forecast Results")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("🔢 Forecast hours", f"{len(generated_forecast):,}")
        f2.metric("📈 Average predicted rentals", f"{generated_forecast['predicted_cnt'].mean():,.1f}")
        f3.metric("🏆 Maximum predicted rentals", f"{generated_forecast['predicted_cnt'].max():,.1f}")
        f4.metric("📉 Minimum predicted rentals", f"{generated_forecast['predicted_cnt'].min():,.1f}")
        st.line_chart(generated_forecast.set_index("timestamp")["predicted_cnt"], height=380)
        st.dataframe(generated_forecast.head(200).round(2), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download generated forecast",
            generated_forecast.to_csv(index=False).encode("utf-8"),
            f"bike_sharing_{forecast_days}_day_forecast.csv",
            "text/csv",
            use_container_width=True,
        )
    st.subheader("📈 Saved 30-Day Forecast")
    st.dataframe(forecast.head(72).round(2), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download full 30-day forecast CSV",
        forecast.to_csv(index=False).encode("utf-8"),
        "next_30_days_forecast.csv",
        "text/csv",
        use_container_width=True,
    )
    st.info("Forecast estimates are for planning and demonstration. Validate on more historical windows before using them for operational decisions.")

# ============================================================
# PAGE 6 — DATASET AND MODEL INFORMATION
# ============================================================

elif page == "🗂️ Dataset & Model Information":
    st.header("🗂️ Dataset and Model Information")
    st.subheader("📁 Project Files")
    project_files = [
        ("Dataset.csv", DATA_PATH, True),
        ("app.py", BASE_DIR / "app.py", True),
        ("pipeline.py", BASE_DIR / "pipeline.py", True),
        ("gbr_model.pkl", MODEL_PATH, False),
        ("feature_columns.pkl", FEATURES_PATH, False),
        ("metadata.pkl", METADATA_PATH, False),
        ("model_comparison.csv", COMPARISON_PATH, False),
        ("next_30_days_forecast.csv", FORECAST_PATH, False),
        ("validation_report.json", BASE_DIR / "validation_report.json", False),
    ]
    file_status = pd.DataFrame([
        {
            "File": name,
            "Required for": "App startup" if required else "Generated model/report artifact",
            "Status": "✅ Found" if path.exists() else "⚠️ Generated on first run" if not required else "⚠️ Missing",
        }
        for name, path, required in project_files
    ])
    st.dataframe(file_status, use_container_width=True, hide_index=True)
    st.subheader("📚 Dataset Columns")
    st.write("The target column is `cnt`, representing total rentals in each observed hour.")
    st.dataframe(pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Non-null values": [int(df[c].notna().sum()) for c in df.columns],
    }), use_container_width=True, hide_index=True)
    st.subheader("🧠 Saved Model Metadata")
    st.json(report)
    st.subheader("🧮 Model Input Features")
    st.code(", ".join(feature_columns), language="text")
    st.subheader("🧰 Preprocessing Notes")
    st.markdown(
        """
        - The source uses date and hour to construct a complete hourly time index.
        - Missing hourly slots are preserved as missing targets for evaluation; prior observed values are forward-filled only when building lag inputs.
        - Calendar features and lag features use only information available before each prediction timestamp.
        - `casual` and `registered` are excluded because they directly sum to `cnt`.
        - Future weather is not modeled because future weather inputs are not supplied.
        - Validation is chronological and recursive over the final 30 hourly days.
        """
    )
    st.subheader("📥 Download Resources")
    st.download_button(
        "⬇️ Download dataset CSV",
        df.drop(columns=["timestamp", "hour", "weekday_name", "month_name", "year"], errors="ignore").to_csv(index=False).encode("utf-8"),
        "Dataset.csv", "text/csv", use_container_width=True,
    )
    st.download_button(
        "⬇️ Download validation report",
        json.dumps(report, indent=2, default=str).encode("utf-8"),
        "validation_report.json", "application/json", use_container_width=True,
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <p class="footer-text">
        🏍️ Bike Sharing Intelligence
        &nbsp;•&nbsp;
        Leakage-Safe Gradient Boosting
        &nbsp;•&nbsp;
        Hourly Rental Demand
        &nbsp;•&nbsp;
        Streamlit
    </p>
    """,
    unsafe_allow_html=True,
)
