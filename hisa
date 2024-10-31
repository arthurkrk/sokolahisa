import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Share Prices',
# Draw the actual page
# -----------------------------------------------------------------------------
import streamlit as st
import matplotlib.pyplot as plt
import datetime
import plotly.graph_objs as go

import yfinance as yf
import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

# Specify title and logo for the webpage.
# Set up your web app
st.set_page_config(page_title="Stock Price App", page_icon="📈")

sidebar = st.sidebar
sidebar.header("Stock Price App")
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Ticker", "AAPL")
with col2:
    start_date = st.date_input("Start Date", datetime.date(2000, 1, 1))
    end_date = st.date_input("End Date", datetime.date.today())

#Download data
data = yf.download(ticker, start_date, end_date)

st.title=f"{ticker} Stock Price"
st.eps_trend

st.line_chart(data['Close'],x_label="Date",y_label="Close")
# -----------------------------------------------------------------------------
