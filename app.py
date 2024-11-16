#import packages
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import appdirs as ad
import pandas as pd
import math
from pathlib import Path
import requests
from bs4 import BeautifulSoup

# Specify title and logo for the webpage.
# Set up your web app
import streamlit as st
import sqlite3
import yfinance as yf
import datetime
from datetime import date, timedelta
from datetime import datetime
ad.user_cache_dir = lambda *args: "/tmp"
#Specify title and logo for the webpage.
st.set_page_config(
    page_title="Stock Price App",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)
# Define a global header for all pages
def render_header(title):
    st.markdown(f"""
    <div style="background-color:#1f4e79;padding:10px;border-radius:5px">
        <h1 style="color:white;text-align:center;">{title}</h1>
    </div>
    """, unsafe_allow_html=True)
# Define a global footer for the app
def render_footer():
    st.markdown("""
    ---
    <div style="text-align:center;">
        <small>© 2024 International University of Japan. All rights reserved.</small>
    </div>
    """, unsafe_allow_html=True)

# Page Title
render_header("S&P 500 Industry Analysis")
# Create tabs
tabs = st.tabs(["Home","Fundamental Analysis", "Technical Analysis", "Comparison", "News", "Contacts"])

# Home
with tabs[0]:
    st.header("Home")
    st.write("Our web app provides insights into stock market trends and helps in making data-driven investment decisions.")
    st.image(
        "https://st3.depositphotos.com/3108485/32120/i/600/depositphotos_321205098-stock-photo-businessman-plan-graph-growth-and.jpg",
        caption="Placeholder image for the Home page."
    )
# Fundamental Analysis
with tabs[1]:
    st.header("Fundamental Analysis")
    st.write("Analyze a firm's prospects using fundamental analysis. Enter a stock ticker below:")

    ticker = st.text_input("Stock Ticker (e.g., AAPL, MSFT):", value="AAPL")

    def analyze_stock_fundamentals(ticker):
        """Perform fundamental analysis for the given stock ticker."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            st.subheader(f"Fundamental Analysis for {ticker.upper()}")
           # Company Overview
            st.write("### Company Overview")
            st.write(f"**Name:** {info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Website:** [Visit Website]({info.get('website', '#')})")
            st.markdown("---")

            # Key Financial Metrics
            market_cap = info.get('marketCap', 0) / 1e9
            pe_ratio = info.get('trailingPE', 'N/A')
            pb_ratio = info.get('priceToBook', 'N/A')
            dividend_yield = info.get('dividendYield', 0) * 100
            forward_pe = info.get('forwardPE', 'N/A')
            st.write("### Key Financial Metrics")
            st.write(f"**Market Cap:** ${market_cap:.2f} Billion")
            st.write(f"**Trailing P/E Ratio:** {pe_ratio}")
            st.write(f"**Forward P/E Ratio:** {forward_pe}")
            st.write(f"**Price-to-Book Ratio:** {pb_ratio}")
            st.write(f"**Dividend Yield:** {dividend_yield:.2f}%")

            st.markdown("---")         
            # Earnings and Growth
            earnings_growth = info.get('earningsGrowth', 'N/A')
            revenue_growth = info.get('revenueGrowth', 'N/A')
            st.write("### Earnings and Growth")
            st.write(f"**Earnings Growth:** {earnings_growth}")
            st.write(f"**Revenue Growth:** {revenue_growth}")
            st.markdown("---")

            # Debt Ratios
            total_debt = info.get('totalDebt', 0)
            free_cashflow = info.get('freeCashflow', 0)
            debt_to_equity = info.get('debtToEquity', 'N/A')
            st.write("### Debt Ratios")
            st.write(f"**Total Debt:** ${total_debt:,}")
            st.write(f"**Free Cash Flow:** ${free_cashflow:,}")
            st.write(f"**Debt-to-Equity Ratio:** {debt_to_equity}")             
            st.markdown("---")
            # Valuation Analysis
            if pe_ratio != 'N/A' and pb_ratio != 'N/A':
                if pe_ratio < 15 and pb_ratio < 1.5:
                    st.success("The stock appears **undervalued**.")
                elif pe_ratio > 25 or pb_ratio > 3:
                    st.warning("The stock appears **overvalued**.")
                else:
                    st.info("The stock has a **neutral valuation**.")
            else:
                st.error("Insufficient data to determine valuation.")
            st.markdown("---")

            # Dividend Analysis
            if dividend_yield > 0:
                st.write(f"The stock offers a **dividend yield of {dividend_yield:.2f}%**.")
            else:
                st.write("The stock does not pay a dividend.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    if ticker:
        analyze_stock_fundamentals(ticker)

# Technical Analysis
with tabs[2]:
    st.header("Stock Information")
    st.write("Select one stock to analyze and visualize.")

    # App title and description
    st.title("Enhanced Stock Information Web App")
    st.write("Enter a ticker symbol to retrieve and visualize stock information interactively.")

    start_date = st.date_input("Start Date", value=datetime(2022, 1, 1), key="start_date")
    end_date = st.date_input("End Date", value=datetime.now(), key="end_date")

    # Ticker input
    ticker_symbol = st.text_input("Enter stock ticker (e.g., AAPL, MSFT):", "AAPL", key="ticker")

    # Recommendation toggle
    show_recommendation = st.checkbox("Show Recommendation", key="show_recommendation")

    # Chart type selection
    chart_type = st.radio("Select Chart Type", ["Line Chart", "Candlestick Chart"])

    # Fetch data and ensure valid date range
    if start_date > end_date:
        st.error("End date must be after the start date. Please adjust your dates.")
    elif ticker_symbol:
        try:
            # Fetch stock data using yfinance
            stock = yf.Ticker(ticker_symbol)
            data = stock.history(start=start_date, end=end_date)

            if data.empty:
                st.warning(f"No data found for {ticker_symbol} in the selected date range.")
            else:
                # Display current price
                current_price = data['Close'].iloc[-1]
                price_change = current_price - data['Close'].iloc[-2]
                percentage_change = (price_change / data['Close'].iloc[-2]) * 100
                price_class = "positive" if price_change > 0 else "negative"

                st.markdown(
                    f"### Current Price: **{current_price:.2f} USD** "
                    f"({price_change:+.2f} / {percentage_change:+.2f}%)"
                )
                
                # Indicator toggles
                st.write("### Select Indicators")
                show_sma = st.checkbox("Simple Moving Average (SMA)")
                show_ema = st.checkbox("Exponential Moving Average (EMA)")
                show_rsi = st.checkbox("Relative Strength Index (RSI)")
                show_macd = st.checkbox("Moving Average Convergence Divergence (MACD)")
                show_vwap = st.checkbox("Volume Weighted Average Price (VWAP)")

                # Initialize buy signal count
                buy_signals = 0
                total_indicators = 0

                # Simple Moving Average (SMA)
                if show_sma:
                    sma_period = st.sidebar.slider("SMA Period", 5, 100, 20)
                    data['SMA'] = data['Close'].rolling(window=sma_period).mean()
                    st.line_chart(data[['Close', 'SMA']])
                    if data['Close'].iloc[-1] > data['SMA'].iloc[-1]:
                        buy_signals += 1
                    total_indicators += 1

                # Exponential Moving Average (EMA)
                if show_ema:
                    ema_period = st.sidebar.slider("EMA Period", 5, 100, 20)
                    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
                    if data['Close'].iloc[-1] > data['EMA'].iloc[-1]:
                        buy_signals += 1
                    total_indicators += 1

                # Relative Strength Index (RSI)
                if show_rsi:
                    rsi_period = st.sidebar.slider("RSI Period", 5, 50, 14)
                    delta = data['Close'].diff(1)
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=rsi_period).mean()
                    avg_loss = loss.rolling(window=rsi_period).mean()
                    rs = avg_gain / avg_loss
                    data['RSI'] = 100 - (100 / (1 + rs))
                    if data['RSI'].iloc[-1] < 30:
                        buy_signals += 1
                    total_indicators += 1

                # Determine Buy or Sell Recommendation
                if show_recommendation:
                    st.write("### Recommendation Summary")
                    st.write(f"Total Indicators: {total_indicators}")
                    st.write(f"Buy Signals: {buy_signals}")
                    st.write(f"Sell Signals: {total_indicators - buy_signals}")

                    if buy_signals > total_indicators / 2:
                        st.success("**Overall Recommendation: Buy**")
                    else:
                        st.warning("**Overall Recommendation: Sell**")

                # Plot stock price
                st.subheader(f"{ticker_symbol} Price Chart")
                fig = go.Figure()

                if chart_type == "Line Chart":
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name="Close Price"))
                elif chart_type == "Candlestick Chart":
                    fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name="Candlestick"
                    ))

                fig.update_layout(
                    title=f"{ticker_symbol} Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    xaxis_rangeslider_visible=(chart_type == "Candlestick Chart")
                )
                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Could not retrieve data for {ticker_symbol}. Error: {e}")


# Tab: Comparison
with tabs[3]:
    st.header("Comparison")
    st.write("This is the Visualization page. Show your plots here.")
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import date, timedelta
    import yfinance as yf
    import pandas as pd
    # Title for date and stock selection
    st.title('Select Date and Stocks')
    # Date range selection
    today = date.today()
    min_date = today - timedelta(days=365 * 5)
    max_date = today
    date_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(today - timedelta(days=365), today)
    )
    sdate, edate = date_range
    # Stock selection
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "TESLA","BRK.B","META","KO","UNH","XOM","LLY","JPM","JNJ","V","PG","MA","AVGO","HD","CVX","MRK","ABBV","COST","PEP","ADBE"]
    selected_stocks = st.multiselect(
        "Select Stocks", symbols, default=["AAPL"]
    )
    # Stock comparison
    st.title("Comparison")
    if selected_stocks:
        # Fetch stock data
        data = yf.download(
            selected_stocks,
            start=sdate,
            end=edate,
            interval="1d",
            auto_adjust=True,
            prepost=True
        )
        if data.empty:
            st.error("Failed to fetch historical data or no data available for the selected period.")
        else:
            # Filter data for the selected date range
            filtered_data = data['Close'][selected_stocks]
            sdate_utc = pd.to_datetime(sdate).tz_localize('UTC')
            edate_utc = pd.to_datetime(edate).tz_localize('UTC')
            filtered_data = filtered_data[(filtered_data.index >= sdate_utc) & (filtered_data.index <= edate_utc)]
            if not filtered_data.empty:
                # Reset index to create a 'Date' column
                filtered_data = filtered_data.reset_index()
                filtered_data = filtered_data.rename(columns={'index': 'Date'})
                # Plot the data
                st.line_chart(
                    filtered_data,
                    x="Date",
                    y=selected_stocks[0] if len(selected_stocks) == 1 else selected_stocks
                )
            else:
                st.warning("No data available for the selected stock(s) and date range.")
    else:
        st.warning("Please select at least one stock.")

# Tab: News
with tabs[4]:
    st.header("News")
    st.write("Stay updated with the latest news on your selected stock.")


# Render the footer on all pages
render_footer()
