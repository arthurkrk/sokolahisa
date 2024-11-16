#import packages
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import appdirs as ad
import pandas as pd
import numpy as np
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
render_header("S&P 500 Features Analysis")
# Create tabs
tabs = st.tabs(["Home","Fundamental Analysis", "Technical Analysis", "Risk Assessment","Comparison", "News", "Contacts"])

# Home
with tabs[0]:
    st.header("Home")
    st.write("This web app offers valuable insights into stock market trends, empowering you to make smarter, data-driven investment choices.")
    st.image(
        "https://st3.depositphotos.com/3108485/32120/i/600/depositphotos_321205098-stock-photo-businessman-plan-graph-growth-and.jpg",
        )
    st.write("With a good perspective on history, we can have a better understanding of the past and present, and thus a clear vision of the future. ~ Carlos Slim Helu.")
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
    st.write("Analyze and visualize stock performance with indicators and recommendations.")
    # Ticker input
    ticker_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL", key="ticker")
    # Date slicer
    st.write("### Select Date Range")
    today = date.today()
    min_date = today - timedelta(days=365 * 5)  # Allow data up to 5 years back
    max_date = today
    date_range = st.slider("Drag to select the range:",
        min_value=min_date,
        max_value=max_date,
        value=(today - timedelta(days=365), today),
        format="YYYY-MM-DD",
    )
    start_date, end_date = date_range

    # Recommendation toggle
    show_recommendation = st.checkbox("Show Recommendation", key="show_recommendation")

    # Indicator toggles
    st.write("### Select Indicators")
    indicators = {
        "SMA_0_50": st.checkbox("SMA (0-50)", key="show_sma_0_50"),
        "SMA_50_100": st.checkbox("SMA (50-100)", key="show_sma_50_100"),
        "RSI": st.checkbox("Relative Strength Index (RSI)", key="show_rsi"),
        "MACD": st.checkbox("Moving Average Convergence Divergence (MACD)", key="show_macd"),
        "VWAP": st.checkbox("Volume Weighted Average Price (VWAP)", key="show_vwap"),
    }
    if ticker_symbol:
        try:
            # Fetch stock data
            stock = yf.Ticker(ticker_symbol)
            data = stock.history(start=start_date, end=end_date)

            if data.empty:
                st.warning(f"No data found for {ticker_symbol} in the selected range.")
            else:
                # Display current price
                current_price = data['Close'].iloc[-1]
                price_change = current_price - data['Close'].iloc[-2]
                percentage_change = (price_change / data['Close'].iloc[-2]) * 100

                st.markdown(
                    f"### Current Price: **${current_price:.2f}** "
                    f"({price_change:+.2f}, {percentage_change:+.2f}%)"
                )

                # Add selected indicators
                buy_signals = 0
                total_indicators = 0

                # Create Plotly figure for all charts
                fig = go.Figure()

                # Line chart for close price
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name="Close Price"))

                # Moving Averages (SMA)
                if indicators["SMA_0_50"]:
                    sma_0_50 = st.slider("SMA (0-50) Period", 1, 50, 20, key="sma_0_50_period")
                    data['SMA_0_50'] = data['Close'].rolling(window=sma_0_50).mean()
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_0_50'], mode='lines', name="SMA (0-50)"))
                    if data['Close'].iloc[-1] > data['SMA_0_50'].iloc[-1]:
                        buy_signals += 1
                    total_indicators += 1

                if indicators["SMA_50_100"]:
                    sma_50_100 = st.slider("SMA (50-100) Period", 50, 100, 75, key="sma_50_100_period")
                    data['SMA_50_100'] = data['Close'].rolling(window=sma_50_100).mean()
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50_100'], mode='lines', name="SMA (50-100)"))
                    if data['Close'].iloc[-1] > data['SMA_50_100'].iloc[-1]:
                        buy_signals += 1
                    total_indicators += 1

                # Relative Strength Index (RSI)
                if indicators["RSI"]:
                    rsi_period = st.slider("RSI Period", 5, 50, 14, key="rsi_period")
                    delta = data['Close'].diff()
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    avg_gain = gain.rolling(window=rsi_period).mean()
                    avg_loss = loss.rolling(window=rsi_period).mean()
                    rs = avg_gain / avg_loss
                    data['RSI'] = 100 - (100 / (1 + rs))
                    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name="RSI", yaxis="y2"))
                    if data['RSI'].iloc[-1] < 30:
                        buy_signals += 1
                    total_indicators += 1

                # Moving Average Convergence Divergence (MACD)
                if indicators["MACD"]:
                    short_span = st.slider("MACD Short Span", 5, 50, 12, key="macd_short")
                    long_span = st.slider("MACD Long Span", 5, 100, 26, key="macd_long")
                    signal_span = st.slider("MACD Signal Span", 5, 20, 9, key="macd_signal")
                    data['MACD'] = data['Close'].ewm(span=short_span).mean() - data['Close'].ewm(span=long_span).mean()
                    data['Signal Line'] = data['MACD'].ewm(span=signal_span).mean()
                    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name="MACD", yaxis="y3"))
                    fig.add_trace(go.Scatter(x=data.index, y=data['Signal Line'], mode='lines', name="Signal Line", yaxis="y3"))
                    if data['MACD'].iloc[-1] > data['Signal Line'].iloc[-1]:
                        buy_signals += 1
                    total_indicators += 1

                # Volume Weighted Average Price (VWAP)
                if indicators["VWAP"]:
                    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                    fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name="VWAP"))
                    if data['Close'].iloc[-1] > data['VWAP'].iloc[-1]:
                        buy_signals += 1
                    total_indicators += 1

                # Show recommendation summary
                if show_recommendation:
                    st.write("### Recommendation Summary")
                    st.write(f"Total Indicators: {total_indicators}")
                    st.write(f"Buy Signals: {buy_signals}")
                    st.write(f"Sell Signals: {total_indicators - buy_signals}")
                    if buy_signals > total_indicators / 2:
                        st.success("**Recommendation: Buy**")
                    else:
                        st.warning("**Recommendation: Sell**")

                # Update layout to display multiple y-axes for different indicators
                fig.update_layout(
                    title=f"{ticker_symbol} Price and Indicators",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    yaxis2=dict(
                        title="RSI",
                        overlaying="y",
                        side="right"
                    ),
                    yaxis3=dict(
                        title="MACD",
                        overlaying="y",
                        side="right",
                        position=0.85
                    ),
                    legend=dict(x=0, y=1.1, orientation="h")
                )

                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Failed to retrieve data for {ticker_symbol}. Error: {e}")
# Risk Assessment Tab
with tabs[3]:
    st.title("Optimal Risk Portfolio for Selected Stocks")

    # Sidebar Settings
    st.header("Portfolio Settings")
    stocks = st.multiselect(
        "Select Stocks for Portfolio", 
        ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META'], 
        default=['AAPL', 'MSFT']
    )

    # Date Range Slider
    today = date.today()
    date_range = st.slider(
        "Select Date Range:",
        min_value=today - timedelta(days=365 * 5),
        max_value=today,
        value=(today - timedelta(days=365), today),
        format="YYYY-MM-DD",
    )
    start_date, end_date = map(pd.Timestamp, date_range)

    # Risk-Free Rate
    risk_free_rate = st.number_input("Risk-Free Rate (%)", value=2.0, step=0.1) / 100

    # Fetch and Process Data
    if stocks:
        st.write(f"### Selected Stocks: {', '.join(stocks)}")
        try:
            # Fetch stock data
            data = yf.download(stocks, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))['Adj Close']
            
            if data.empty:
                st.error("No data found for the selected stocks and date range.")
            else:
                # Calculate Returns and Excess Returns
                returns = data.pct_change().dropna()
                excess_returns = returns - risk_free_rate / 252

                # Plot Holding Period Returns
                st.write("### Holding Period Returns")
                st.line_chart(returns)

                # Plot Excess Returns
                st.write("### Excess Returns")
                st.line_chart(excess_returns)

                # Standard Deviation and Risk Premium
                std_deviation = excess_returns.std()
                mean_excess_returns = excess_returns.mean()
                risk_premium = mean_excess_returns - risk_free_rate / 252

                st.write("**Standard Deviation of Excess Returns**")
                st.write(std_deviation)

                st.write("**Risk Premium**")
                st.write(risk_premium)

                # Optimal Portfolio Calculation
                cov_matrix = returns.cov()
                weights = np.linalg.inv(cov_matrix) @ mean_excess_returns
                weights /= weights.sum()  # Normalize weights

                # Plot Optimal Portfolio Weights
                st.write("### Optimal Portfolio Weights")
                st.bar_chart(pd.DataFrame(weights, index=returns.columns, columns=["Weight"]))

                # Portfolio Statistics
                portfolio_return = np.dot(weights, mean_excess_returns) * 252
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std

                st.write("### Portfolio Statistics")
                stats_df = pd.DataFrame({
                    "Metric": ["Expected Portfolio Return", "Portfolio Volatility (Risk)", "Sharpe Ratio"],
                    "Value": [portfolio_return, portfolio_std, sharpe_ratio]
                })

                st.write(stats_df)

                # Plot Portfolio Statistics as a bar chart
                st.write("### Portfolio Statistics Chart")
                st.bar_chart(stats_df.set_index('Metric')['Value'])

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please select at least one stock.")
# Comparison
with tabs[4]:
    st.header("Comparison")
    st.write("Compare stocks based on fundamental and technical analysis.")

    # Function to fetch fundamental data
    def get_fundamental_data(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                "Ticker": ticker,
                "Market Cap (Billion)": info.get('marketCap', 0) / 1e9,
                "Trailing P/E Ratio": info.get('trailingPE', 'N/A'),
                "Forward P/E Ratio": info.get('forwardPE', 'N/A'),
                "Price-to-Book Ratio": info.get('priceToBook', 'N/A'),
                "Dividend Yield (%)": info.get('dividendYield', 0) * 100,
                "Earnings Growth (%)": info.get('earningsGrowth', 'N/A'),
                "Revenue Growth (%)": info.get('revenueGrowth', 'N/A'),
                "Debt-to-Equity Ratio": info.get('debtToEquity', 'N/A'),
                "Free Cash Flow (Billion)": info.get('freeCashflow', 0) / 1e9,
            }
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
            return None

    # Stock Selection
    available_tickers = [
        'AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'META', 'NFLX', 'NVDA', 'BRK.B',
        'KO', 'UNH', 'XOM', 'LLY', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'AVGO', 'HD',
        'CVX', 'MRK', 'ABBV', 'COST', 'PEP', 'ADBE'
    ]
    selected_tickers = st.multiselect(
        "Select Stocks for Analysis (both Fundamental and Technical):",
        available_tickers,
        default=['AAPL', 'MSFT', 'TSLA']
    )

    if selected_tickers:
        # Fundamental Analysis
        st.subheader("Fundamental Analysis")
        fundamental_data = [get_fundamental_data(ticker) for ticker in selected_tickers if get_fundamental_data(ticker)]
        if fundamental_data:
            st.dataframe(pd.DataFrame(fundamental_data), use_container_width=True)
        else:
            st.warning("No valid fundamental data available.")

        # Technical Analysis
        st.subheader("Technical Analysis")
        today, min_date = date.today(), date.today() - timedelta(days=365*5)
        date_range = st.slider("Select Date Range", min_date, today, (today - timedelta(days=365), today))
        sdate, edate = date_range

        # Fetch historical data
        data = yf.download(selected_tickers, start=sdate, end=edate, interval="1d", auto_adjust=True)
        if not data.empty:
            for ticker in selected_tickers:
                st.write(f"### {ticker}")

                # Fetch stock's closing price
                stock_data = data['Close'][ticker]
                df = pd.DataFrame(stock_data).rename(columns={ticker: 'Close'})

                # Add moving averages
                df['SMA 50'] = stock_data.rolling(window=50).mean()
                df['SMA 100'] = stock_data.rolling(window=100).mean()

                # Plot stock price with moving averages
                st.line_chart(df)

        else:
            st.error("No historical data available for the selected period.")
    else:
        st.warning("Please select at least one stock.")
# News
with tabs[5]:
    st.header("News")
    st.write("Stay updated with the latest news on your selected stock.")
    # Function to extract news from Google News RSS
    def extract_news_from_google_rss(ticker):
        """Fetch news articles for a given stock ticker using Google News RSS."""
        url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        news_articles = []
        for entry in feed.entries[:15]:  # Limit to the latest 15 articles
            published_date = datetime(*entry.published_parsed[:6])  # Convert to datetime
            news_articles.append({"title": entry.title, "url": entry.link, "date": published_date})
        return news_articles

    # Function to fetch and preprocess text
    def fetch_article_content(url):
        """Fetch article content using BeautifulSoup."""
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            headline = soup.title.string if soup.title else "No headline"
            paragraphs = soup.find_all("p")
            content = " ".join([para.get_text() for para in paragraphs])
            return headline, content
        except Exception as e:
            return None, None

    # App layout and styling
    st.title("Stock News Fetcher")
    ticker_symbol_news = st.text_input("Enter stock ticker (e.g., AAPL, MSFT):", key="ticker_news")  # Unique key

    if ticker_symbol_news:
        try:
            # Fetch news for the given ticker automatically
            news = extract_news_from_google_rss(ticker_symbol_news)
            if news:
                st.subheader(f"Latest News for {ticker_symbol_news.upper()}")
                for article in news:
                    st.write(f"**{article['title']}**")
                    st.write(f"[Read more]({article['url']}) - {article['date'].strftime('%Y-%m-%d %H:%M:%S')}")
                    st.write("---")
            else:
                st.warning("No news articles found for this ticker.")
        except Exception as e:
            st.error(f"An error occurred while fetching news: {e}")
    else:
        st.info("Enter a stock ticker above to fetch the latest news.")    
# Technical Analysis
with tabs[6]:
    st.title("Contact Us")
    # University Information
    st.subheader("International University of Japan")
    st.write("**Address:** 777 Kokusai-cho, Minami Uonuma-shi, Niigata 949-7277, Japan")
    st.write("**Phone:** +81 (0) 25-779-1111")
    st.write("**FAX:** +81 (0) 25-779-4441")  
    st.markdown("---")
    # Feedback Section
    st.write("### Rate Your Experience")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("😊 Excellent"):
            st.success("Thank you for your positive feedback!")
    with col2:
        if st.button("😐 Neutral"):
            st.info("Thank you for your feedback!")
    with col3:
        if st.button("☹️ Poor"):
            st.warning("We appreciate your feedback and will work to improve.")
            st.markdown("---")
    # Developers' Information
    st.write("### Meet Our Developers")
    developers = [
            {"name": "Arthur Kariuki", "email": "a.nj58@iuj.ac.jp", "github": "https://github.com/arthurkrk"},
            {"name": "Fahad Mirza", "email": "fmmirza@iuj.ac.jp", "github": "https://github.com/fmmirza7"},
            {"name": "Merwan Limam", "email": "l.merwan@iuj.ac.jp", "github": "https://github.com/Lmerwan"},
            {"name": "Adama Cisse", "email": "acisse@iuj.ac.jp", "github": "https://github.com/adama6cpython"},
            {"name": "Ibra Ndiaye", "email": "ibrahim7@iuj.ac.jp", "github": "https://github.com/rabihimo"},
            {"name": "Trymore Musasiri", "email": "tmusariri@iuj.ac.jp", "github": "https://github.com"},
            ]
    for dev in developers:
        st.write(f"**{dev['name']}**")
        st.write(f"Email: [{dev['email']}](mailto:{dev['email']})")
        st.write(f"GitHub: [{dev['github']}]({dev['github']})")
        st.markdown("---")
    #Thank you note
    st.write("Thank you for visiting us today 😊.")
# Render the footer on all pages

render_footer()
