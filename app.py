#import packages
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import appdirs as ad
import pandas as pd
import math
from pathlib import Path

# Specify title and logo for the webpage.
# Set up your web app
import streamlit as st
import yfinance as yf
import datetime
from datetime import date, timedelta

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
render_header("S&P 500 Stock ANalysis")
# Create tabs
tabs = st.tabs(["Home", "Data", "Visualization", "Contacts"])
# Tab: Home
with tabs[0]:
    st.header("Home")
    st.write("Welcome to the Home page!")
    st.image("https://via.placeholder.com/600x300", caption="Placeholder image for the Home page.")

# Tab: Data
with tabs[1]:
    st.header("Data")
    st.write("This is the Data page. You can upload and display data here.")

# Tab: Visualization
with tabs[2]:
    st.header("Visualization")
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
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    selected_stocks = st.multiselect(
        "Select Stocks", symbols, default=["AAPL"]
    )

    # Stock comparison
    st.title("Stock Comparison")
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


# Tab: Contact Us
with tabs[3]:
    st.header("Contact Us")
    st.write("We'd love to hear your feedback! Please use the form below.")

    # Feedback form
    with st.form("feedback_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            if name and email and message:
                st.success("Thank you for your feedback!")
                st.write(f"**Name:** {name}")
                st.write(f"**Email:** {email}")
                st.write(f"**Message:** {message}")
            else:
                st.error("Please fill out all fields.")

    # Contact Information
    st.write("---")
    st.write("**International University of Japan**")
    st.write("777 Kokusai-cho, Minami Uonuma-shi, Niigata 949-7277 Japan")
    st.write("**Phone:** 81+(0) 25-779-1111")
    st.write("**FAX:** 81+(0) 25-779-4441")

# Render the footer on all pages
render_footer()
