# Libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import os
from datetime import datetime, timedelta

# Configure the API key IMPORTANT: Use Streamlit secrets or environment variables for security
# Get your API key from Google AI Studio - https://makersuite.google.com/app/apikey
#GOOGLE_API_KEY = st.secrets["AIzaSyA9Nu01_vdfX2yezq-yHBx5ENhm_kjfPFc"] # Use Streamlit secrets for security, or environment variables
GOOGLE_API_KEY = "AIzaSyA9Nu01_vdfX2yezq-yHBx5ENhm_kjfPFc" # Use Streamlit secrets for security, or environment variables
genai.configure(api_key=GOOGLE_API_KEY  )

# Select the Gemini model
MODEL_NAME = 'gemini-2.0-flash' # or 'gemini-pro-vision' if you need to analyze images
gen_model = genai.GenerativeModel(MODEL_NAME)

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# Sidebar inputs
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker", "TCS.NS").upper()
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())
analysis_type = st.sidebar.selectbox("Analysis Type", ["Technical Analysis Summary", "Sentiment Analysis", "Trend Prediction"])
indicator_list = ['SMA', 'EMA', 'RSI', 'MACD', 'Bollinger Bands']
indicators_to_display = st.sidebar.multiselect("Technical Indicators", indicator_list, default=['SMA', 'EMA'])


@st.cache_data
def load_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

def calculate_technical_indicators(df, indicators):
    if 'SMA' in indicators:
        df['SMA'] = df['Close'].rolling(window=20).mean() # Simple Moving Average
    if 'EMA' in indicators:
        df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean() # Exponential Moving Average
    if 'RSI' in indicators:
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.rolling(window=14).mean() # Changed from ewm for simplicity
        ema_down = down.rolling(window=14).mean() # Changed from ewm for simplicity
        rs = ema_up / ema_down
        df['RSI'] = 100 - (100 / (1 + rs)) # Relative Strength Index
    if 'MACD' in indicators:
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26 # Moving Average Convergence Divergence
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean() # Signal Line
    if 'Bollinger Bands' in indicators:
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = rolling_mean + (rolling_std * 2) # Upper Bollinger Band
        df['Lower_Band'] = rolling_mean - (rolling_std * 2) # Lower Bollinger Band
    return df

def generate_ai_analysis(stock_data, analysis_type, indicators):
    indicator_str = ", ".join(indicators) if indicators else "stock price action"
    prompt_text = f"Provide a concise technical analysis summary for {ticker_symbol} stock based on the last {len(stock_data)} days of data, considering {indicator_str}. Focus on {analysis_type}."

    response = gen_model.generate_content(prompt_text)
    return response.text

# Main app flow
if ticker_symbol:
    try:
        stock_df = load_stock_data(ticker_symbol, start_date, end_date)

        if not stock_df.empty:
            stock_df = calculate_technical_indicators(stock_df, indicators_to_display)

            st.subheader(f"{ticker_symbol} Stock Data and Analysis")

            # Candlestick chart
            fig = go.Figure(data=[go.Candlestick(x=stock_df.index,
                open=stock_df['Open'],
                high=stock_df['High'],
                low=stock_df['Low'],
                close=stock_df['Close'],
                name='Candlestick')])

            # Add technical indicators to the chart
            for indicator in indicators_to_display:
                if indicator == 'SMA':
                    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['SMA'], name='SMA'))
                elif indicator == 'EMA':
                    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['EMA'], name='EMA'))
                elif indicator == 'RSI':
                    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['RSI'], name='RSI', yaxis="y2"))
                elif indicator == 'MACD':
                    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['MACD'], name='MACD'))
                    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Signal_Line'], name='Signal Line'))
                elif indicator == 'Bollinger Bands':
                    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Upper_Band'], name='Upper Bollinger Band'))
                    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Lower_Band'], name='Lower Bollinger Band'))

            # Layout adjustments
            fig.update_layout(
                yaxis_title='Stock Price',
                yaxis2=dict(title='RSI', overlaying='y', side='right'), # RSI on secondary y-axis
                title=f'{ticker_symbol} Candlestick Chart with Indicators',
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # AI Analysis
            if analysis_type != "None":
                ai_analysis = generate_ai_analysis(stock_df, analysis_type, indicators_to_display)
                st.subheader(f"AI-Powered {analysis_type}")
                st.write(ai_analysis)
        else:
            st.error("Could not retrieve stock data. Please check the ticker symbol and date range.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.warning("Please ensure you have a valid Gemini API key in your Streamlit secrets or environment variables.")
else:
    st.info("Please enter a stock ticker symbol to begin.")