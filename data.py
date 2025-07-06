from fredapi import Fred
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re



fred = None

def set_fred_api_key(api_key: str):
    global fred
    fred = Fred(api_key=api_key)


def parse_time_arg(time_str: str) -> datetime:
    """
    Parses a time string returns the date that amount of time ago from today.

    Args:
        time_str (str): Time string, e.g. '1yr', '6mo'

    Returns:
        datetime: The date corresponding to that time ago from today.
    """
    now = datetime.today()
    pattern = r"(\d+)(yr|mo)"
    match = re.fullmatch(pattern, time_str.lower())

    if not match:
        raise ValueError(f"Invalid time string format: {time_str}")

    quantity, unit = match.groups()
    quantity = int(quantity)

    if unit == "yr":
        return now - relativedelta(years=quantity)
    elif unit == "mo":
        return now - relativedelta(months=quantity)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")
    
def get_historical_prices(symbol: str, time_arg: str = "1yr") -> pd.DataFrame:
    """
    Get historical price data for a given symbol using yfinance.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., "AAPL")
        time_arg (str): time from present (e.g., "6mo")
        
    Returns:
        pd.DataFrame: DataFrame with historical prices (Open, High, Low, Close, Volume)
    """

    start_date = parse_time_arg(time_arg).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")

    df = yf.download(symbol, start=start_date, end=end_date)

    if df.empty:
        print(f"No data found for {symbol} between {start_date} and {end_date}")
    return df


def calculate_log_returns(df: pd.DataFrame) -> tuple:
    """
    Calculate log returns and their statistics from historical price data.
    
    Args:
        df (pd.DataFrame): DataFrame containing historical price data with a 'close' column
        
    Returns:
        tuple: (log_returns, mean_return, std_return)
            - log_returns: Series of daily log returns
            - mean_return: Mean of the log returns
            - std_return: Standard deviation of the log returns
    """
    # Calculate log returns
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    
    # Remove the first row which will be NaN due to the shift
    log_returns = log_returns.dropna()
    
    # Calculate mean and standard deviation
    mean_return = log_returns.mean().iloc[0]
    std_return = log_returns.std().iloc[0]

    
    return log_returns, mean_return, std_return

def get_risk_free_rate() -> float:
    """
    Get the current interest rate for a 3-month T-bill
    """
    if fred is None:
        raise RuntimeError("FRED API key not set. Call set_fred_api_key(api_key) first.")
    
    # 'DTB3' is the FRED series ID for 3-Month Treasury Bill: Secondary Market Rate
    rates = fred.get_series('DTB3')
    latest_rate = rates.dropna().iloc[-1] / 100.0 
    return latest_rate



