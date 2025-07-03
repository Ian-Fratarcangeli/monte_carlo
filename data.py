from openbb import obb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def set_fred_api_key(api_key: str):
    obb.user.credentials.fred_api_key = api_key


def get_historical_prices(symbol: str, provider: str = "yfinance") -> pd.DataFrame:
    """
    Get historical price data for a given symbol.
    
    Args:
        symbol (str): The stock symbol to get data for (e.g., "AAPL")
        provider (str): The data provider to use (default: "yfinance")
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing historical price data
    """
    try:
        historical_data = obb.equity.price.historical(
            symbol=symbol,
            provider=provider
        )
        return historical_data.to_df()
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

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
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Remove the first row which will be NaN due to the shift
    log_returns = log_returns.dropna()
    
    # Calculate mean and standard deviation
    mean_return = log_returns.mean()
    std_return = log_returns.std()

    
    return log_returns, mean_return, std_return

def get_risk_free_rate() -> float:
    """
    Get the current interest rate for a 3-month T-bill
    """
    rates = obb.fixedincome.government.treasury_rates(provider='federal_reserve')
    return rates.results[-1].month_3



