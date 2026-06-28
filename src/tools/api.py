import os
import time
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
import json


load_dotenv(".env", override=True)

HYPERLIQUID_API_URL = os.environ.get("HYPERLIQUID_API_URL")
BINANCE_API_URL = os.environ.get("BINANCE_API_URL")
API_COPIN_OI = os.environ.get("API_COPIN_OI")


def date_to_timestamp(date):
    """Convert a date/datetime string or datetime object to millisecond timestamp."""
    if isinstance(date, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                date = datetime.strptime(date, fmt)
                break
            except ValueError:
                continue
    timestamp_seconds = datetime.timestamp(date)
    timestamp_milliseconds = int(timestamp_seconds * 1000)
    return timestamp_milliseconds


def get_price_API_HYPERLIQUID(pair, open_time, close_time, max_retries: int = 3, candle_interval: str = "1h"):
    """
    Fetch historical price data from HyperLiquid API.

    Args:
        pair (str): Trading pair symbol
        open_time (str or datetime): Start time for data fetch
        close_time (str or datetime): End time for data fetch
        max_retries (int): Number of retry attempts on connection failure

    Returns:
        pandas.DataFrame: DataFrame containing OHLCV data, or None on failure
    """
    open_time_ms = date_to_timestamp(open_time)
    close_time_ms = date_to_timestamp(close_time)
    APIURL = HYPERLIQUID_API_URL

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": pair,
            "interval": candle_interval,
            "startTime": open_time_ms,
            "endTime": close_time_ms,
        },
    }
    headers = {"Content-Type": "application/json"}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(APIURL, json=payload, headers=headers, timeout=15)
            if response.status_code != 200:
                print(f"[HyperLiquid] HTTP {response.status_code}: {response.text[:200]}")
                return None
            candles = response.json()
            if not candles:
                print(f"[HyperLiquid] Empty response for {pair} {open_time_ms}–{close_time_ms}")
                return None
            df = pd.DataFrame(candles)
            df.rename(
                columns={
                    "t": "timestamp",
                    "T": "close_time",
                    "s": "symbol",
                    "i": "interval",
                    "o": "open",
                    "c": "close",
                    "h": "high",
                    "l": "low",
                    "v": "volume",
                    "n": "number_of_trades",
                },
                inplace=True,
            )
            df = df[["open", "close", "high", "low", "volume"]]
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            print(f"[HyperLiquid] Attempt {attempt}/{max_retries} failed for {pair}: {e}")
            if attempt < max_retries:
                time.sleep(5)

    print(f"[HyperLiquid] All {max_retries} attempts failed for {pair}, giving up.")
    return None


def get_price_API_BINANCE(pair, open_time, close_time, limit: int = 1000):
    open_time = date_to_timestamp(open_time)
    close_time = date_to_timestamp(close_time)
    APIURL = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": pair + "USDT",
        "interval": "60",
        "start": open_time,
        "end": close_time,
        "limit": limit,
    }
    try:
        response = requests.get(APIURL, params=params, timeout=15)
        data = response.json()
        rows = data["result"]["list"]
        df = pd.DataFrame(rows, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df = df[["open", "close", "high", "low", "volume"]]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[Bybit] Exception for {pair}: {e}")
        return None


def get_OI_position_Copin(pair: str, isLong: bool):
    """
    Fetch open interest data for a specific position type from Copin API.

    Args:
        pair (str): Trading pair symbol (without -USDT suffix)
        isLong (bool): True for long positions, False for short positions

    Returns:
        float: Total size of open interest for the specified position type
        str: Error message if request fails
    """
    APIURL = API_COPIN_OI
    pair = pair + "-USDT"
    if isLong:
        value_long = "true"
    else:
        value_long = "false"
    query = {
        "pagination": {"limit": 500, "offset": 0},
        "queries": [
            {"fieldName": "pair", "value": pair},
            {"fieldName": "isLong", "value": value_long},
        ],
        "sortBy": "size",
        "sortType": "desc",
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(APIURL, headers=headers, data=json.dumps(query), timeout=15)
        data = response.json()
        df = data["data"]
        total_size = sum(d["size"] for d in df)
        return total_size
    except Exception as e:
        print(e)
        return "Cannot find OI of this crypto"


def get_LS_OI_Copin(pair):
    """
    Fetch both long and short open interest data from Copin API.

    Args:
        pair (str): Trading pair symbol (without -USDT suffix)

    Returns:
        tuple: (long_oi, short_oi) containing the total open interest for long and short positions
        str: Error message if request fails
    """
    longOI = get_OI_position_Copin(pair, True)
    shortOI = get_OI_position_Copin(pair, False)
    if isinstance(longOI, str) | isinstance(shortOI, str):
        return "Cannot find OI of this crypto"

    return longOI, shortOI
