import pandas as pd
import numpy as np


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df['DateTime'] = pd.to_datetime(
        df['Date-Hour(NMT)'], format='%d.%m.%Y-%H:%M')
    df['Date'] = df['DateTime'].dt.date
    df['Hour'] = df['DateTime'].dt.hour
    df['Month'] = df['DateTime'].dt.month
    df['Month_Name'] = df['DateTime'].dt.strftime('%B')
    df['Day'] = df['DateTime'].dt.day
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek
    df['DayOfWeek_Name'] = df['DateTime'].dt.strftime('%A')
    return df


def build_daily_ts(df: pd.DataFrame) -> pd.Series:
    daily = df.groupby('Date')['SystemProduction'].mean().reset_index()
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.sort_values('Date')
    ts = daily.set_index('Date')['SystemProduction']
    ts = ts.asfreq('D').ffill().bfill()
    return ts


UNITS = {
    'WindSpeed': 'm/s',
    'Sunshine': 'W/m²',
    'AirPressure': 'hPa',
    'Radiation': 'W/m²',
    'AirTemperature': '°C',
    'RelativeAirHumidity': '%',
    'SystemProduction': 'kW',
}
