import streamlit as st
from elements import (
    show_data_upload,
    show_point_statistics,
    show_interval_statistics,
    show_correlation,
    show_regression,
    show_weather_impact,
    show_hourly_monthly,
    show_categorical_regression,
    show_time_series,
    show_summary_statistics,
)

st.set_page_config(page_title="Solar Power Analysis", layout="wide")
st.title("☀️ Արևային էներգիայի արտադրություն - Վիճակագրական վերլուծություն")

df = show_data_upload()

if df is not None:
    numeric_cols = ['WindSpeed', 'Sunshine', 'AirPressure', 'Radiation',
                    'AirTemperature', 'RelativeAirHumidity', 'SystemProduction']

    col = show_point_statistics(df, numeric_cols)
    show_interval_statistics(df, numeric_cols, col)
    show_correlation(df, numeric_cols)
    show_regression(df, numeric_cols)
    show_weather_impact(df)
    show_hourly_monthly(df)
    show_categorical_regression(df)
    show_time_series(df)
    show_summary_statistics(df, numeric_cols)
