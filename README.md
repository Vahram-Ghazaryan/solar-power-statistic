# ☀️ Solar Power Analysis — Statistical Dashboard

A Streamlit-powered interactive web application for statistical analysis and forecasting of solar power system production data.

---

## Overview

This application enables a comprehensive study of solar energy production through descriptive statistics, regression modeling, time series analysis, sinusoidal seasonal modeling, and production forecasting, all presented through an interactive interface with Armenian language labels.

---

## Project Structure

```
solar_app/
├── app.py            # Entry point — wires all sections together
├── elements.py       # Streamlit UI sections (show_* functions)
├── processing.py     # Statistical functions
├── graphics.py       # Matplotlib/Seaborn plotting functions
├── utils.py          # Data preprocessing and daily time series builder
├── constants.py      # All text constants, stat explanations, month names
└── requirements.txt  # Python dependencies
```

---

## Features

| Section                    | Description                                                                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Upload**            | Upload a CSV file; automatic parsing and preview                                                                                            |
| **Point Statistics**       | Mean, median, mode, std dev, variance, skewness, kurtosis, quartiles, IQR — with inline explanations                                        |
| **Interval Statistics**    | Adjustable-bin frequency table with descriptive stats per interval                                                                          |
| **Correlation Analysis**   | Matrix correlations across all numeric variables                                                                                            |
| **Regression Analysis**    | Linear regression with 1–N predictors; R², RMSE, MAE metrics; 1D/2D/residual plots                                                          |
| **Weather Impact**         | Scatter plots of Radiation, Sunshine, Temperature, and WindSpeed vs. SystemProduction                                                       |
| **Categorical Regression** | One-hot encoded monthly regression to quantify month-level effects on production                                                            |
| **Time Series Analysis**   | Daily/hourly/monthly aggregations, ACF-based seasonal period detection, additive/multiplicative decomposition, sinusoidal seasonal modeling |
| **Forecasting**            | Date-picker tool combining ACF seasonal decomposition and sinusoidal models with linear or last-value trend extrapolation                   |
| **Summary Statistics**     | Full table across all numeric columns                                                                                                       |

---

## Input Data Format

The application expects a CSV file with the following columns:

| Column                | Description                                  | Unit |
| --------------------- | -------------------------------------------- | ---- |
| `Date-Hour(NMT)`      | Datetime string in `DD.MM.YYYY-HH:MM` format | —    |
| `WindSpeed`           | Wind speed                                   | m/s  |
| `Sunshine`            | Solar irradiance                             | W/m² |
| `AirPressure`         | Atmospheric pressure                         | hPa  |
| `Radiation`           | Radiation                                    | W/m² |
| `AirTemperature`      | Air temperature                              | °C   |
| `RelativeAirHumidity` | Relative humidity                            | %    |
| `SystemProduction`    | Solar system output power                    | kW   |

---

## Installation & Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.9+.

---

## Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
scipy>=1.11.0
statsmodels>=0.14.0
```

---

## Module Responsibilities

### `app.py`

The entry point. Sets the page config, renders the title, calls `show_data_upload()`, and chains all analysis sections in order.

### `elements.py`

Contains all `show_*` functions that render each section in the Streamlit app. Handles user inputs (selectboxes, sliders, date pickers) and calls processing/graphics functions to display results.

### `processing.py`

Pure computation — no UI code. Key functions:

- `compute_point_stats()` — descriptive statistics for a pandas Series
- `fit_regression()` — sklearn LinearRegression wrapper with metrics
- `detect_seasonal_period()` — ACF-based seasonal period detection using `find_peaks`
- `decompose_ts()` — wraps `statsmodels.seasonal_decompose`
- `fit_sinusoidal()` — fits `A × sin(2π × day/365 + φ) + C` via `scipy.optimize.curve_fit`
- `predict_acf()` / `predict_sino()` — future production forecasting using decomposition and sinusoidal models respectively

### `graphics.py`

All Matplotlib/Seaborn figure creation. Returns `fig` objects for use with `st.pyplot()`. Includes histogram, boxplot, correlation heatmap, regression plots (1D/2D/residual), ACF bar charts, decomposition subplots, sinusoidal curve plots, and forecast timeline.

### `utils.py`

- `preprocess_dataframe()` — parses `Date-Hour(NMT)`, extracts hour, month, day, weekday columns
- `build_daily_ts()` — aggregates hourly data to a daily `pd.Series` with forward/backward fill
- `UNITS` — dict mapping column names to their measurement units

### `constants.py`

All static text content: Armenian-language explanations for each statistic and regression metric (`STAT_EXPLANATIONS`, `METRIC_EXPLANATIONS`), sinusoidal model documentation (`SINUSOIDAL_EXPLANATION`), decomposition model descriptions (`MODEL_INFO`), and month name/day arrays.

---

## Forecasting Methods

The app supports two complementary forecasting approaches:

**ACF Seasonal Decomposition Model**
Uses `statsmodels.seasonal_decompose` with an automatically detected period (via ACF peak detection). Supports both additive (`Y = T + S + E`) and multiplicative (`Y = T × S × E`) models. Trend is extrapolated using either the last known value or linear regression over the trailing 30 days.

**Sinusoidal Model**
Fits a continuous sinusoidal curve to the detrended seasonal component:

```
S(day) = A × sin(2π × day/365 + φ) + C
```

where `A` is the seasonal amplitude, `φ` is the phase shift, and `C` is the offset. This model is physically motivated by Earth's orbital mechanics and can extrapolate to any future date in the year.

Forecast reliability guidance:

- **1–30 days ahead** → high confidence
- **31–90 days ahead** → moderate confidence
- **90+ days ahead** → lower confidence
