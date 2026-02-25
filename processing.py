import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, detrend as scipy_detrend
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ──────────────────────────────────────────────
# Point statistics
# ──────────────────────────────────────────────

def compute_point_stats(series: pd.Series, unit: str) -> pd.DataFrame:
    s = series.dropna()
    stats = {
        f"Միջին ({unit})": s.mean(),
        f"Մեդիան ({unit})": s.median(),
        f"Մոդա ({unit})": s.mode().iloc[0] if not s.mode().empty else np.nan,
        f"Մինիմում ({unit})": s.min(),
        f"Մաքսիմում ({unit})": s.max(),
        f"Տարածություն ({unit})": s.max() - s.min(),
        f"Վարիացիա ({unit}²)": s.var(),
        f"Ստանդարտ շեղում ({unit})": s.std(),
        "Վարիացիայի գործակից CV (%)": (s.std() / s.mean() * 100) if s.mean() != 0 else np.nan,
        "Ասիմետրիայի գործակից (Skewness)": skew(s),
        "Էքսցես (Kurtosis)": kurtosis(s),
        f"Q1 ({unit})": s.quantile(0.25),
        f"Q2 ({unit})": s.quantile(0.50),
        f"Q3 ({unit})": s.quantile(0.75),
        f"IQR ({unit})": s.quantile(0.75) - s.quantile(0.25),
    }
    df = pd.DataFrame(stats.items(), columns=["Բնութագիր", "Արժեք"])
    df["Արժեք"] = df["Արժեք"].apply(
        lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
    return df


# ──────────────────────────────────────────────
# Regression
# ──────────────────────────────────────────────

def fit_regression(df: pd.DataFrame, x_cols: list, y_col: str):
    df_c = df[x_cols + [y_col]].dropna()
    X, y = df_c[x_cols], df_c[y_col]
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    metrics = {
        "r2": r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "mae": mean_absolute_error(y, y_pred),
        "intercept": model.intercept_,
        "coef": model.coef_,
    }
    return model, X, y, y_pred, metrics


# ──────────────────────────────────────────────
# ACF seasonal period detection
# ──────────────────────────────────────────────

def detect_seasonal_period(ts: pd.Series, min_period: int = 3) -> dict:
    n = len(ts)
    max_lag = min(n // 2 - 1, 365)
    detrended = scipy_detrend(ts.values)
    acf_vals = acf(detrended, nlags=max_lag, fft=True)
    confidence = 1.96 / np.sqrt(n)

    search = acf_vals[min_period:]
    peaks, _ = find_peaks(search, height=confidence, distance=3)
    peaks_adj = peaks + min_period

    result = {
        "acf_values": acf_vals,
        "confidence": confidence,
        "max_lag": max_lag,
        "peaks": peaks_adj,
        "top_peaks": None,
        "top_vals": None,
        "best_period": min(7, n // 2),
        "period_type": "կankhаdrvatc (7 or)",
    }

    if len(peaks_adj) > 0:
        peak_vals = acf_vals[peaks_adj]
        idx = np.argsort(peak_vals)[::-1]
        top = peaks_adj[idx[:5]]
        top_v = peak_vals[idx[:5]]
        result.update({
            "top_peaks": top,
            "top_vals": top_v,
            "best_period": int(top[0]),
            "period_type": f"ACF-ov voroshvats ({int(top[0])} or)",
        })
    return result


# ──────────────────────────────────────────────
# Decomposition
# ──────────────────────────────────────────────

def decompose_ts(ts: pd.Series, period: int, model_type: str):
    shift = 0
    ts_t = ts.copy()
    if model_type == 'multiplicative' and (ts <= 0).any():
        shift = abs(ts.min()) + 1
        ts_t = ts + shift
    result = seasonal_decompose(
        ts_t, model=model_type, period=period, extrapolate_trend='freq')
    return result, shift


# ──────────────────────────────────────────────
# Sinusoidal model
# ──────────────────────────────────────────────

def sinusoidal_model(day, amplitude, phase, offset):
    return amplitude * np.sin(2 * np.pi * day / 365 + phase) + offset


def build_yearly_trend(ts: pd.Series) -> pd.Series:
    """365-day (or shorter) centred rolling mean for sinusoidal detrending."""
    n = len(ts)
    win = min(365, n // 2 * 2 - 1)
    win = max(win, 3)
    if win % 2 == 0:
        win -= 1
    return ts.rolling(window=win, center=True, min_periods=win // 2).mean()


def fit_sinusoidal(ts: pd.Series, yearly_trend: pd.Series):
    """
    Fit sinusoidal model to (ts - yearly_trend).
    Returns dict with params or sino_success=False on failure.
    """
    day_of_year = np.array(
        [(d - pd.Timestamp(f'{d.year}-01-01')).days for d in ts.index]
    )
    detrended = ts.values - yearly_trend.values
    valid = ~np.isnan(detrended)

    if valid.sum() < 10:
        return {"sino_success": False}

    try:
        p0 = [detrended[valid].std(), 0, detrended[valid].mean()]
        params, _ = curve_fit(sinusoidal_model, day_of_year[valid], detrended[valid],
                              p0=p0, maxfev=10000)
        amplitude, phase, offset = params

        fitted = sinusoidal_model(day_of_year[valid], *params)
        ss_res = np.sum((detrended[valid] - fitted) ** 2)
        ss_tot = np.sum((detrended[valid] - detrended[valid].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

        peak_day = int((365 / (2 * np.pi)) * (np.pi / 2 - phase)) % 365
        trough_day = int((365 / (2 * np.pi)) * (3 * np.pi / 2 - phase)) % 365

        return {
            "sino_success": True,
            "amplitude": amplitude,
            "phase": phase,
            "offset": offset,
            "r2": r2,
            "day_of_year": day_of_year,
            "detrended": detrended,
            "valid": valid,
            "peak_day": peak_day,
            "trough_day": trough_day,
        }
    except Exception:
        return {"sino_success": False}


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

def predict_acf(
    trend: pd.Series,
    seasonal: pd.Series,
    seasonal_period: int,
    days_ahead: int,
    trend_method: str,
    model_type: str,
    shift: float,
) -> dict:
    trend_clean = trend.dropna()

    if "Gitsayin" in trend_method or "Гծ" in trend_method or "Գծ" in trend_method:
        n = min(30, len(trend_clean))
        slope, _ = np.polyfit(np.arange(n), trend_clean.iloc[-n:].values, 1)
        t_pred = float(trend_clean.iloc[-1]) + slope * days_ahead
        method_name = f"Gitsayin (slope={slope:.2f} kW/or)"
    else:
        slope = 0.0
        t_pred = float(trend_clean.iloc[-1])
        method_name = "Verdjin arjeq"

    s_idx = days_ahead % seasonal_period
    s_pred = float(seasonal.iloc[s_idx])

    if model_type == 'additive':
        pred = t_pred + s_pred
    else:
        pred = t_pred * s_pred - shift

    return {
        "trend_pred": t_pred,
        "seasonal_pred": s_pred,
        "pred": max(0.0, pred),
        "slope": slope,
        "method_name": method_name,
        "seasonal_idx": s_idx,
    }


def predict_sino(
    yearly_trend: pd.Series,
    sino: dict,
    days_ahead: int,
    pred_day_of_year: int,
    trend_method: str,
) -> dict:
    yt_clean = yearly_trend.dropna()

    if "Գծային" in trend_method:
        n = min(60, len(yt_clean))
        slope2, _ = np.polyfit(np.arange(n), yt_clean.iloc[-n:].values, 1)
        t_pred = float(yt_clean.iloc[-1]) + slope2 * days_ahead
    else:
        t_pred = float(yt_clean.iloc[-1])

    s_pred = sinusoidal_model(
        pred_day_of_year, sino["amplitude"], sino["phase"], sino["offset"]
    )
    pred = t_pred + s_pred

    return {
        "trend_pred": t_pred,
        "seasonal_pred": s_pred,
        "pred": max(0.0, pred),
    }
