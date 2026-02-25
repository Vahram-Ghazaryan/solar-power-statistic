import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from processing import sinusoidal_model
from constants import MONTH_STARTS, MONTH_SHORT


def plot_histogram(series: pd.Series, col: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(series.dropna(), bins=30, color='skyblue', edgecolor='black')
    ax.set_title(f"{col} - Հիստոգրամ", fontsize=14, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel("Հաճախականություն")
    ax.grid(alpha=0.3)
    return fig


def plot_boxplot(series: pd.Series, col: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(series.dropna(), vert=True)
    ax.set_title(f"{col} - Box Plot", fontsize=14, fontweight='bold')
    ax.set_ylabel(col)
    ax.grid(alpha=0.3)
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax,
                fmt='.2f', linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Կորելացիոն մատրիցա", fontsize=16, fontweight='bold')
    return fig


def plot_regression_1d(X, y, y_pred, x_col, y_col):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(X[x_col], y, color='blue', alpha=0.5, s=20, label='Տվյալներ')
    ax1.plot(X[x_col], y_pred, color='red', linewidth=2, label='Ռեգրեսիոն գիծ')
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)
    ax1.set_title(f"{y_col} vs {x_col}", fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    residuals = y - y_pred
    ax2.scatter(y_pred, residuals, color='green', alpha=0.5, s=20)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel("Կանխատեսված արժեքներ")
    ax2.set_ylabel("Մնացորդներ (Residuals)")
    ax2.set_title("Residual Plot", fontweight='bold')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_regression_2d(X, y, y_pred, x_cols, y_col, model):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[x_cols[0]], X[x_cols[1]], y,
                c='blue', marker='o', alpha=0.5, s=20)
    x1r = np.linspace(X[x_cols[0]].min(), X[x_cols[0]].max(), 20)
    x2r = np.linspace(X[x_cols[1]].min(), X[x_cols[1]].max(), 20)
    x1m, x2m = np.meshgrid(x1r, x2r)
    ym = model.predict(np.c_[x1m.ravel(), x2m.ravel()]).reshape(x1m.shape)
    ax1.plot_surface(x1m, x2m, ym, alpha=0.3, cmap='viridis')
    ax1.set_xlabel(x_cols[0])
    ax1.set_ylabel(x_cols[1])
    ax1.set_zlabel(y_col)
    ax1.set_title(f"{y_col} - Ռեգրեսիայի հարթություն (3D)", fontweight='bold')

    ax2 = fig.add_subplot(122)
    ax2.scatter(y_pred, y - y_pred, color='green', alpha=0.5, s=20)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel("Կանխատեսված արժեքներ")
    ax2.set_ylabel("Մնացորդներ (Residuals)")
    ax2.set_title("Residual Plot", fontweight='bold')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_weather_scatter(df):
    weather_vars = ['Radiation', 'Sunshine', 'AirTemperature', 'WindSpeed']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, var in enumerate(weather_vars):
        axes[idx].scatter(df[var], df['SystemProduction'], alpha=0.3, s=10)
        axes[idx].set_xlabel(var)
        axes[idx].set_ylabel('SystemProduction')
        axes[idx].set_title(f'SystemProduction vs {var}', fontweight='bold')
        axes[idx].grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_monthly_bar(coeffs_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    coeffs_df.plot(x='Ամիս', y='Գործակից', kind='bar',
                   ax=ax, color='coral', legend=False)
    ax.set_title("Ամիսների ազդեցությունը գործակիցների վրա",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Ամիս")
    ax.set_ylabel("Գործակից")
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    return fig


def plot_daily_ts(ts: pd.Series):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ts.index, ts.values, color='orange', linewidth=1.5)
    ax.set_xlabel("Ամիսաթիվ")
    ax.set_ylabel("Միջին արտադրություն (kW)")
    ax.set_title("Օրակական միջին արտադրություն",
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    return fig


def plot_hourly_bar(hourly_avg: pd.Series):
    fig, ax = plt.subplots(figsize=(12, 5))
    hourly_avg.plot(kind='bar', color='steelblue', ax=ax)
    ax.set_xlabel("Jam")
    ax.set_ylabel("Միջին արտադրություն (kW)")
    ax.set_title("24-ժամյա միջին արտադրություն",
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0)
    return fig


def plot_monthly_line(monthly_avg: pd.Series):
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_avg.plot(kind='line', marker='o', color='green',
                     linewidth=2, markersize=8, ax=ax)
    ax.set_xlabel("Ամիս")
    ax.set_ylabel("Միջին արտադրություն (kW)")
    ax.set_title("Ամսական միջին արտադրություն", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    return fig


# ──────────────────────────────────────────────
# ACF plots
# ──────────────────────────────────────────────

def plot_acf_bar(acf_values: np.ndarray, confidence: float):
    lags = np.arange(len(acf_values))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(lags, acf_values, color='steelblue', alpha=0.7, width=0.8)
    ax.axhline(confidence,  color='red', linestyle='--',
               linewidth=1.5, label=f'95% CI (±{confidence:.3f})')
    ax.axhline(-confidence, color='red', linestyle='--', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Lag (օր)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Autocorrelation Function (ACF) — Սեզոնային պերիոդի որոշում",
                 fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_acf_peaks(acf_values, confidence, top_peaks, top_vals, best_period):
    lags = np.arange(len(acf_values))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(lags, acf_values, color='steelblue',
           alpha=0.5, width=0.8, label='ACF')
    ax.axhline(confidence,  color='red', linestyle='--',
               linewidth=1.5, label='95% CI')
    ax.axhline(-confidence, color='red', linestyle='--', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.8)
    for i, (pk, pv) in enumerate(zip(top_peaks[:5], top_vals[:5])):
        color = 'red' if i == 0 else 'orange'
        ax.bar(pk, acf_values[pk], color=color, alpha=0.9, width=0.8,
               label=f'Peak lag={pk} (ACF={pv:.3f})' if i < 3 else None)
    ax.set_xlabel("Lag (օր)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(
        f"ACF Գագաթներ — Որոշված սեզոնային պերիոդ: {best_period} օր", fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


# ──────────────────────────────────────────────
# Decomposition plot
# ──────────────────────────────────────────────

def plot_decomposition(ts, trend, seasonal, residual, seasonal_period, model_type):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    axes[0].plot(ts.index, ts.values, color='blue', linewidth=1.5)
    axes[0].set_ylabel('SystemProduction (kW)')
    axes[0].set_title('Սկզբնական տվյալներ (Y)', fontweight='bold', fontsize=12)
    axes[0].grid(alpha=0.3)

    axes[1].plot(trend.index, trend.values, color='red', linewidth=2)
    axes[1].set_ylabel('Trend (T)')
    axes[1].set_title('Տենդենց (T)', fontweight='bold', fontsize=12)
    axes[1].grid(alpha=0.3)

    axes[2].plot(seasonal.index, seasonal.values, color='green', linewidth=1.5)
    hline = 1 if model_type == 'multiplicative' else 0
    axes[2].axhline(hline, color='black', linestyle='--',
                    linewidth=1, alpha=0.5)
    label = 'Seasonal (S) [Գործակից]' if model_type == 'multiplicative' else 'Seasonal (S)'
    axes[2].set_ylabel(label)
    axes[2].set_title(
        f'Սեզոնային բաղադրիչ (S) — պերիոդ={seasonal_period} օր (ACF)', fontweight='bold', fontsize=12)
    axes[2].grid(alpha=0.3)

    axes[3].plot(residual.index, residual.values, color='purple', linewidth=1)
    axes[3].axhline(hline, color='black', linestyle='--', linewidth=1)
    axes[3].set_ylabel('Residual (E)')
    axes[3].set_xlabel('Ամսաթիվ')
    axes[3].set_title('Մնացորդներ (E)', fontweight='bold', fontsize=12)
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
# Sinusoidal plots
# ──────────────────────────────────────────────

def plot_sinusoidal(sino: dict):
    amplitude = sino['amplitude']
    phase = sino['phase']
    offset = sino['offset']
    r2 = sino['r2']
    day_of_year = sino['day_of_year']
    detrended = sino['detrended']
    valid = sino['valid']
    peak_day = sino['peak_day']
    trough_day = sino['trough_day']

    full_days = np.arange(0, 365)
    yearly = sinusoidal_model(full_days, amplitude, phase, offset)
    fitted = sinusoidal_model(day_of_year[valid], amplitude, phase, offset)

    peak_dt = pd.Timestamp('2024-01-01') + pd.Timedelta(days=peak_day)
    trough_dt = pd.Timestamp('2024-01-01') + pd.Timedelta(days=trough_day)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(day_of_year[valid], detrended[valid], 'o', alpha=0.4,
                 label='Իրական տվյալներ', markersize=4, color='steelblue')
    axes[0].plot(day_of_year[valid], fitted, 'r-', linewidth=2.5,
                 label=f'Սինուսոիդ (R²={r2:.3f})')
    axes[0].set_xlabel('Տարվա օր')
    axes[0].set_ylabel('Սեզոնային բաղադրիչ (kW)')
    axes[0].set_title(
        'Տարվա տվյալների և սինուսոիդի համեմատություն', fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(full_days, yearly, 'g-', linewidth=2.5, label='Սինուսոիդ')
    axes[1].fill_between(full_days, yearly, alpha=0.15, color='green')
    axes[1].axvline(peak_day,   color='red',  linestyle='--', linewidth=1.5,
                    label=f'Գագաթ — {peak_dt.strftime("%b %d")} (օր {peak_day})')
    axes[1].axvline(trough_day, color='blue', linestyle='--', linewidth=1.5,
                    label=f'Անկում — {trough_dt.strftime("%b %d")} (օր {trough_day})')
    axes[1].axvspan(day_of_year.min(), day_of_year.max(),
                    alpha=0.1, color='orange', label='Տվյալների հատաված')
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_xticks(MONTH_STARTS)
    axes[1].set_xticklabels(MONTH_SHORT, fontsize=8)
    axes[1].set_xlabel('Ամիս')
    axes[1].set_ylabel('Սեզոնային բաղադրիչ (kW)')
    axes[1].set_title('Տարեկան սինուսոիդ (ամբողջ տարի)', fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig, peak_dt, trough_dt


# ──────────────────────────────────────────────
# Prediction timeline plot
# ──────────────────────────────────────────────

def plot_prediction_timeline(
    ts: pd.Series,
    last_date,
    pred_date_ts,
    trend_clean: pd.Series,
    slope: float,
    trend_method: str,
    pred_acf: float,
    pred_sino: float,
    sino_success: bool,
):
    fig, ax = plt.subplots(figsize=(14, 5))
    hist = ts.iloc[-min(90, len(ts)):]
    ax.plot(hist.index, hist.values, color='steelblue',
            linewidth=1.5, label='Իրական տվյալներ')

    future = pd.date_range(start=last_date, end=pred_date_ts, freq='D')
    if "Gitsayin" in trend_method or "Гծ" in trend_method or "Գծ" in trend_method:
        fv = [float(trend_clean.iloc[-1]) + slope *
              i for i in range(len(future))]
    else:
        fv = [float(trend_clean.iloc[-1])] * len(future)
    ax.plot(future, fv, color='red', linewidth=1.5,
            linestyle='--', alpha=0.6, label='Trend (կանխատեսում)')

    ax.scatter([pred_date_ts], [pred_acf], color='red', s=200, zorder=5,
               label=f'ACF կանխատեսում {pred_acf:.1f} kW')
    if sino_success:
        ax.scatter([pred_date_ts], [pred_sino], color='green', s=200, marker='D', zorder=5,
                   label=f'Սինուսոիդ կանխատեսում {pred_sino:.1f} kW')

    ax.axvline(last_date, color='gray', linestyle=':',
               alpha=0.7, label='Տվյալների վերջ')
    ax.set_xlabel("Ամիս")
    ax.set_ylabel("SystemProduction (kW)")
    ax.set_title(
        f"Կանխատեսում — {pred_date_ts.strftime('%d %B %Y')}", fontweight='bold', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
