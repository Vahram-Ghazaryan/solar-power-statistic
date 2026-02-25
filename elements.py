import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from utils import preprocess_dataframe, build_daily_ts, UNITS
from constants import (
    STAT_EXPLANATIONS, METRIC_EXPLANATIONS, SINUSOIDAL_EXPLANATION,
    MONTH_NAMES_ARM, MONTH_MID_DAYS, MODEL_INFO
)
from processing import (
    compute_point_stats, fit_regression,
    detect_seasonal_period, decompose_ts,
    build_yearly_trend, fit_sinusoidal, sinusoidal_model,
    predict_acf, predict_sino,
)
from graphics import (
    plot_histogram, plot_boxplot, plot_correlation_heatmap,
    plot_regression_1d, plot_regression_2d,
    plot_weather_scatter, plot_monthly_bar,
    plot_daily_ts, plot_hourly_bar, plot_monthly_line,
    plot_acf_bar, plot_acf_peaks,
    plot_decomposition, plot_sinusoidal,
    plot_prediction_timeline,
)


# ══════════════════════════════════════════════════════════
# 1. DATA UPLOAD
# ══════════════════════════════════════════════════════════

def show_data_upload():
    uploaded = st.file_uploader("Բեռնել CSV ֆայլը", type=["csv"])
    if uploaded is None:
        st.info("⬆️ Խնդրում ենք բեռնել CSV ֆայլը՝ վերլուծությունը սկսելու համար")
        st.write("""
        ### Ակնկալվող ֆայլի ձևաչափ:
        - **Date-Hour(NMT)**: Ամսաթիվ-Ժամ (օրինակ՝ 01.01.2017-00:00)
        - **WindSpeed**: Քամու արագություն
        - **Sunshine**: Արևի ճառագայթում
        - **AirPressure**: Օդի ճնշում
        - **Radiation**: Ռադիացիա
        - **AirTemperature**: Օդի ջերմաստիճան
        - **RelativeAirHumidity**: Օդի հարաբերական խոնավություն
        - **SystemProduction**: Համակարգի արտադրություն
        """)
        return None

    df = pd.read_csv(uploaded)
    df = preprocess_dataframe(df)

    st.subheader("📄 Տվյալների նախադիտում")
    st.dataframe(df.head(20))
    st.write(f"**Ընդհանուր տողեր:** {len(df)}")
    st.write(
        f"**Ժամանակահատված:** {df['DateTime'].min()} - {df['DateTime'].max()}")
    return df


# ══════════════════════════════════════════════════════════
# 2. POINT STATISTICS
# ══════════════════════════════════════════════════════════

def show_point_statistics(df: pd.DataFrame, numeric_cols: list) -> str:
    st.subheader("📌 Կետային բնութագրիչներ")

    with st.expander("ℹ️ Կետային բնութագրիչների մասին"):
        st.markdown("""
        **Կետային բնութագրիչները** օգտագործվում են տվյալների կենտրոնական տեղադրությունը,
        փոփոխականությունը և բաշխման ձևը բնութագրելու համար։
        """)

    col = st.selectbox("Ընտրիր սյունը վերլուծության համար", numeric_cols,
                       index=numeric_cols.index('SystemProduction'))
    unit = UNITS.get(col, '')

    stats_df = compute_point_stats(df[col], unit)
    st.table(stats_df)

    selected = st.selectbox("Ընտրեք բնութագիրը՝ մանրամասն տեղեկություն ստանալու համար:",
                            options=list(STAT_EXPLANATIONS.keys()))
    with st.expander(f"ℹ️ {selected}", expanded=True):
        st.markdown(STAT_EXPLANATIONS[selected])

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_histogram(df[col], col))
    with c2:
        st.pyplot(plot_boxplot(df[col], col))
    return col


# ══════════════════════════════════════════════════════════
# 3. INTERVAL STATISTICS
# ══════════════════════════════════════════════════════════

def show_interval_statistics(df: pd.DataFrame, numeric_cols: list, col: str = 'SystemProduction'):
    st.subheader("📦 Միջակայքային վիճակագրություն")
    col = numeric_cols[numeric_cols.index(col)]
    bins = st.slider("Միջակայքերի քանակը", 3, 20, 5)
    df_t = df.copy()
    df_t["interval"] = pd.cut(df_t[col], bins=bins)
    ist = df_t.groupby("interval")[col].agg(
        ["count", "mean", "std", "min", "max"])
    ist.columns = ["Քանակ", "Միջին",
                   "Ստանդարտ շեղում", "Մինիմում", "Մաքսիմում"]
    st.dataframe(ist.style.format("{:.2f}"))


# ══════════════════════════════════════════════════════════
# 4. CORRELATION
# ══════════════════════════════════════════════════════════

def show_correlation(df: pd.DataFrame, numeric_cols: list):
    st.subheader("🔗 Կոռելյացիոն վերլուծություն")
    corr = df[numeric_cols].corr()
    st.pyplot(plot_correlation_heatmap(corr))

    if 'SystemProduction' in numeric_cols:
        pc = corr['SystemProduction'].sort_values(ascending=False)[1:]
        st.write("**Ամենաուժեղ կապերը SystemProduction-ի հետ:**")
        for var, v in pc.items():
            st.write(f"- {var}: {v:.4f}")


# ══════════════════════════════════════════════════════════
# 5. REGRESSION
# ══════════════════════════════════════════════════════════

def show_regression(df: pd.DataFrame, numeric_cols: list):
    st.subheader("📈 Ռեգրեսիոն վերլուծություն")

    y_col = st.selectbox("Ընտրեք կախյալ փոփոխականը (Y)", numeric_cols,
                         index=numeric_cols.index('SystemProduction'))
    avail = [c for c in numeric_cols if c != y_col]
    x_cols = st.multiselect("Ընտրեք անկախ փոփոխականները (X)", avail,
                            default=['Radiation', 'Sunshine'] if 'Radiation' in avail else avail[:2])
    if not x_cols:
        return

    model, X, y, y_pred, metrics = fit_regression(df, x_cols, y_col)

    st.write("### 📐 Ռեգրեսիոն արժեքներ")
    with st.expander("ℹ️ Ռեգրեսիոն մետրիկների մանրամասն բացատրություն"):
        st.markdown("**Ռեգրեսիոն վերլուծության մասին**՝ ցույց է տալիս, թե կախյալ փոփոխականը (Y) "
                    "ինչպես է կապված անկախ փոփոխականների (X) հետ։")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{metrics['r2']:.4f}")
    c2.metric("RMSE", f"{metrics['rmse']:.2f}")
    c3.metric("MAE", f"{metrics['mae']:.2f}")
    c4.metric("Intercept", f"{metrics['intercept']:.2f}")

    sel_m = st.selectbox("Ընտրեք մետրիկը՝ մանրամասն տեղեկություն ստանալու համար:",
                         list(METRIC_EXPLANATIONS.keys()), key="metric_selector")
    with st.expander(f"ℹ️ {sel_m}", expanded=True):
        st.markdown(METRIC_EXPLANATIONS[sel_m])

    coeffs = pd.DataFrame({"Փոփոխական": x_cols, "Գործակից": model.coef_})
    st.dataframe(coeffs.style.format({"Գործակից": "{:.4f}"}))

    eq = f"{y_col} = {model.intercept_:.2f}"
    for var, c in zip(x_cols, model.coef_):
        eq += f" + ({c:.4f}) × {var}"
    st.write("**Ռեգրեսիայի հավասարում:**")
    st.code(eq)

    if len(x_cols) == 1:
        st.pyplot(plot_regression_1d(X, y, y_pred, x_cols[0], y_col))
    elif len(x_cols) == 2:
        st.pyplot(plot_regression_2d(X, y, y_pred, x_cols, y_col, model))
    else:
        st.info(
            "ℹ️ Գրաֆիկական պատկերումը հասանելի է միայն 1 կամ 2 անկախ փոփոխականների դեպքում։")

# ══════════════════════════════════════════════════════════
# 6. WEATHER IMPACT
# ══════════════════════════════════════════════════════════


def show_weather_impact(df: pd.DataFrame):
    st.subheader("🌤️ Եղանակային պայմանների ազդեցությունը")
    st.pyplot(plot_weather_scatter(df))

# ══════════════════════════════════════════════════════════
# 7. HOURLY & MONTHLY
# ══════════════════════════════════════════════════════════


def show_hourly_monthly(df: pd.DataFrame):
    st.write("### 🕐 Ժամային միջին արտադրություն (24 ժամ)")
    hourly = df.groupby('Hour')['SystemProduction'].mean()
    st.pyplot(plot_hourly_bar(hourly))

    st.write("### 📆 Ամսական միջին արտադրություն")
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    monthly = df.groupby('Month_Name')[
        'SystemProduction'].mean().reindex(month_order)
    st.pyplot(plot_monthly_line(monthly))


# ══════════════════════════════════════════════════════════
# 8. CATEGORICAL REGRESSION
# ══════════════════════════════════════════════════════════

def show_categorical_regression(df: pd.DataFrame):
    st.subheader("📅 Ամսական ռեգրեսիոն վերլուծություն")
    df_m = df[["Month_Name", "SystemProduction"]].dropna()
    X_m = pd.get_dummies(df_m["Month_Name"], drop_first=True)
    y_m = df_m["SystemProduction"]
    mdl = LinearRegression().fit(X_m, y_m)
    r2 = r2_score(y_m, mdl.predict(X_m))
    st.write(f"**R² Score (ամսական ռեգրեսիոն մոդել):** {r2:.4f}")

    cm = pd.DataFrame({"Ամիս": X_m.columns, "Գործակից": mdl.coef_})
    st.write("**Ամիսների ազդեցությունը SystemProduction-ի վրա:**")
    st.dataframe(cm.style.format({"Գործակից": "{:.4f}"}))
    st.pyplot(plot_monthly_bar(cm))


# ══════════════════════════════════════════════════════════
# 9. TIME SERIES
# ══════════════════════════════════════════════════════════

def show_time_series(df: pd.DataFrame):
    st.subheader("⏱️ Ժամանակային շարքի վերլուծություն")
    st.write("### 📅 Օրական միջին արտադրություն")
    ts = build_daily_ts(df)
    st.pyplot(plot_daily_ts(ts))

    st.write("### 📊 Ժամանակային շարքի դեկոմպոզիցիա")
    with st.expander("ℹ️ Ժամանակային շարքի դեկոմպոզիցիայի մասին"):
        st.markdown("""
        **Ժամանակային շարքի դեկոմպոզիցիան** բաժանում է ժամանակային շարքը երեք հիմնական
        բաղադրիչների՝ տենդենց (T), սեզոնային բաղադրիչ (S) և մնացորդներ (E)։
        """)

    st.write(
        "**Մոդելներ:** Y = T + S + E (հավելումային) և Y = T × S × E (բազմապատկային)")

    decomp_choice = st.radio(
        "Ընտրիր մոդելը:", [
            "Հավելումային (Y = T + S + E)", "Բազմապատկային (Y = T × S × E)"],
        horizontal=True,
    )
    model_key = ("Հավելումային մոդել (Y = T + S + E)"
                 if "Հավելումային" in decomp_choice
                 else "Բազմապատկային մոդել (Y = T × S × E)")
    with st.expander(f"ℹ️ {model_key} - Մանրամասն բացատրություն", expanded=False):
        st.markdown(MODEL_INFO[model_key])
    model_type = 'additive' if 'Հավելումային' in decomp_choice else 'multiplicative'

    data_days = len(ts)
    _show_acf_section(ts, data_days, model_type)


def _show_acf_section(ts: pd.Series, data_days: int, model_type: str):
    st.write("### 🔍 Սեզոնային պերիոդի որոշումը ACF-ի միջոցով")

    with st.expander("ℹ️ ACF-ի մասին (Autocorrelation Function)"):
        st.markdown("""
        **Autocorrelation Function (ACF)** — ավտոկոռելյացիոն ֆունկցիա — չափում է, թե
        ժամանակային շարքը որքանով է կապված ինքն իր հետ՝ lag (ուշացում) կիրառելով։

        **Ինչպես է ACF-ն օգնում սեզոնայնությունը գտնել:**
        - Եթե ACF(lag=k) ունի **մեծ (գագաթնային) արժեք**, ապա k օր ուշացումով տվյալները խիստ կոռելյացված են
        - Ամենամեծ գագաթնային lag-ը → **սեզոնային պերիոդ**

        **Բանաձև:** ACF(k) = Σ[(Yₜ - Ȳ)(Yₜ₋ₖ - Ȳ)] / Σ[(Yₜ - Ȳ)²]

        - ACF ≈ 1 → շատ ուժեղ կապ
        - ACF ≈ 0 → կապ չկա
        - ACF ≈ -1 → հակառակ կապ
        """)

    max_lag = min(data_days // 2 - 1, 365)
    if max_lag < 7:
        st.warning(
            "Անբավարար տվյալներ ACF հաշվարկի համար (պետք է առնվազն 14 օր)։")
        return

    acf_info = detect_seasonal_period(ts)
    acf_vals = acf_info["acf_values"]
    confidence = acf_info["confidence"]

    st.pyplot(plot_acf_bar(acf_vals, confidence))

    if acf_info["top_peaks"] is not None:
        top_peaks = acf_info["top_peaks"]
        top_vals = acf_info["top_vals"]
        best_period = acf_info["best_period"]
        best_acf_val = float(top_vals[0])

        st.pyplot(plot_acf_peaks(acf_vals, confidence,
                  top_peaks, top_vals, best_period))

        st.success(f"""
**✅ ACF-ի արդյունք — Սեզոնային պերիոդ**

| Կարգ | Lag (օր) | ACF արժեք | Նշանակություն |
|------|----------|-----------|---------------|
{''.join([f"| {'🥇' if i == 0 else '🥈' if i == 1 else '🥉' if i == 2 else '▫️'} {i+1} | **{pk}** | {pv:.4f} | {'← Ընտրված' if i == 0 else ''} |\n" for i, (pk, pv) in enumerate(zip(top_peaks[:5], top_vals[:5]))])}

**Ընտրված սեզոնային պերիոդ:** `{best_period}` օր  
**ACF արժեք:** `{best_acf_val:.4f}`  
**Տվյալների ծավալ:** `{data_days}` օր
            """)
        seasonal_period = best_period
        period_type = f"ACF-ով որոշված ({best_period} օր)"
    else:
        st.warning(
            "⚠️ ACF-ը չգտավ վիճակագրորեն նշանակալի գագաթ։ Կկիրառվի կանխադրված պերիոդ՝ 7 օր։")
        seasonal_period = min(7, data_days // 2)
        period_type = "կանխադրված (7 օր)"

    start_date = ts.index.min()
    end_date = ts.index.max()
    max_period = data_days // 2

    st.info(f"""
**📊 Տվյալների ժամանակահատված:** {data_days} օր  
**Ամիսներ:** {start_date.strftime('%B %Y')} — {end_date.strftime('%B %Y')}  
**Սեզոնային պերիոդ (ACF):** {seasonal_period} օր ({period_type})  
**Ամբողջ ցիկլեր:** {data_days / seasonal_period:.1f}
        """)

    if data_days < 14 or seasonal_period > max_period:
        msg = (f"ACF-ով որոշված պերիոդը ({seasonal_period}) գերազանցում է տվյալների կեսը ({max_period})։"
               if seasonal_period > max_period
               else "Անբավարար տվյալներ (պետք է առնվազն 14 օր)։")
        st.warning(msg)
        return

    _show_decomposition_section(ts, data_days, seasonal_period, model_type)


def _show_decomposition_section(ts, data_days, seasonal_period, model_type):
    try:
        decomp, shift = decompose_ts(ts, seasonal_period, model_type)
        if shift > 0:
            st.warning(
                f"⚠️ Բազմապատկային մոդելի համար ավելացվել է {shift:.2f} kW տեղաշարժ։")

        trend = decomp.trend
        seasonal = decomp.seasonal
        residual = decomp.resid

        st.pyplot(plot_decomposition(ts, trend, seasonal,
                  residual, seasonal_period, model_type))

        st.write("### 📈 Դեկոմպոզիցիայի վիճակագրություն")
        ul = 'kW' if model_type == 'additive' else 'գործ.'
        c1, c2, c3 = st.columns(3)
        for ax_col, comp, name in [(c1, trend,    "Տենդենց (T)"),
                                   (c2, seasonal, "Սեզոնային (S)"),
                                   (c3, residual, "Մնացորդներ (E)")]:
            with ax_col:
                st.write(f"**{name}:**")
                st.write(f"Միջին: {comp.mean():.2f} {ul}")
                st.write(f"Ստ.շեղում: {comp.std():.2f} {ul}")
                st.write(f"Մին:   {comp.min():.2f} {ul}")
                st.write(f"Մաքս:  {comp.max():.2f} {ul}")
        if model_type == 'additive':
            st.info("**📌 Հավելումային մոդել:** Y = T + S + E\n\n- S > 0 → արտ. ավելի բարձր\n- S < 0 → արտ. ավելի ցածր\n- E ≈ 0 → մոդելը լավ է")
        else:
            st.info("**📌 Բազմապատկային մոդել:** Y = T × S × E\n\n- S > 1 → արտ. ավելի բարձր\n- S < 1 → արտ. ավելի ցածր\n- E ≈ 1 → մոդելը լավ է")
        st.write("---")
        st.write("### 🎯 Կանխատեսման բանաձևեր")
        _show_quick_prediction(ts, trend, seasonal,
                               seasonal_period, model_type, shift)

        st.write("---")
        st.write("### 🔮 Սինուսոիդային սեզոնային մոդելավորում")
        _show_sinusoidal_section(ts, trend, data_days)

        st.write("---")
        st.write("### 📅 Կանխատեսում ըստ ամսաթվի")
        _show_prediction_tool(ts, trend, seasonal,
                              seasonal_period, model_type, shift, data_days)

    except Exception as e:
        st.warning(f"Դեկոմպոզիցիան չի կատարվել: {e}")


def _show_quick_prediction(ts, trend, seasonal, seasonal_period, model_type, shift):
    t_last = float(trend.dropna().iloc[-1])
    s_next = float(seasonal.iloc[len(ts) % seasonal_period])

    if model_type == 'additive':
        pred = t_last + s_next
        st.success(f"""
**✅ Y(t) = T(t) + S(t) + E(t)**  
Կանխատեսման համար (E ≈ 0): **Y_predicted(t) = T(t) + S(t)**

Սեզոնային պերիոդ (ACF): **{seasonal_period} օր**  
Վաղվա սեզոնային արժեք:  **{s_next:.2f} kW**  
Վերջին տենդենց: **{t_last:.2f} kW**  
→ Կանխատեսում:  **{pred:.2f} kW**
        """)
    else:
        pred = t_last * s_next - shift
        st.success(f"""
**✅ Y(t) = T(t) × S(t) × E(t)**  
Կանխատեսման համար (E ≈ 1): **Y_predicted(t) = T(t) × S(t)**

Սեզոնային պերիոդ (ACF): **{seasonal_period} օր**  
Վաղվա սեզոնային գործ.: **{s_next:.4f}**  
Վերջին տենդենց: **{t_last:.2f} kW**  
→ Կանխատեսում: **{pred:.2f} kW**
        """)


def _show_sinusoidal_section(ts, trend, data_days):
    with st.expander("ℹ️ Ի՞նչ է սինուսոիդային մոդելը — Մանրամասն բացատրություն", expanded=False):
        st.markdown(SINUSOIDAL_EXPLANATION)

    yearly_trend = build_yearly_trend(ts)
    sino = fit_sinusoidal(ts, yearly_trend)

    if not sino["sino_success"]:
        st.warning("Սինուսոիդ մոդելը չհաջողվեց։")
        return

    fig, peak_dt, trough_dt = plot_sinusoidal(sino)
    st.pyplot(fig)

    amplitude = sino["amplitude"]
    phase = sino["phase"]
    offset = sino["offset"]
    r2 = sino["r2"]
    peak_day = sino["peak_day"]
    trough_day = sino["trough_day"]

    st.success(f"""
**✅ Սինուսոիդ մոդելի պարամետրեր**

| Պարամետր | Արժեք | Իմաստ |
|----------|-------|-------|
| **Ամպլիտուդա (A)** | {amplitude:.2f} kW | Ամառ/ձմեռ տատանման կեսը |
| **Ֆազա (φ)** | {phase:.4f} ռ ({phase*180/np.pi:.1f}°) | Ցիկլի ժամ. տեղաշարժ |
| **Օֆսեթ (C)** | {offset:.2f} kW | Սեզ. բաղ. միջին մակ. |
| **R²** | {r2:.4f} | Համապ. որակ |
| **Ամռան գագաթ** | {peak_dt.strftime("%B %d")} (օր {peak_day}) | Ամենամեծ արտ. |
| **Ձմռան անկում** | {trough_dt.strftime("%B %d")} (օր {trough_day}) | Ամենափոքր արտ. |

```
S(day) = {amplitude:.2f} × sin(2π × day/365 {"+" if phase >= 0 else ""}{phase:.4f}) {"+" if offset >= 0 else ""}{offset:.2f}
```
    """)

    st.write("**📅 Ամսական սինուսոիդ արժեքներ**")
    day_of_year = np.array(
        [(d - pd.Timestamp(f'{d.year}-01-01')).days for d in ts.index])
    rows = []
    for mn, md in zip(MONTH_NAMES_ARM, MONTH_MID_DAYS):
        sv = sinusoidal_model(md, amplitude, phase, offset)
        has = day_of_year.min() <= md <= day_of_year.max()
        rows.append({
            'Ամիս': mn, 'Տ. օր': md,
            'S(day) kW': f"{sv:.2f}",
            'Աղբյուր': '✅ Իրական' if has else '🔮 Մոդ.'
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.session_state['_sino'] = sino
    st.session_state['_yearly_trend'] = yearly_trend


def _show_prediction_tool(ts, trend, seasonal, seasonal_period, model_type, shift, data_days):
    with st.expander("🔮 Կանխատեսման գործիք", expanded=True):
        st.markdown("**Ընտրիր ամսաթիվ՝ կանխատեսելու համար**")

        c_date, c_method = st.columns(2)
        with c_date:
            min_future = (ts.index[-1] + pd.Timedelta(days=1)).date()
            pred_date = st.date_input(
                "📆 Կանխատեսման ամսաթիվ",
                value=min_future + pd.Timedelta(days=30),
                min_value=min_future,
                help="Ընտրիր ամսաթիվ տվյալների վերջին օրվանից հետո",
            )
        with c_method:
            trend_method = st.selectbox(
                "📈 Տրենդի գնահատման մեթոդ",
                ["Վերջին արժեք (պարզ)",
                 "Գծային էքստրապոլյացիա (ավելի ճշգրիտ)"],
                help="Ինչպես գնահատել տրենդի ապագա արժեքը",
            )

        if st.button("🚀 Կատարել կանխատեսումը", type="primary"):
            pred_ts = pd.Timestamp(pred_date)
            last_date = ts.index[-1]
            days_ahead = (pred_ts - last_date).days
            pred_doy = (pred_ts - pd.Timestamp(f'{pred_ts.year}-01-01')).days

            acf_res = predict_acf(
                trend, seasonal, seasonal_period,
                days_ahead, trend_method, model_type, shift,
            )

            sino = st.session_state.get('_sino', {'sino_success': False})
            yearly_trend = st.session_state.get('_yearly_trend', None)
            sino_success = sino.get('sino_success', False)

            if sino_success and yearly_trend is not None:
                sino_res = predict_sino(
                    yearly_trend, sino, days_ahead, pred_doy, trend_method)
                pred_sino_val = sino_res["pred"]
                trend_pred_sino = sino_res["trend_pred"]
                seasonal_pred_sino = sino_res["seasonal_pred"]
            else:
                pred_sino_val = trend_pred_sino = seasonal_pred_sino = 0.0

            st.write("---")
            st.write(
                f"#### 📊 Կանխատեսման արդյունքներ — {pred_date.strftime('%d %B %Y')}")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("📅 Ամսաթիվ",  pred_date.strftime('%d.%m.%Y'))
            m2.metric("⏳ Օրեր հետո", f"{days_ahead} օր")
            m3.metric("🗓️ Տարվա օր", str(pred_doy))
            m4.metric("🌙 Ամիս",      pred_ts.strftime('%B'))

            r1, r2_col = st.columns(2)
            with r1:
                st.metric("🔄 ACF սեզոնային մոդել", f"{acf_res['pred']:.1f} kW",
                          help="Seasonal Decompose + ACF period")
                st.caption(
                    f"Տրենդ: {acf_res['trend_pred']:.1f} kW | Սեզ.: {acf_res['seasonal_pred']:.2f}")
            with r2_col:
                if sino_success:
                    st.metric("〰️ Սինուսոիդ մոդել", f"{pred_sino_val:.1f} kW",
                              help="A × sin(2π × day/365 + φ) + C")
                    st.caption(
                        f"Տրենդ: {trend_pred_sino:.1f} kW | Սեզ.: {seasonal_pred_sino:.2f}")
                else:
                    st.warning("Սինուսոիդ մոդելը հասանելի չէ")

            with st.expander("🔍 Հաշվարկի մանրամասներ"):
                if model_type == 'additive':
                    f_acf = f"Y = T + S = {acf_res['trend_pred']:.2f} + {acf_res['seasonal_pred']:.2f} = {acf_res['pred']:.2f} kW"
                    f_sino = f"Y = T + S = {trend_pred_sino:.2f} + {seasonal_pred_sino:.2f} = {pred_sino_val:.2f} kW"
                else:
                    f_acf = f"Y = T×S = {acf_res['trend_pred']:.2f}×{acf_res['seasonal_pred']:.4f} = {acf_res['pred']:.2f} kW"
                    f_sino = f"Y = T + S = {trend_pred_sino:.2f} + {seasonal_pred_sino:.2f} = {pred_sino_val:.2f} kW"

                trend_clean = trend.dropna()
                st.markdown(f"""
**Տրենդի գնահատում ({acf_res['method_name']}):**
- Վերջին հայտնի տրենդ՝ **{float(trend_clean.iloc[-1]):.2f} kW** ({last_date.strftime('%d.%m.%Y')})
- Կանխատեսված տրենդ՝ **{acf_res['trend_pred']:.2f} kW**

**Սեզոնային բաղադրիչ:**
- ACF (պ={seasonal_period}, ինդ={acf_res['seasonal_idx']})՝ **{acf_res['seasonal_pred']:.4f}**
- Սինուսոիդ (օր={pred_doy})՝ **{seasonal_pred_sino:.4f}**

**Կանխատեսման բանաձևեր:**
```
ACF:  {f_acf}
Sino: {f_sino}
```
**⚠️ Կանխատեսման ճշգրտության մասին:**
- 1-30 օր → ավելի հուսալի
- 31-90 օր → միջին կանխատեսելիություն
- 90+ օր → ավելի ոչ հուսալի
                """)

            st.pyplot(plot_prediction_timeline(
                ts, last_date, pred_ts,
                trend.dropna(), acf_res['slope'], trend_method,
                acf_res['pred'], pred_sino_val, sino_success,
            ))


# ══════════════════════════════════════════════════════════
# 10. SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════

def show_summary_statistics(df: pd.DataFrame, numeric_cols: list):
    st.subheader("📊 Ամփոփ վիճակագրություն")
    ss = df[numeric_cols].describe().T
    ss.columns = ['Քանակ', 'Միջին', 'Ստ.շեղ',
                  'Մին', '25%', '50%', '75%', 'Մաքս']
    st.dataframe(ss.style.format("{:.2f}"))
