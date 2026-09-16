"""Bike Demand -- Live Ops dashboard.

Streamlit app for the bike demand prediction project. Reads real pipeline
outputs (model registry, test metrics, prediction log) rather than mocking
any numbers, and wires the prediction form to src.predict.predict_new_raw().
"""
import sys
from datetime import date, timedelta
from pathlib import Path

# Ensure the project root (parent of this app/ dir) is on sys.path so
# `from src import ...` resolves regardless of the cwd Streamlit was
# launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from src import config
from src.data import load_raw_data
from src.evaluate import evaluate_model, load_test_data, residual_breakdown_tables
from src.predict import load_current_model, predict_new_raw

# ---------------------------------------------------------------- page setup

st.set_page_config(
    page_title="Bike Demand -- Live Ops",
    page_icon="\U0001F6B2",
    layout="wide",
)

# One color language, used the same way everywhere: teal for the model /
# "good", red for the naive baseline's mistakes / drift breach / negative
# residuals, slate for neutral/secondary ink. No chart in this app invents
# its own palette.
TEAL = "#0F766E"
TEAL_SOFT = "#CCEAE6"
RED = "#B91C1C"
RED_SOFT = "#F8D7D3"
INK = "#111827"
SUBTLE = "#6B7280"
LINE = "#E5E7EB"
CARD_BG = "#FFFFFF"

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

_CSS = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {{ font-family: {FONT}; }}
.stApp {{ background: #FFFFFF; }}
[data-testid="stHeader"] {{ background: #FFFFFF; }}
.block-container {{ padding-top: 2rem; max-width: 1180px; }}
.num {{ font-variant-numeric: tabular-nums; }}
.app-header {{ display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 18px; margin-bottom: 22px; border-bottom: 1px solid {LINE}; }}
.app-header h1 {{ font-size: 1.6rem; margin: 0; color: {INK}; font-weight: 700; letter-spacing: -0.01em; }}
.app-header .meta {{ font-size: 0.82rem; color: {SUBTLE}; text-align: right; line-height: 1.5; }}
.kpi-tile {{ background: {CARD_BG}; border: 1px solid {LINE}; border-radius: 10px; padding: 16px 18px; height: 100%; }}
.kpi-label {{ font-size: 0.78rem; color: {SUBTLE}; margin-bottom: 8px; font-weight: 500; }}
.kpi-value {{ font-size: 1.65rem; font-weight: 700; color: {INK}; }}
.kpi-value.result {{ color: {TEAL}; }}
.kpi-foot {{ font-size: 0.78rem; color: {SUBTLE}; margin-top: 4px; }}
.card {{ background: {CARD_BG}; border: 1px solid {LINE}; border-radius: 10px; padding: 18px 20px 8px 20px; }}
.section-title {{ font-size: 1rem; font-weight: 600; color: {INK}; margin: 0 0 2px 0; display: flex; align-items: center; }}
.section-sub {{ color: {SUBTLE}; font-size: 0.83rem; margin-bottom: 6px; }}
.kpi-label {{ display: flex; align-items: center; }}
.status-pill {{ display: inline-flex; align-items: center; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; }}
.status-ok {{ background: {TEAL_SOFT}; color: {TEAL}; }}
.status-warn {{ background: {RED_SOFT}; color: {RED}; }}
.auto-note {{ font-size: 0.82rem; color: {SUBTLE}; background: #F9FAFB; border: 1px solid {LINE}; border-radius: 8px; padding: 9px 14px; margin: 8px 0 18px 0; }}
.auto-note b {{ color: {INK}; font-weight: 600; }}
.legend-row {{ display: flex; gap: 18px; font-size: 0.78rem; color: {SUBTLE}; margin-top: 6px; }}
.legend-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
</style>
"""
st.markdown(
    _CSS,
    unsafe_allow_html=True,
)


def style_fig(fig, height=320):
    """The single layout every chart in this app goes through, so a bar
    chart, a line chart, and a small-multiple all read as the same
    dashboard instead of three different ones stitched together."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        font=dict(family=FONT, size=12, color=INK),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, size=12), bordercolor=LINE),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=LINE, ticks=""),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", zeroline=False, ticks=""),
    )
    return fig


def signed_colors(values):
    return [TEAL if v >= 0 else RED for v in values]


# ---------------------------------------------------------------- icons
# Small hand-drawn line icons (generic geometric shapes -- no external
# image files, no network dependency), so the dashboard reads as more than
# text-and-numbers without dragging in a decorative stock illustration.
# Each is a raw <svg> string; icon() wraps one with a size/color/margin.

_ICON_PATHS = {
    "bike": '<circle cx="6" cy="17" r="3.5"/><circle cx="18" cy="17" r="3.5"/>'
            '<path d="M6 17 L10 8 L15 8 M10 8 L13 13 M13 13 L18 17 M13 13 L9 13"/>'
            '<circle cx="10" cy="5" r="1.6" fill="currentColor" stroke="none"/>',
    "calendar": '<rect x="3.5" y="5" width="17" height="15" rx="2"/>'
                '<path d="M3.5 9.5 H20.5 M8 3 V7 M16 3 V7"/>',
    "clock": '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5 V12 L15.2 14"/>',
    "trend": '<path d="M4 16 L10 10 L14 14 L20 6"/><path d="M14.5 6 H20 V11.5"/>',
    "percent": '<circle cx="7" cy="7" r="2.4"/><circle cx="17" cy="17" r="2.4"/><path d="M5 19 L19 5"/>',
    "check": '<circle cx="12" cy="12" r="8.5"/><path d="M8 12.5 L11 15.5 L16 9"/>',
    "alert": '<path d="M12 4 L21 19 H3 Z" stroke-linejoin="round"/><path d="M12 10 V14"/>'
             '<circle cx="12" cy="16.8" r="0.9" fill="currentColor" stroke="none"/>',
    "bars": '<path d="M5 19 V13 M12 19 V8 M19 19 V4"/>',
    "activity": '<path d="M3 12 H8 L10.5 5 L14 19 L16.5 12 H21"/>',
    "grid": '<rect x="3.5" y="3.5" width="7.5" height="7.5" rx="1.2"/>'
            '<rect x="13" y="3.5" width="7.5" height="7.5" rx="1.2"/>'
            '<rect x="3.5" y="13" width="7.5" height="7.5" rx="1.2"/>'
            '<rect x="13" y="13" width="7.5" height="7.5" rx="1.2"/>',
    "target": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/><circle cx="12" cy="12" r="0.9" fill="currentColor" stroke="none"/>',
    "flag": '<path d="M6 3 V21"/><path d="M6 4 H18 L15 8 L18 12 H6"/>',
    "sun": '<circle cx="12" cy="12" r="4.2"/>'
           '<path d="M12 2.5 V5 M12 19 V21.5 M2.5 12 H5 M19 12 H21.5 '
           'M5 5 L6.8 6.8 M17.2 17.2 L19 19 M19 5 L17.2 6.8 M6.8 17.2 L5 19"/>',
    "cloud": '<path d="M7 18 H17 A4 4 0 0 0 16.5 10.1 A5.5 5.5 0 0 0 6 11.5 A3.5 3.5 0 0 0 7 18 Z"/>',
    "cloud-rain": '<path d="M7 15 H17 A4 4 0 0 0 16.5 7.1 A5.5 5.5 0 0 0 6 8.5 A3.5 3.5 0 0 0 7 15 Z"/>'
                  '<path d="M9 18.5 L8 21 M13 18.5 L12 21 M17 18.5 L16 21"/>',
    "cloud-snow": '<path d="M7 15 H17 A4 4 0 0 0 16.5 7.1 A5.5 5.5 0 0 0 6 8.5 A3.5 3.5 0 0 0 7 15 Z"/>'
                  '<circle cx="9" cy="19.2" r="0.9" fill="currentColor" stroke="none"/>'
                  '<circle cx="13" cy="19.8" r="0.9" fill="currentColor" stroke="none"/>'
                  '<circle cx="16.5" cy="19" r="0.9" fill="currentColor" stroke="none"/>',
}


def icon(name, color=INK, size=18, margin_right=8):
    """Inline <svg> for one of the icons above, styled to match currentColor."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:-4px; margin-right:{margin_right}px;">{_ICON_PATHS[name]}</svg>'
    )


WEATHER_ICON = {1: "sun", 2: "cloud", 3: "cloud-rain", 4: "cloud-snow"}


# ---------------------------------------------------------------- cached loads

@st.cache_resource
def get_model_and_registry():
    return load_current_model()


@st.cache_data
def get_full_test_evaluation():
    """Runs the model (and the naive baseline) once over the entire frozen
    test window and returns per-row results. Everything the dashboard shows
    "as of" a chosen date is a slice of these two frames, sliced fresh on
    every rerun -- cheap, since slicing ~1,400 rows is instant, and it
    keeps the model itself from being re-run every time the date slider
    moves.
    """
    model, _ = get_model_and_registry()
    test_df = load_test_data()
    test_final, _ = evaluate_model(model, test_df)

    lag_col = f"{config.TARGET_COL}_lag_168h"
    naive_df = test_df.loc[test_df[lag_col].notna(), ["timestamp", config.TARGET_COL, lag_col]].copy()
    naive_df = naive_df.rename(columns={lag_col: "naive_pred"})
    return test_final, naive_df


def metrics_from(df, actual_col, pred_col):
    if df.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}
    y, p = df[actual_col].values, df[pred_col].values
    return {
        "mae": mean_absolute_error(y, p),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mape": mean_absolute_percentage_error(y, p) * 100,
    }


@st.cache_data
def get_prediction_log():
    path = config.PREDICTION_LOG_PATH
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "actual", "predicted", "abs_error"])
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data
def get_raw_history():
    return load_raw_data()


def rolling_mae_series(log_df, window_days):
    if log_df.empty:
        return pd.Series(dtype=float)
    s = log_df.set_index("timestamp")["abs_error"]
    return s.rolling(f"{window_days}D").mean()


# ---------------------------------------------------------------- date-derived feature helpers
# The raw dataset encodes several fields that a user should never have to
# supply by hand -- they follow mechanically from the calendar date. These
# helpers keep that derivation in one place instead of asking for a
# "weekday code" or "season number" in the UI.

SEASON_NAMES = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
WEEKDAY_NAMES = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
                 4: "Thursday", 5: "Friday", 6: "Saturday"}
WEATHER_OPTIONS = {
    1: "Clear / partly cloudy",
    2: "Mist / cloudy",
    3: "Light snow or rain",
    4: "Heavy rain, snow, or fog",
}


def month_to_season(month):
    if month in (3, 4, 5):
        return 1  # spring
    if month in (6, 7, 8):
        return 2  # summer
    if month in (9, 10, 11):
        return 3  # fall
    return 4  # 12, 1, 2 -- winter


def uci_weekday(d):
    """Python's date.weekday() is Monday=0..Sunday=6; the dataset uses
    Sunday=0..Saturday=6, so shift by one."""
    return (d.weekday() + 1) % 7


def celsius_to_norm_temp(t_c):
    """Dataset docs: temp normalized via (t - t_min) / (t_max - t_min),
    t_min=-8, t_max=+39."""
    return float(np.clip((t_c - (-8)) / (39 - (-8)), 0, 1))


def celsius_to_norm_atemp(t_c):
    """Same idea for the 'feels like' temperature, t_min=-16, t_max=+50."""
    return float(np.clip((t_c - (-16)) / (50 - (-16)), 0, 1))


def pct_to_norm_hum(pct):
    return float(np.clip(pct / 100.0, 0, 1))


def kmh_to_norm_wind(kmh):
    """Dataset docs: windspeed normalized by dividing by 67 (max)."""
    return float(np.clip(kmh / 67.0, 0, 1))


# ---------------------------------------------------------------- data

model, registry = get_model_and_registry()
test_final_full, naive_full = get_full_test_evaluation()
log_df = get_prediction_log()
rolling_full = rolling_mae_series(log_df, config.ROLLING_WINDOWS)
baseline_mae = registry["val_metrics"]["val_mae_mean"]
threshold = baseline_mae * config.DRIFT_MAE_MULTIPLIER
raw_history = get_raw_history()

# The monitored test window (what src/monitor.py replays) and the wider
# range the model can still backtest into. One date control drives both
# the prediction and the monitoring metrics below it -- previously these
# were two separate date pickers with two different ranges, which was
# confusing and didn't match how someone would actually use this.
window_start = config.TEST_START.date()
window_end = date(2013, 1, 1)
first_date = raw_history["dteday"].min().date()
last_date = raw_history["dteday"].max().date()
max_predictable_date = last_date + timedelta(days=1)  # == window_end

# ---------------------------------------------------------------- header

st.markdown(
    f"""
    <div class="app-header">
        <h1>{icon('bike', size=26, margin_right=10)}Bike Demand &mdash; Live Ops</h1>
        <div class="meta">Model <b>{registry['current_model']}</b><br>
        trained {registry['trained_at'][:19].replace('T', ' ')} UTC</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- predict a new hour
# Leads the page: pick when, see the estimate immediately. The monitoring
# metrics further down explain how much to trust it, which only makes
# sense to read after you've seen the number they're vouching for.

st.markdown('<div class="card" style="padding-bottom:20px;">', unsafe_allow_html=True)
st.markdown(f"<div class='section-title'>{icon('target', color=TEAL)}Predict a new hour</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-sub'>Pick a date and hour -- the calendar fields below fill in automatically</div>",
    unsafe_allow_html=True,
)

# Date and hour live outside the form so the auto-derived fields (and the
# monitoring section below) update live as soon as either changes, instead
# of only after the weather inputs are submitted.
date_col, hour_col, holiday_col = st.columns([2, 1, 1])
dteday = date_col.date_input(
    "Date", value=max_predictable_date, min_value=first_date, max_value=max_predictable_date,
)
hr = hour_col.selectbox("Hour", list(range(24)), index=8, format_func=lambda h: f"{h:02d}:00")
date_col.caption(
    f"Backtest any date back to {first_date:%b %d, %Y}, or predict {max_predictable_date:%b %d, %Y} -- "
    "the one day past the historical record the model's lag features can still reach."
)

season = month_to_season(dteday.month)
weekday = uci_weekday(dteday)
is_weekend = weekday in (0, 6)
yr = 1 if dteday.year >= 2012 else 0

holiday = 1 if holiday_col.checkbox("Public holiday", value=False) else 0
workingday = 0 if (is_weekend or holiday) else 1

st.markdown(
    f"<div class='auto-note'>Auto-detected &mdash; "
    f"<b>{WEEKDAY_NAMES[weekday]}</b> &middot; <b>{SEASON_NAMES[season]}</b> &middot; "
    f"<b>{'Not a working day' if not workingday else 'Working day'}</b>"
    f"{' (weekend)' if is_weekend and not holiday else ''}</div>",
    unsafe_allow_html=True,
)

with st.form("predict_form"):
    c1, c2 = st.columns(2)
    weathersit = c1.selectbox("Weather", list(WEATHER_OPTIONS.keys()), format_func=lambda k: WEATHER_OPTIONS[k])
    c1.markdown(
        f"<div style='margin-top:-6px; color:{SUBTLE}; font-size:0.82rem;'>"
        f"{icon(WEATHER_ICON[weathersit], color=SUBTLE, size=16)}{WEATHER_OPTIONS[weathersit]}</div>",
        unsafe_allow_html=True,
    )
    temp_c = c2.slider("Temperature (°C)", -10, 40, 20)

    c3, c4 = st.columns(2)
    atemp_c = c3.slider("Feels-like temperature (°C)", -15, 50, 20)
    hum_pct = c4.slider("Humidity (%)", 0, 100, 50)

    wind_kmh = st.slider("Wind speed (km/h)", 0, 67, 15)

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    new_row = pd.DataFrame([{
        "dteday": pd.Timestamp(dteday), "season": season, "yr": yr, "mnth": dteday.month,
        "hr": hr, "holiday": holiday, "weekday": weekday, "workingday": workingday,
        "weathersit": weathersit,
        "temp": celsius_to_norm_temp(temp_c),
        "atemp": celsius_to_norm_atemp(atemp_c),
        "hum": pct_to_norm_hum(hum_pct),
        "windspeed": kmh_to_norm_wind(wind_kmh),
        "casual": 0, "registered": 0, "cnt": 0,
    }])
    preds = predict_new_raw(new_row, historical_raw_df=raw_history)
    pred_value = preds[0]
    if np.isnan(pred_value):
        st.warning("Not enough recent history to compute lag/rolling features for this hour -- prediction unavailable.")
    else:
        st.markdown(
            f"""<div class="kpi-tile" style="max-width:280px; margin-top:14px; border-left:4px solid {TEAL};">
            <div class="kpi-label">{icon('bike', color=TEAL, size=15)}Estimated ride count</div>
            <div class="kpi-value result num">{pred_value:.0f}</div></div>""",
            unsafe_allow_html=True,
        )

if dteday.year not in (2011, 2012):
    st.caption(
        "Note: the model was trained only on 2011-2012 data (the dataset is historically frozen), "
        "so predictions for other years are extrapolations beyond what it has seen."
    )
st.caption(
    "predict_new_raw() concatenates this row onto historical raw data, re-runs the same "
    "feature pipeline used in training, and predicts on the engineered row."
)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- monitoring, as of the same date
# Reuses the date picked above rather than a second control. Monitoring
# only has real data for the test window (Nov 1, 2012 onward), so a
# backtest date further back is clamped up to that window's start.

as_of_date = min(max(dteday, window_start), window_end)

st.markdown(f"<div class='section-title' style='margin-top:4px;'>{icon('activity', color=TEAL)}Monitoring, as of {as_of_date:%b %d, %Y}</div>", unsafe_allow_html=True)
if dteday < window_start:
    st.caption(
        f"Monitoring only covers the simulated test window ({window_start:%b %d, %Y} onward) -- "
        f"showing its earliest point since {dteday:%b %d, %Y} is before that."
    )
else:
    st.caption("Replays the same frozen test window src/monitor.py simulates, using only data through this date.")

test_final = test_final_full[test_final_full["timestamp"].dt.date <= as_of_date]
naive_sliced = naive_full[naive_full["timestamp"].dt.date <= as_of_date]
rolling = rolling_full[rolling_full.index.date <= as_of_date] if len(rolling_full) else rolling_full

model_metrics = metrics_from(test_final, config.TARGET_COL, "prediction")
naive_metrics = metrics_from(naive_sliced, config.TARGET_COL, "naive_pred")
tables = residual_breakdown_tables(test_final) if not test_final.empty else {
    k: pd.DataFrame(columns=["mean_residual", "count"]) for k in ["hr", "weathersit", "holiday", "workingday"]
}
peak_rolling_mae = float(rolling.max()) if len(rolling) else float("nan")
current_rolling_mae = float(rolling.iloc[-1]) if len(rolling) else float("nan")
drift_breached = current_rolling_mae > threshold if len(rolling) else False
mae_improvement = (
    (1 - model_metrics["mae"] / naive_metrics["mae"]) * 100
    if naive_metrics["mae"] and not np.isnan(naive_metrics["mae"]) else float("nan")
)

st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- KPI strip

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""<div class="kpi-tile"><div class="kpi-label">{icon('clock', color=SUBTLE, size=14)}MAE through {as_of_date:%b %d}</div>
        <div class="kpi-value num">{model_metrics['mae']:.1f}</div>
        <div class="kpi-foot num">rides / hour, avg. error</div></div>""",
        unsafe_allow_html=True,
    )
with k2:
    improvement_txt = f"-{mae_improvement:.0f}%" if not np.isnan(mae_improvement) else "n/a"
    st.markdown(
        f"""<div class="kpi-tile"><div class="kpi-label">{icon('trend', color=SUBTLE, size=14)}vs. naive baseline</div>
        <div class="kpi-value num" style="color:{TEAL};">{improvement_txt}</div>
        <div class="kpi-foot num">naive MAE {naive_metrics['mae']:.1f}</div></div>""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""<div class="kpi-tile"><div class="kpi-label">{icon('percent', color=SUBTLE, size=14)}MAPE through {as_of_date:%b %d}</div>
        <div class="kpi-value num">{model_metrics['mape']:.1f}%</div>
        <div class="kpi-foot num">RMSE {model_metrics['rmse']:.1f}</div></div>""",
        unsafe_allow_html=True,
    )
with k4:
    pill_class = "status-warn" if drift_breached else "status-ok"
    pill_text = "Breached" if drift_breached else "Healthy"
    pill_icon = icon("alert", color=RED, size=14) if drift_breached else icon("check", color=TEAL, size=14)
    peak_txt = f"{peak_rolling_mae:.1f}" if not np.isnan(peak_rolling_mae) else "n/a"
    st.markdown(
        f"""<div class="kpi-tile"><div class="kpi-label">Drift status</div>
        <div class="kpi-value num"><span class="status-pill {pill_class}">{pill_icon}{pill_text}</span></div>
        <div class="kpi-foot num">peak rolling MAE {peak_txt} / threshold {threshold:.1f}</div></div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- residuals by hour / drift monitor

col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{icon('bars')}Residuals by hour of day</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-sub'>Average error through {as_of_date:%b %d, %Y}, teal = overpredicts, red = underpredicts</div>",
        unsafe_allow_html=True,
    )
    hr_table = tables["hr"]
    fig = go.Figure()
    fig.add_bar(x=hr_table.index, y=hr_table["mean_residual"], marker_color=signed_colors(hr_table["mean_residual"]))
    fig.add_hline(y=0, line_width=1, line_color=LINE)
    fig = style_fig(fig)
    fig.update_layout(xaxis_title="Hour", yaxis_title="Mean residual", xaxis=dict(dtick=2, **fig.layout.xaxis.to_plotly_json()))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{icon('activity')}Drift monitor</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-sub'>Rolling {config.ROLLING_WINDOWS}-day MAE against a threshold set from validation error</div>",
        unsafe_allow_html=True,
    )
    if len(rolling):
        fig = go.Figure()
        fig.add_scatter(x=rolling.index, y=rolling.values, mode="lines", line=dict(color=TEAL, width=2.5),
                         fill="tozeroy", fillcolor="rgba(15,118,110,0.06)")
        fig.add_hline(y=threshold, line_dash="dash", line_width=1.5, line_color=RED)
        fig = style_fig(fig)
        fig.update_layout(xaxis_title="Simulated date", yaxis_title="Rolling MAE")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"<div class='legend-row'>"
            f"<span><span class='legend-dot' style='background:{TEAL};'></span>rolling MAE</span>"
            f"<span><span class='legend-dot' style='background:{RED};'></span>threshold ({threshold:.1f})</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No prediction log yet -- run `python -m src.monitor` to simulate the replay.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- segment breakdown

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(f"<div class='section-title'>{icon('grid')}Residuals by segment</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='section-sub'>Same over/under-prediction check through {as_of_date:%b %d, %Y}, sliced by weather, holidays, and working days</div>",
    unsafe_allow_html=True,
)
seg_cols = st.columns(3)
seg_labels = {"weathersit": "Weather situation", "holiday": "Holiday", "workingday": "Working day"}
seg_icons = {"weathersit": "cloud", "holiday": "flag", "workingday": "calendar"}
for col, seg in zip(seg_cols, ["weathersit", "holiday", "workingday"]):
    seg_table = tables[seg]
    with col:
        st.markdown(
            f"<div style='font-size:0.85rem; font-weight:600; color:{INK}; margin-bottom:4px;'>"
            f"{icon(seg_icons[seg], color=SUBTLE, size=15)}{seg_labels[seg]}</div>",
            unsafe_allow_html=True,
        )
        fig = go.Figure()
        fig.add_bar(x=[str(i) for i in seg_table.index], y=seg_table["mean_residual"],
                    marker_color=signed_colors(seg_table["mean_residual"]))
        fig.add_hline(y=0, line_width=1, line_color=LINE)
        fig = style_fig(fig, height=220)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "The retraining trigger is a simulated replay of the frozen dataset -- see src/monitor.py."
)
