"""PeakWise AI - Streamlit Interactive Web Application.

Demonstrates the 24-hour Peak-Constrained Energy Optimization Advisor:
1. Probabilistic Forecasting (LightGBM Quantile Models P10, P50, P90)
2. Peak Breach Risk Quantification (Continuous Quantile Interpolation)
3. Domain-Aware Load Disaggregation (Shiftable vs. Non-Shiftable)
4. Linear Programming Load Shift Optimization (Google OR-Tools GLOP)
5. Independent Constraint Validation & Auditing (Zero Tolerance)
6. Quantified Financial Savings & Time-of-Use Economics
7. Live Weather Re-Forecasting (Heatwave Sensitivity Bonus)
8. Rolling-Origin Backtest Evaluation on Realized Actuals
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as go_st
import streamlit as st

# Import core pipeline modules (DO NOT modify logic; UI only imports and calls)
from src.config import (
    BUILDING_ID,
    DEMAND_CHARGE_PER_KW,
    MODELS_DIR,
    PEAK_CAP_KW,
    PROJECT_ROOT,
    RISK_LEVEL,
    TIMESTAMP_COL,
    TOU_HOURLY_RATES,
)
from src.data import load_clean_data
from src.economics import evaluate_economics
from src.forecast import load_models, predict
from src.loads import split
from src.optimizer import optimize_schedule
from src.peak import detect
from src.validator import validate_schedule

# Configure Streamlit page
st.set_page_config(
    page_title="PeakWise AI - Peak-Constrained Energy Advisor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Caching Infrastructure (Sub-3-second live re-forecasts)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_models_and_features():
    """Load and cache pre-trained LightGBM quantile models."""
    return load_models()


@st.cache_data(show_spinner=False)
def get_cached_facility_data(building_id: int) -> pd.DataFrame:
    """Load and cache full historical data for the facility."""
    return load_clean_data(building_id=building_id)


@st.cache_data(show_spinner=False)
def run_cached_forecast(
    building_id: int,
    start_ts_str: str,
    temp_delta: float,
) -> pd.DataFrame:
    """Execute cached 24h probabilistic forecast with optional weather delta."""
    override = {"air_temperature": float(temp_delta)} if temp_delta != 0.0 else None
    return predict(
        building_id=building_id,
        start_ts=start_ts_str,
        weather_override=override,
    )


@st.cache_data(show_spinner=False)
def load_metrics_json() -> Optional[Dict[str, Any]]:
    """Load model evaluation metrics from models/metrics.json."""
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.is_file():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# Custom Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffff;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #DC2626;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-med {
        background-color: #FEF3C7;
        color: #D97706;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #16A34A;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Pre-warm cached models
try:
    get_cached_models_and_features()
except Exception as e:
    st.error(f"Error loading forecast models from {MODELS_DIR}: {e}")

# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/electricity.png", width=64)
st.sidebar.title("Advisor Controls")

st.sidebar.markdown("### 1. Horizon & Date")
# Default to 2016-12-15 (a known high-demand breach day in the test set)
default_date = datetime.date(2016, 12, 15)
min_date = datetime.date(2016, 1, 1)
max_date = datetime.date(2016, 12, 31)

selected_date = st.sidebar.date_input(
    "Forecast Origin Date",
    value=default_date,
    min_value=min_date,
    max_value=max_date,
    help="Select a date for 24-hour ahead peak forecasting and load optimization.",
)

st.sidebar.markdown("### 2. Operational Parameters")
cap_kw = st.sidebar.slider(
    "Peak Power Cap (kW)",
    min_value=70.0,
    max_value=175.0,
    value=135.0,  # Calibrated facility peak cap default to demonstrate breach
    step=1.0,
    help=f"Facility demand charge threshold. Historical 92nd percentile: {PEAK_CAP_KW:.2f} kW.",
)

heatwave_delta = st.sidebar.slider(
    "Heatwave Delta (°C)",
    min_value=-5.0,
    max_value=10.0,
    value=0.0,
    step=0.5,
    help="Live temperature offset applied to forecast window to stress-test HVAC peak response.",
)

shiftable_mult = st.sidebar.slider(
    "Shiftable Share Multiplier",
    min_value=0.5,
    max_value=1.5,
    value=1.0,
    step=0.1,
    help="Sensitivity scaling for flexible loads (e.g., HVAC precool, EV charging, thermal storage).",
)

reforecast_clicked = st.sidebar.button("⚡ Re-forecast & Optimize", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Facility ID**: {BUILDING_ID} (Education)\n\n"
    f"**Demand Charge**: ${DEMAND_CHARGE_PER_KW:.2f} / kW-mo\n\n"
    f"**Optimization Risk**: Planning against P90 risk ceiling"
)

# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">⚡ PeakWise AI: Peak-Constrained Energy Advisor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">24-Hour Probabilistic Forecasting, Cap Breach Detection, '
    'and OR-Tools Constrained Load-Shift Advisory</div>',
    unsafe_allow_html=True,
)

# Navigation Tabs
tab_ops, tab_weather, tab_backtest = st.tabs([
    "📊 24-Hour Operations & Optimization",
    "🌡️ Live Weather Re-Forecast (Bonus)",
    "📈 Test Backtest & Model Metrics",
])

# -----------------------------------------------------------------------------
# Core Pipeline Execution
# -----------------------------------------------------------------------------
start_ts_str = f"{selected_date.strftime('%Y-%m-%d')} 00:00:00"

try:
    # 1. 24h Probabilistic Forecast
    forecast_df = run_cached_forecast(
        building_id=BUILDING_ID,
        start_ts_str=start_ts_str,
        temp_delta=heatwave_delta,
    )

    # 2. Peak Detection & Risk Quantification
    peak_df, peak_summary = detect(forecast_df, cap=cap_kw)

    # 3. Load Disaggregation into Shiftable and Fixed Components
    loads_df = split(forecast_df, primary_use="Education")

    # Apply sensitivity multiplier to shiftable loads
    if shiftable_mult != 1.0:
        loads_df = loads_df.copy()
        loads_df["shiftable_p90"] = np.clip(
            loads_df["shiftable_p90"] * shiftable_mult, 0.0, loads_df["p90"]
        ).round(4)
        loads_df["fixed_p90"] = (loads_df["p90"] - loads_df["shiftable_p90"]).round(4)
        loads_df["shiftable_kw"] = loads_df["shiftable_p90"]
        loads_df["fixed_kw"] = loads_df["fixed_p90"]

    # 4. OR-Tools Two-Stage Linear Programming Optimization
    schedule_df, moves_df, opt_meta = optimize_schedule(
        loads_df=loads_df,
        cap=cap_kw,
        risk_level=RISK_LEVEL,
    )

    # 5. Independent Constraint Validation
    val_report = validate_schedule(
        schedule_df=schedule_df,
        moves_df=moves_df,
        cap=cap_kw,
        is_infeasible=not opt_meta["is_feasible"],
        min_achievable_peak=opt_meta["min_achievable_peak_kw"],
    )

    # 6. Economic Valuation
    econ = evaluate_economics(schedule_df=schedule_df, moves_df=moves_df, cap=cap_kw)

    # Fetch realized actual load if present in dataset
    all_data = get_cached_facility_data(BUILDING_ID)
    date_mask = all_data[TIMESTAMP_COL].dt.date == selected_date
    has_actuals = date_mask.any()
    actuals_series = all_data.loc[date_mask, "meter_reading"].values if has_actuals else None

except Exception as err:
    st.error(f"Error executing advisory pipeline for {selected_date}: {err}")
    st.stop()


# =============================================================================
# TAB 1: 24-Hour Operations & Optimization
# =============================================================================
with tab_ops:
    # -------------------------------------------------------------------------
    # 1. KPI Row
    # -------------------------------------------------------------------------
    kpi_cols = st.columns(5)
    
    baseline_peak = peak_summary["daily_peak_p90_kw"]
    scheduled_peak = opt_meta["scheduled_peak_kw"]
    peak_delta = scheduled_peak - baseline_peak
    breach_hours = peak_summary["breach_hours_count"]
    net_savings = econ["daily_economics"]["net_daily_savings_usd"]
    
    with kpi_cols[0]:
        st.metric(
            label="Predicted Peak (P90)",
            value=f"{baseline_peak:.1f} kW",
            delta=f"Hour {peak_summary['peak_hour']:02d}:00",
            delta_color="off",
        )
    with kpi_cols[1]:
        st.metric(
            label="Facility Peak Cap",
            value=f"{cap_kw:.1f} kW",
            delta=f"Risk: {peak_summary['overall_risk_level']}",
            delta_color="inverse" if peak_summary['overall_risk_level'] != "LOW" else "normal",
        )
    with kpi_cols[2]:
        st.metric(
            label="Cap Breach Hours",
            value=f"{breach_hours} hrs",
            delta=f"Max Over: {peak_summary['max_overshoot_kw']:.1f} kW",
            delta_color="inverse" if breach_hours > 0 else "normal",
        )
    with kpi_cols[3]:
        st.metric(
            label="Scheduled Peak",
            value=f"{scheduled_peak:.1f} kW",
            delta=f"{peak_delta:.1f} kW" if peak_delta != 0 else "No change",
            delta_color="normal" if peak_delta <= 0 else "inverse",
        )
    with kpi_cols[4]:
        st.metric(
            label="Net Daily Savings",
            value=f"${net_savings:.2f}",
            delta=f"Annual: ${econ['annualized_extrapolation']['annual_net_savings_usd']:,.0f}",
            delta_color="normal" if net_savings > 0 else "off",
        )

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2. Chart 1 (Plotly): Forecast Band, Actuals, Cap, Breach Hours
    # -------------------------------------------------------------------------
    st.subheader("1. 24-Hour Probabilistic Forecast & Breach Threat Horizon")

    timestamps = pd.to_datetime(forecast_df[TIMESTAMP_COL])
    hours = timestamps.dt.hour

    fig1 = go.Figure()

    # P10 - P90 Uncertainty Confidence Band
    fig1.add_trace(
        go.Scatter(
            x=hours,
            y=forecast_df["p90"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=hours,
            y=forecast_df["p10"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.18)",
            name="P10–P90 Forecast Uncertainty Band",
            hoverinfo="skip",
        )
    )

    # Median Forecast (P50)
    fig1.add_trace(
        go.Scatter(
            x=hours,
            y=forecast_df["p50"],
            mode="lines+markers",
            line=dict(color="#2563EB", width=2.5),
            marker=dict(size=4),
            name="P50 Expected Load (kW)",
            hovertemplate="Hour %{x:02d}:00<br>P50: %{y:.1f} kW<extra></extra>",
        )
    )

    # Upper Risk Limit (P90)
    fig1.add_trace(
        go.Scatter(
            x=hours,
            y=forecast_df["p90"],
            mode="lines",
            line=dict(color="#1D4ED8", width=1.5, dash="dash"),
            name="P90 Risk Ceiling (kW)",
            hovertemplate="Hour %{x:02d}:00<br>P90: %{y:.1f} kW<extra></extra>",
        )
    )

    # Realized Actual Load (if available for that date)
    if has_actuals and actuals_series is not None and len(actuals_series) == 24:
        fig1.add_trace(
            go.Scatter(
                x=hours,
                y=actuals_series,
                mode="lines+markers",
                line=dict(color="#0F172A", width=2, dash="dot"),
                marker=dict(size=4, symbol="circle-open"),
                name="Realized Actual Demand (kW)",
                hovertemplate="Hour %{x:02d}:00<br>Actual: %{y:.1f} kW<extra></extra>",
            )
        )

    # Peak Cap Threshold
    fig1.add_hline(
        y=cap_kw,
        line_dash="dash",
        line_color="#DC2626",
        line_width=2.5,
        annotation_text=f"Peak Cap Threshold ({cap_kw:.1f} kW)",
        annotation_position="top right",
        annotation_font=dict(color="#DC2626", size=12),
    )

    # Shaded Breach Rectangles for Exceeding Hours
    for idx, row in peak_df.iterrows():
        if row["breach"]:
            h_val = int(pd.to_datetime(row[TIMESTAMP_COL]).hour)
            fig1.add_vrect(
                x0=h_val - 0.45,
                x1=h_val + 0.45,
                fillcolor="rgba(239, 68, 68, 0.22)",
                line_width=0,
                layer="below",
            )

    fig1.update_layout(
        title=f"24-Hour Probabilistic Demand Forecast (Building {BUILDING_ID} | {selected_date})",
        xaxis=dict(
            title="Hour of Day",
            tickmode="linear",
            tick0=0,
            dtick=2,
            range=[-0.5, 23.5],
        ),
        yaxis=dict(title="Electrical Demand (kW)"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=60, b=40),
        height=420,
    )
    st.plotly_chart(fig1, use_container_width=True)

    # -------------------------------------------------------------------------
    # 3. Alerts Table
    # -------------------------------------------------------------------------
    st.subheader("2. Hourly Breach Risk Alerts & Continuous Probability")

    display_peak_df = peak_df.copy()
    display_peak_df["hour_fmt"] = pd.to_datetime(display_peak_df[TIMESTAMP_COL]).dt.strftime("%H:00")
    display_peak_df["breach_prob_pct"] = (display_peak_df["breach_probability"] * 100).round(1).astype(str) + "%"
    display_peak_df["p50_fmt"] = display_peak_df["p50"].map(lambda x: f"{x:.2f} kW")
    display_peak_df["p90_fmt"] = display_peak_df["p90"].map(lambda x: f"{x:.2f} kW")
    display_peak_df["overshoot_fmt"] = display_peak_df["expected_overshoot_kw"].map(lambda x: f"{x:.2f} kW")

    # Format risk badges with HTML
    def format_risk_badge(val: str) -> str:
        if val == "HIGH":
            return '<span class="badge-high">HIGH</span>'
        elif val == "MED":
            return '<span class="badge-med">MED</span>'
        return '<span class="badge-low">LOW</span>'

    display_peak_df["Risk Badge"] = display_peak_df["risk_level"].apply(format_risk_badge)

    table_cols = [
        "hour_fmt",
        "p50_fmt",
        "p90_fmt",
        "breach_prob_pct",
        "Risk Badge",
        "overshoot_fmt",
    ]
    st.write(
        display_peak_df[table_cols]
        .rename(
            columns={
                "hour_fmt": "Hour",
                "p50_fmt": "P50 Expected",
                "p90_fmt": "P90 Risk Level",
                "breach_prob_pct": "Breach Probability",
                "overshoot_fmt": "Expected Overshoot",
            }
        )
        .to_html(escape=False, index=False),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 4. Chart 2: Before vs After Load Curve & Stacked Fixed/Shifted Load
    # -------------------------------------------------------------------------
    st.subheader("3. Optimized Load Profile & Peak Shaving Schedule")

    fig2 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Original vs. Scheduled Load Profile",
            "Disaggregated Load Stack (Fixed + Shifted)",
        ),
    )

    # Left: Before vs After
    fig2.add_trace(
        go.Scatter(
            x=hours,
            y=schedule_df["baseline_total_kw"],
            mode="lines+markers",
            name="Baseline Demand (P90)",
            line=dict(color="#EF4444", width=2.5),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=hours,
            y=schedule_df["scheduled_total_kw"],
            mode="lines+markers",
            name="Scheduled Demand (Optimized)",
            line=dict(color="#10B981", width=2.5),
            marker=dict(size=5),
        ),
        row=1,
        col=1,
    )
    fig2.add_hline(
        y=cap_kw,
        line_dash="dash",
        line_color="#DC2626",
        row=1,
        col=1,
        annotation_text="Cap",
        annotation_position="bottom right",
    )

    # Right: Stacked Fixed + Shiftable served
    fig2.add_trace(
        go.Bar(
            x=hours,
            y=schedule_df["fixed_kw"],
            name="Non-Shiftable Fixed Load",
            marker_color="#94A3B8",
        ),
        row=1,
        col=2,
    )
    # Calculate scheduled shiftable served = scheduled_total - fixed
    sched_shiftable = np.maximum(0.0, schedule_df["scheduled_total_kw"] - schedule_df["fixed_kw"])
    fig2.add_trace(
        go.Bar(
            x=hours,
            y=sched_shiftable,
            name="Scheduled Shiftable Load",
            marker_color="#3B82F6",
        ),
        row=1,
        col=2,
    )

    fig2.update_layout(
        barmode="stack",
        height=420,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig2.update_xaxes(title_text="Hour of Day", dtick=2, row=1, col=1)
    fig2.update_xaxes(title_text="Hour of Day", dtick=2, row=1, col=2)
    fig2.update_yaxes(title_text="Demand (kW)", row=1, col=1)
    fig2.update_yaxes(title_text="Demand (kW)", row=1, col=2)

    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------------------------------------------------
    # 5. Shift Schedule Table
    # -------------------------------------------------------------------------
    st.subheader("4. Actionable Load-Shift Dispatch Instructions")

    if not moves_df.empty:
        total_kwh_moved = moves_df["moved_kw"].sum()
        total_kwh_served = moves_df["served_kw"].sum()
        rebound_losses = total_kwh_served - total_kwh_moved

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Load Shifts Scheduled", f"{len(moves_df)} moves")
        c2.metric("Total Energy Shifted", f"{total_kwh_moved:.2f} kWh")
        c3.metric("Thermodynamic Rebound Loss", f"{rebound_losses:.2f} kWh (+5% overhead)")

        moves_display = moves_df.copy()
        moves_display["from_hour_fmt"] = moves_display["from_hour"].map(lambda h: f"{int(h):02d}:00")
        moves_display["to_hour_fmt"] = moves_display["to_hour"].map(lambda h: f"{int(h):02d}:00")
        moves_display["offset_hrs"] = moves_display["to_hour"] - moves_display["from_hour"]
        moves_display["shift_direction"] = moves_display["offset_hrs"].map(
            lambda o: f"Precool ({o}h)" if o < 0 else f"Defer (+{o}h)"
        )
        moves_display["moved_kw"] = moves_display["moved_kw"].round(2)
        moves_display["served_kw"] = moves_display["served_kw"].round(2)

        st.dataframe(
            moves_display[[
                "from_hour_fmt",
                "to_hour_fmt",
                "shift_direction",
                "moved_kw",
                "served_kw",
            ]].rename(
                columns={
                    "from_hour_fmt": "Origin Hour (from_hour)",
                    "to_hour_fmt": "Dispatch Hour (to_hour)",
                    "shift_direction": "Operation Type",
                    "moved_kw": "Curtailed Load (kW)",
                    "served_kw": "Served Load with Rebound (kW)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("✅ **No Action Needed**: Baseline demand operates safely below the peak cap threshold.")

    # -------------------------------------------------------------------------
    # 6. Constraint Check Panel
    # -------------------------------------------------------------------------
    st.subheader("5. Independent Constraint Verification Audit")

    if not opt_meta["is_feasible"]:
        st.error(
            f"⚠️ **Infeasible Under Requested Cap ({cap_kw:.1f} kW)**\n\n"
            f"**Minimum Achievable Peak ($P^*$)**: `{opt_meta['min_achievable_peak_kw']:.2f} kW`\n\n"
            f"**Diagnostic Root Cause**: {opt_meta['reason']}"
        )

    val_cols = st.columns(len(val_report["constraints"]))
    for col, (c_name, c_res) in zip(val_cols, val_report["constraints"].items()):
        passed = c_res["passed"]
        icon = "✅" if passed else "❌"
        status_label = "PASS" if passed else "FAIL"
        clean_name = c_name.replace("_", " ").title()

        with col:
            st.markdown(
                f"""
                <div style="background-color: {'#ECFDF5' if passed else '#FEF2F2'}; 
                            border: 1px solid {'#10B981' if passed else '#EF4444'};
                            border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-size: 1.1rem;">{icon} <b>{status_label}</b></div>
                    <div style="font-weight: 600; font-size: 0.85rem; margin-top: 4px;">{clean_name}</div>
                    <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">Max viol: {c_res['max_violation']:.1e}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 7. Economics Table
    # -------------------------------------------------------------------------
    st.subheader("6. Quantified Financial Impact & Savings Breakdown")

    p = econ["peaks"]
    d = econ["daily_economics"]
    a = econ["annualized_extrapolation"]

    econ_rows = [
        {"Metric": "Peak Electrical Demand", "Before Optimization": f"{p['baseline_peak_kw']:.2f} kW", "After Optimization": f"{p['scheduled_peak_kw']:.2f} kW", "Net Financial Benefit": f"-{p['peak_reduction_kw']:.2f} kW reduction"},
        {"Metric": "Prorated Daily Demand Charge", "Before Optimization": f"${d['demand_charge_before_usd']:.2f}", "After Optimization": f"${d['demand_charge_after_usd']:.2f}", "Net Financial Benefit": f"+${d['demand_charge_savings_usd']:.2f} / day"},
        {"Metric": "Peak Cap Breach Penalty", "Before Optimization": f"${d['breach_penalty_before_usd']:.2f}", "After Optimization": f"${d['breach_penalty_after_usd']:.2f}", "Net Financial Benefit": f"+${d['breach_penalty_avoided_usd']:.2f} / day avoided"},
        {"Metric": "TOU Volumetric Energy Cost", "Before Optimization": f"${d['tou_energy_cost_before_usd']:.2f}", "After Optimization": f"${d['tou_energy_cost_after_usd']:.2f}", "Net Financial Benefit": f"${d['tou_energy_delta_usd']:.2f} / day delta"},
        {"Metric": "Thermodynamic Rebound Overhead Cost", "Before Optimization": "$0.00", "After Optimization": f"${d['rebound_overhead_cost_usd']:.2f}", "Net Financial Benefit": f"{d['rebound_overhead_kwh']:.2f} kWh overhead"},
        {"Metric": "NET DAILY REALIZED SAVINGS", "Before Optimization": "$0.00", "After Optimization": f"${d['net_daily_savings_usd']:.2f}", "Net Financial Benefit": f"${d['net_daily_savings_usd']:.2f} / day"},
        {"Metric": "EXTRAPOLATED ANNUAL SAVINGS (ESTIMATE)", "Before Optimization": "$0.00", "After Optimization": f"${a['annual_net_savings_usd']:,.2f}", "Net Financial Benefit": f"${a['annual_net_savings_usd']:,.2f} / yr (365d basis)"},
    ]

    st.dataframe(pd.DataFrame(econ_rows), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 2: Live Weather Re-Forecast (Bonus Demonstration)
# =============================================================================
with tab_weather:
    st.subheader("Live Weather-Responsive Sensitivity & Re-Forecasting Engine")
    st.markdown(
        """
        Thermodynamic building energy consumption is heavily sensitive to outdoor temperature swings.
        PeakWise AI recomputes nonlinear weather features (e.g., dew point depression, temperature interactions) 
        and updates the OR-Tools optimization dispatch **in under 3 seconds**.
        """
    )

    col_sim1, col_sim2 = st.columns([1, 2])
    with col_sim1:
        st.markdown("#### Simulation Setup")
        sim_temp = st.slider("Simulated Ambient Temperature Spike (°C)", min_value=-5.0, max_value=12.0, value=5.0, step=0.5, key="tab2_heatwave")
        sim_cap = st.number_input("Peak Cap (kW)", value=float(cap_kw), step=1.0, key="tab2_cap")

    # Generate baseline (0 deg C) vs heatwave scenario
    with st.spinner("Generating live thermodynamic re-forecast..."):
        fc_base = run_cached_forecast(BUILDING_ID, start_ts_str, temp_delta=0.0)
        fc_heat = run_cached_forecast(BUILDING_ID, start_ts_str, temp_delta=sim_temp)

        _, peak_base_sum = detect(fc_base, cap=sim_cap)
        _, peak_heat_sum = detect(fc_heat, cap=sim_cap)

        loads_base = split(fc_base)
        loads_heat = split(fc_heat)

        _, moves_base, opt_base = optimize_schedule(loads_base, cap=sim_cap)
        _, moves_heat, opt_heat = optimize_schedule(loads_heat, cap=sim_cap)

        econ_base = evaluate_economics(optimize_schedule(loads_base, cap=sim_cap)[0], moves_base, cap=sim_cap)
        econ_heat = evaluate_economics(optimize_schedule(loads_heat, cap=sim_cap)[0], moves_heat, cap=sim_cap)

    with col_sim2:
        st.markdown(f"#### Side-by-Side Comparison: Normal vs. Heatwave (+{sim_temp:.1f}°C)")
        comp_df = pd.DataFrame([
            {"Parameter": "P90 Predicted Peak Demand", "Baseline Day (0.0°C)": f"{peak_base_sum['daily_peak_p90_kw']:.2f} kW", f"Heatwave Day (+{sim_temp:.1f}°C)": f"{peak_heat_sum['daily_peak_p90_kw']:.2f} kW", "Impact": f"{peak_heat_sum['daily_peak_p90_kw'] - peak_base_sum['daily_peak_p90_kw']:+.2f} kW"},
            {"Parameter": "Peak Cap Breach Hours", "Baseline Day (0.0°C)": f"{peak_base_sum['breach_hours_count']} hrs", f"Heatwave Day (+{sim_temp:.1f}°C)": f"{peak_heat_sum['breach_hours_count']} hrs", "Impact": f"{peak_heat_sum['breach_hours_count'] - peak_base_sum['breach_hours_count']:+d} hrs"},
            {"Parameter": "Max Cap Overshoot", "Baseline Day (0.0°C)": f"{peak_base_sum['max_overshoot_kw']:.2f} kW", f"Heatwave Day (+{sim_temp:.1f}°C)": f"{peak_heat_sum['max_overshoot_kw']:.2f} kW", "Impact": f"{peak_heat_sum['max_overshoot_kw'] - peak_base_sum['max_overshoot_kw']:+.2f} kW"},
            {"Parameter": "Optimized Scheduled Peak", "Baseline Day (0.0°C)": f"{opt_base['scheduled_peak_kw']:.2f} kW", f"Heatwave Day (+{sim_temp:.1f}°C)": f"{opt_heat['scheduled_peak_kw']:.2f} kW", "Impact": f"{opt_heat['scheduled_peak_kw'] - opt_base['scheduled_peak_kw']:+.2f} kW"},
            {"Parameter": "Total Energy Shifted", "Baseline Day (0.0°C)": f"{moves_base['moved_kw'].sum():.2f} kWh", f"Heatwave Day (+{sim_temp:.1f}°C)": f"{moves_heat['moved_kw'].sum():.2f} kWh", "Impact": f"{moves_heat['moved_kw'].sum() - moves_base['moved_kw'].sum():+.2f} kWh"},
            {"Parameter": "Net Daily Savings", "Baseline Day (0.0°C)": f"${econ_base['daily_economics']['net_daily_savings_usd']:.2f}", f"Heatwave Day (+{sim_temp:.1f}°C)": f"${econ_heat['daily_economics']['net_daily_savings_usd']:.2f}", "Impact": f"${econ_heat['daily_economics']['net_daily_savings_usd'] - econ_base['daily_economics']['net_daily_savings_usd']:+.2f}"},
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Plotly comparison curves
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=hours, y=fc_base["p90"], mode="lines", name="Baseline P90 (kW)", line=dict(color="#64748B", dash="dash")))
    fig_comp.add_trace(go.Scatter(x=hours, y=fc_heat["p90"], mode="lines+markers", name=f"Heatwave (+{sim_temp}°C) P90 (kW)", line=dict(color="#EF4444", width=2.5)))
    fig_comp.add_trace(go.Scatter(x=hours, y=optimize_schedule(loads_heat, cap=sim_cap)[0]["scheduled_total_kw"], mode="lines+markers", name="Optimized Heatwave Schedule (kW)", line=dict(color="#10B981", width=2.5)))
    fig_comp.add_hline(y=sim_cap, line_dash="dash", line_color="#DC2626", annotation_text=f"Cap ({sim_cap} kW)")

    fig_comp.update_layout(
        title=f"Thermodynamic Load Profile Comparison (+{sim_temp:.1f}°C Live Weather Response)",
        xaxis=dict(title="Hour of Day", dtick=2),
        yaxis=dict(title="Demand (kW)"),
        hovermode="x unified",
        height=380,
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# =============================================================================
# TAB 3: Test Backtest & Model Metrics
# =============================================================================
with tab_backtest:
    st.subheader("1. 28-Day Rolling-Origin Test Backtest on Realized Actuals")
    st.markdown(
        """
        **Methodology**: Across all 28 consecutive days in the holdout test period (Dec 4 – Dec 31, 2016),
        the advisor plans load shifts strictly using the **24-hour ahead forecast**, and then executes
        the resulting dispatch schedule against the **realized actual meter reading**.
        """
    )

    # Compute or display rolling backtest table
    test_dates = [
        "2016-12-04", "2016-12-05", "2016-12-06", "2016-12-07", "2016-12-08",
        "2016-12-09", "2016-12-10", "2016-12-11", "2016-12-12", "2016-12-13",
        "2016-12-14", "2016-12-15", "2016-12-16", "2016-12-17", "2016-12-18",
        "2016-12-19", "2016-12-20", "2016-12-21", "2016-12-22", "2016-12-23",
        "2016-12-24", "2016-12-25", "2016-12-26", "2016-12-27", "2016-12-28",
        "2016-12-29", "2016-12-30", "2016-12-31"
    ]

    backtest_records = []
    total_savings = 0.0
    breaches_count = 0
    avoided_count = 0

    with st.spinner("Compiling rolling backtest records across test horizon..."):
        for d_str in test_dates:
            fc_day = run_cached_forecast(BUILDING_ID, f"{d_str} 00:00:00", temp_delta=0.0)
            d_obj = datetime.datetime.strptime(d_str, "%Y-%m-%d").date()
            m = all_data[TIMESTAMP_COL].dt.date == d_obj
            act_load = all_data.loc[m, "meter_reading"].values.astype(float)
            base_pk = float(np.max(act_load))

            l_day = split(fc_day)
            _, mv_day, opt_d = optimize_schedule(l_day, cap=cap_kw)

            # Apply moves to realized actuals
            act_sched = act_load.copy()
            if not mv_day.empty:
                for _, mv in mv_day.iterrows():
                    h = int(mv["from_hour"])
                    t = int(mv["to_hour"])
                    act_sched[h] -= float(mv["moved_kw"])
                    act_sched[t] += float(mv["served_kw"])

            sched_pk = float(np.max(act_sched))
            pk_red = max(0.0, base_pk - sched_pk)

            had_br = base_pk > cap_kw
            avoided = had_br and (sched_pk <= cap_kw)
            if had_br:
                breaches_count += 1
            if avoided:
                avoided_count += 1

            ec = evaluate_economics(
                pd.DataFrame({"baseline_total_kw": act_load, "scheduled_total_kw": act_sched}),
                mv_day,
                cap=cap_kw,
            )
            d_sav = ec["daily_economics"]["net_daily_savings_usd"]
            total_savings += d_sav

            backtest_records.append({
                "Date": d_str,
                "Forecast P90 Peak": f"{float(fc_day['p90'].max()):.2f} kW",
                "Actual Base Peak": f"{base_pk:.2f} kW",
                "Actual Sched Peak": f"{sched_pk:.2f} kW",
                "Realized Reduction": f"{pk_red:.2f} kW",
                "Breach Avoided?": "YES (AVOIDED)" if avoided else ("NO (PARTIAL)" if had_br else "N/A (SAFE)"),
                "Net Realized Savings": f"${d_sav:.2f}",
            })

    # Summary KPIs
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    b_col1.metric("Evaluated Test Days", "28 Days")
    b_col2.metric("Realized Breach Days", f"{breaches_count} Days")
    b_col3.metric("Breaches Avoided", f"{avoided_count} of {breaches_count} ({avoided_count/max(1, breaches_count)*100:.1f}%)")
    b_col4.metric("Total Realized Net Savings", f"${total_savings:,.2f}", delta=f"${total_savings/28:.2f} / day avg")

    st.dataframe(pd.DataFrame(backtest_records), use_container_width=True, hide_index=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2. Machine Learning Metrics vs. Baselines
    # -------------------------------------------------------------------------
    st.subheader("2. Forecasting Accuracy: LightGBM vs. Industry Baselines")

    metrics_data = load_metrics_json()
    if metrics_data and "models" in metrics_data:
        m_dict = metrics_data["models"]
        ml_rows = [
            {"Model": "LightGBM Quantile (Q50)", "MAE (kW)": f"{m_dict.get('lightgbm_q50', {}).get('mae', 0):.2f}", "RMSE (kW)": f"{m_dict.get('lightgbm_q50', {}).get('rmse', 0):.2f}", "CV-RMSE": f"{m_dict.get('lightgbm_q50', {}).get('cv_rmse', 0):.2f}%", "Daily Max Error": f"{m_dict.get('lightgbm_q50', {}).get('daily_max_error', 0):.2f} kW", "Peak-Hour Recall": f"{m_dict.get('lightgbm_q50', {}).get('peak_hour_recall', 0):.1f}%"},
            {"Model": "LightGBM Quantile (Q90)", "MAE (kW)": f"{m_dict.get('lightgbm_q90', {}).get('mae', 0):.2f}", "RMSE (kW)": f"{m_dict.get('lightgbm_q90', {}).get('rmse', 0):.2f}", "CV-RMSE": f"{m_dict.get('lightgbm_q90', {}).get('cv_rmse', 0):.2f}%", "Daily Max Error": f"{m_dict.get('lightgbm_q90', {}).get('daily_max_error', 0):.2f} kW", "Peak-Hour Recall": f"{m_dict.get('lightgbm_q90', {}).get('peak_hour_recall', 0):.1f}%"},
            {"Model": "LightGBM Point (L2)", "MAE (kW)": f"{m_dict.get('lightgbm_l2', {}).get('mae', 0):.2f}", "RMSE (kW)": f"{m_dict.get('lightgbm_l2', {}).get('rmse', 0):.2f}", "CV-RMSE": f"{m_dict.get('lightgbm_l2', {}).get('cv_rmse', 0):.2f}%", "Daily Max Error": f"{m_dict.get('lightgbm_l2', {}).get('daily_max_error', 0):.2f} kW", "Peak-Hour Recall": f"{m_dict.get('lightgbm_l2', {}).get('peak_hour_recall', 0):.1f}%"},
            {"Model": "Yesterday Naive (24h Lag)", "MAE (kW)": f"{m_dict.get('naive_yesterday_24', {}).get('mae', 0):.2f}", "RMSE (kW)": f"{m_dict.get('naive_yesterday_24', {}).get('rmse', 0):.2f}", "CV-RMSE": f"{m_dict.get('naive_yesterday_24', {}).get('cv_rmse', 0):.2f}%", "Daily Max Error": f"{m_dict.get('naive_yesterday_24', {}).get('daily_max_error', 0):.2f} kW", "Peak-Hour Recall": f"{m_dict.get('naive_yesterday_24', {}).get('peak_hour_recall', 0):.1f}%"},
            {"Model": "Seasonal Naive (168h Lag)", "MAE (kW)": f"{m_dict.get('seasonal_naive_168', {}).get('mae', 0):.2f}", "RMSE (kW)": f"{m_dict.get('seasonal_naive_168', {}).get('rmse', 0):.2f}", "CV-RMSE": f"{m_dict.get('seasonal_naive_168', {}).get('cv_rmse', 0):.2f}%", "Daily Max Error": f"{m_dict.get('seasonal_naive_168', {}).get('daily_max_error', 0):.2f} kW", "Peak-Hour Recall": f"{m_dict.get('seasonal_naive_168', {}).get('peak_hour_recall', 0):.1f}%"},
        ]
        st.dataframe(pd.DataFrame(ml_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Metrics not found in models/metrics.json. Execute `python train.py` to generate complete validation metrics.")
