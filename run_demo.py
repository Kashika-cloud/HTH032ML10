"""PeakWise AI CLI Demonstration & Evaluation Runner (Phase 3).

Demonstrates the full end-to-end advisor workflow:
1. Probabilistic Forecasting (P10, P50, P90 via src.forecast)
2. Peak Detection & Risk Quantification (src.peak)
3. Load Disaggregation into Shiftable and Fixed Components (src.loads)
4. Linear Programming Load Shift Optimization (src.optimizer)
5. Independent Schedule Constraint Validation (src.validator)
6. Financial Savings & Economic Quantification (src.economics)

Supports:
- Single-day operational scenario:
    python run_demo.py --date 2016-12-15
    python run_demo.py --date 2016-12-15 --heatwave 5.0
- Test-set rolling backtest on realized actual loads:
    python run_demo.py --backtest
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    BUILDING_ID,
    PEAK_CAP_KW,
    RISK_LEVEL,
    TIMESTAMP_COL,
)
from src.data import load_clean_data
from src.economics import evaluate_economics
from src.forecast import predict
from src.loads import split
from src.optimizer import optimize_schedule
from src.peak import detect
from src.validator import validate_schedule

# Suppress informational logs during clean CLI presentation
for mod_name in ["peakwise.forecast", "peakwise.data", "peakwise.features", "peakwise.peak", "peakwise.loads", "peakwise.optimizer", "peakwise.validator", "peakwise.economics"]:
    logging.getLogger(mod_name).setLevel(logging.WARNING)


def run_single_day_demo(
    date_str: str,
    heatwave_delta: float = 0.0,
    cap_kw: float = PEAK_CAP_KW,
    building_id: int = BUILDING_ID,
) -> None:
    """Execute the end-to-end optimization pipeline for a single 24-hour day."""
    start_ts = f"{date_str} 00:00:00"
    weather_override = {"air_temperature": heatwave_delta} if heatwave_delta != 0.0 else None

    print("\n" + "=" * 88)
    print(f"PEAKWISE AI ADVISOR DEMO - FACILITY {building_id} | DATE: {date_str}")
    if heatwave_delta != 0.0:
        print(f"LIVE WEATHER OVERRIDE: +{heatwave_delta:.1f} deg C Heatwave Simulation")
    print("=" * 88)

    # 1. 24h Probabilistic Forecast
    print("\n[STEP 1/6] Generating 24-Hour Ahead Probabilistic Forecast (LightGBM)...")
    forecast_df = predict(
        building_id=building_id,
        start_ts=start_ts,
        weather_override=weather_override,
    )
    print(f"  * P50 Expected Load:  Mean={forecast_df['p50'].mean():.2f} kW, Peak={forecast_df['p50'].max():.2f} kW")
    print(f"  * P90 Risk Ceiling:   Mean={forecast_df['p90'].mean():.2f} kW, Peak={forecast_df['p90'].max():.2f} kW")

    # 2. Peak Detection & Risk Quantification
    print(f"\n[STEP 2/6] Detecting Peak-Cap Breaches (Threshold = {cap_kw:.2f} kW)...")
    peak_df, peak_summary = detect(forecast_df, cap=cap_kw)
    print(f"  * Peak P90 Demand:    {peak_summary['daily_peak_p90_kw']:.2f} kW at hour {peak_summary['peak_hour']:02d}:00")
    print(f"  * Risk Level:         {peak_summary['overall_risk_level']}")
    print(f"  * Breach Hours:       {peak_summary['breach_hours_count']} of 24 hours")
    print(f"  * Max Overshoot:      {peak_summary['max_overshoot_kw']:.2f} kW")

    # 3. Load Disaggregation
    print("\n[STEP 3/6] Disaggregating Load (Shiftable vs. Non-Shiftable)...")
    loads_df = split(forecast_df, primary_use="Education")
    print(f"  * Shiftable Load:     Mean={loads_df['shiftable_p90'].mean():.2f} kW ({loads_df['shiftable_share'].mean()*100:.1f}% avg)")
    print(f"  * Fixed Base Load:    Mean={loads_df['fixed_p90'].mean():.2f} kW")

    # 4. Optimization
    print(f"\n[STEP 4/6] Solving Linear Program Schedule (OR-Tools GLOP against {RISK_LEVEL})...")
    schedule_df, moves_df, opt_meta = optimize_schedule(
        loads_df=loads_df,
        cap=cap_kw,
        risk_level=RISK_LEVEL,
    )

    print(f"  * Solver Status:      {opt_meta['status']}")
    print(f"  * Baseline Peak:      {opt_meta['baseline_peak_kw']:.2f} kW")
    print(f"  * Scheduled Peak:     {opt_meta['scheduled_peak_kw']:.2f} kW (P*={opt_meta['min_achievable_peak_kw']:.2f} kW)")
    print(f"  * Peak Reduction:     {opt_meta.get('peak_reduction_kw', 0.0):.2f} kW")
    print(f"  * Total Shift Moves:  {opt_meta['total_moves_count']}")

    if not opt_meta["is_feasible"]:
        print(f"  [WARNING] INFEASIBILITY DIAGNOSTIC: {opt_meta['reason']}")

    # Print Moves Table if moves occurred
    if not moves_df.empty:
        print("\n  RECOMMENDED LOAD SHIFTS:")
        print("  " + "-" * 56)
        print(f"  {'FROM HOUR':<12} | {'TO HOUR':<10} | {'ORIGINAL kW':<13} | {'SERVED kW'}")
        print("  " + "-" * 56)
        for _, m in moves_df.iterrows():
            print(f"  {int(m['from_hour']):02d}:00        | {int(m['to_hour']):02d}:00     | {m['moved_kw']:<13.2f} | {m['served_kw']:.2f}")
        print("  " + "-" * 56)
    else:
        print("  * Action Required:    None (baseline load operates safely below cap).")

    # 5. Independent Validation
    print("\n[STEP 5/6] Independently Auditing Schedule Constraints...")
    val_report = validate_schedule(
        schedule_df=schedule_df,
        moves_df=moves_df,
        cap=cap_kw,
        is_infeasible=not opt_meta["is_feasible"],
        min_achievable_peak=opt_meta["min_achievable_peak_kw"],
    )

    print("  " + "-" * 76)
    print(f"  {'CONSTRAINT':<30} | {'STATUS':<8} | {'MAX VIOLATION':<14} | {'DETAILS'}")
    print("  " + "-" * 76)
    for c_name, c_res in val_report["constraints"].items():
        st = "PASS" if c_res["passed"] else "FAIL"
        print(f"  {c_name:<30} | {st:<8} | {c_res['max_violation']:<14.2e} | {c_res['details']}")
    print("  " + "-" * 76)
    print(f"  * Overall Audit:      {'[PASSED ALL CONSTRAINTS]' if val_report['all_passed'] else '[FAILED]'}")

    # 6. Economics & Financial Savings
    print("\n[STEP 6/6] Quantifying Economic Savings & Tariff Impact...")
    econ = evaluate_economics(schedule_df=schedule_df, moves_df=moves_df, cap=cap_kw)
    d = econ["daily_economics"]
    p = econ["peaks"]
    a = econ["annualized_extrapolation"]

    print("  " + "-" * 60)
    print(f"  {'METRIC':<36} | {'BEFORE':<10} | {'AFTER':<10}")
    print("  " + "-" * 60)
    print(f"  {'Daily Peak Power (kW)':<36} | {p['baseline_peak_kw']:<10.2f} | {p['scheduled_peak_kw']:<10.2f}")
    print(f"  {'Prorated Demand Charge ($/day)':<36} | ${d['demand_charge_before_usd']:<9.2f} | ${d['demand_charge_after_usd']:<9.2f}")
    print(f"  {'Peak Breach Penalty ($/day)':<36} | ${d['breach_penalty_before_usd']:<9.2f} | ${d['breach_penalty_after_usd']:<9.2f}")
    print(f"  {'TOU Energy Expenditure ($/day)':<36} | ${d['tou_energy_cost_before_usd']:<9.2f} | ${d['tou_energy_cost_after_usd']:<9.2f}")
    print("  " + "-" * 60)
    print(f"  * Demand Charge Savings:   ${d['demand_charge_savings_usd']:.2f} / day")
    print(f"  * Breach Penalties Avoided: ${d['breach_penalty_avoided_usd']:.2f} / day")
    print(f"  * Rebound Energy Overhead:  ${d['rebound_overhead_cost_usd']:.2f} / day ({d['rebound_overhead_kwh']:.2f} kWh)")
    print(f"  * NET DAILY SAVINGS:        ${d['net_daily_savings_usd']:.2f} / day")
    print(f"  * Extrapolated Annual:     ${a['annual_net_savings_usd']:.2f} / year ({a['extrapolation_basis']})")
    print("=" * 88 + "\n")


def run_backtest_evaluation(
    cap_kw: float = PEAK_CAP_KW,
    building_id: int = BUILDING_ID,
) -> None:
    """Execute rolling backtest across the 28-day test period on realized actuals."""
    print("\n" + "=" * 105)
    print(f"PEAKWISE AI - ROLLING-ORIGIN BACKTEST ON REALIZED ACTUALS (FACILITY {building_id})")
    print(f"Horizon: 28 Days (Test Set: 2016-12-04 to 2016-12-31) | Cap: {cap_kw:.2f} kW")
    print("Methodology: Schedule planned strictly from FORECAST, then applied to REALIZED ACTUAL load.")
    print("=" * 105)

    clean_df = load_clean_data(building_id=building_id)
    test_start = pd.to_datetime("2016-12-04 00:00:00")
    test_df = clean_df[clean_df[TIMESTAMP_COL] >= test_start].copy()
    test_dates = sorted(test_df[TIMESTAMP_COL].dt.date.unique())

    records: List[Dict[str, Any]] = []

    for d in test_dates:
        date_str = str(d)
        start_ts = f"{date_str} 00:00:00"

        # 1. Forecast from model
        forecast_df = predict(building_id=building_id, start_ts=start_ts)

        # 2. Extract realized actual loads for this day
        mask = test_df[TIMESTAMP_COL].dt.date == d
        day_actuals = test_df.loc[mask, "meter_reading"].values.astype(float)
        realized_base_peak = float(np.max(day_actuals))

        # 3. Disaggregate and optimize based on forecast
        loads_df = split(forecast_df, primary_use="Education")
        schedule_df, moves_df, opt_meta = optimize_schedule(loads_df, cap=cap_kw)

        # 4. Apply planned moves to REALIZED actuals
        # Reconstruct actual shift schedule
        realized_sched = day_actuals.copy()
        if not moves_df.empty:
            for _, m in moves_df.iterrows():
                h = int(m["from_hour"])
                t = int(m["to_hour"])
                moved_kw = float(m["moved_kw"])
                served_kw = float(m["served_kw"])
                realized_sched[h] -= moved_kw
                realized_sched[t] += served_kw

        realized_sched_peak = float(np.max(realized_sched))
        realized_reduction = max(0.0, realized_base_peak - realized_sched_peak)

        had_actual_breach = realized_base_peak > cap_kw
        breach_avoided = had_actual_breach and (realized_sched_peak <= cap_kw)

        # Economic impact on realized actuals
        econ = evaluate_economics(
            pd.DataFrame({"baseline_total_kw": day_actuals, "scheduled_total_kw": realized_sched}),
            moves_df,
            cap=cap_kw,
        )
        daily_savings = econ["daily_economics"]["net_daily_savings_usd"]

        records.append(
            {
                "date": date_str,
                "forecast_p90_peak": round(float(forecast_df["p90"].max()), 2),
                "actual_base_peak": round(realized_base_peak, 2),
                "actual_sched_peak": round(realized_sched_peak, 2),
                "peak_reduction": round(realized_reduction, 2),
                "had_actual_breach": had_actual_breach,
                "breach_avoided": breach_avoided,
                "moves_count": opt_meta["total_moves_count"],
                "daily_savings": round(daily_savings, 2),
                "status": opt_meta["status"],
            }
        )

    # Format Output Table
    print(f"\n{'DATE':<12} | {'FCAST P90':<10} | {'ACTUAL BASE':<12} | {'ACTUAL SCHED':<13} | {'REDUCTION':<10} | {'BREACH AVOIDED?':<16} | {'SAVINGS ($)'}")
    print("-" * 105)

    total_actual_breaches = sum(1 for r in records if r["had_actual_breach"])
    total_breaches_avoided = sum(1 for r in records if r["breach_avoided"])
    total_savings = sum(r["daily_savings"] for r in records)

    for r in records:
        if r["had_actual_breach"]:
            avoided_str = "YES [AVOIDED]" if r["breach_avoided"] else "NO (PARTIAL)"
        else:
            avoided_str = "N/A (SAFE)"

        print(
            f"{r['date']:<12} | "
            f"{r['forecast_p90_peak']:<10.2f} | "
            f"{r['actual_base_peak']:<12.2f} | "
            f"{r['actual_sched_peak']:<13.2f} | "
            f"{r['peak_reduction']:<10.2f} | "
            f"{avoided_str:<16} | "
            f"${r['daily_savings']:<8.2f}"
        )

    print("-" * 105)
    print("\nBACKTEST SUMMARY ACROSS 28 TEST DAYS:")
    print(f"  * Days Evaluated:                 {len(records)}")
    print(f"  * Days with Realized Breaches:    {total_actual_breaches}")
    print(f"  * Breaches Completely Avoided:   {total_breaches_avoided} of {total_actual_breaches} ({total_breaches_avoided/max(1, total_actual_breaches)*100:.1f}%)")
    print(f"  * Total Realized Net Savings:     ${total_savings:.2f}")
    print(f"  * Average Daily Realized Savings: ${total_savings / len(records):.2f} / day")
    print(f"  * Annualized Realized Projection: ${total_savings / len(records) * 365.0:.2f} / year")
    print("=" * 105 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="PeakWise AI Advisor CLI Demonstration")
    parser.add_argument("--date", type=str, default="2016-12-15", help="Target date (YYYY-MM-DD)")
    parser.add_argument("--heatwave", type=float, default=0.0, help="Additive heatwave temperature delta in \u00b0C")
    parser.add_argument("--cap", type=float, default=PEAK_CAP_KW, help="Peak cap threshold in kW")
    parser.add_argument("--building_id", type=int, default=BUILDING_ID, help="Target facility building ID")
    parser.add_argument("--backtest", action="store_true", help="Execute rolling backtest across test period")

    args = parser.parse_args()

    if args.backtest:
        run_backtest_evaluation(cap_kw=args.cap, building_id=args.building_id)
    else:
        run_single_day_demo(
            date_str=args.date,
            heatwave_delta=args.heatwave,
            cap_kw=args.cap,
            building_id=args.building_id,
        )


if __name__ == "__main__":
    main()
