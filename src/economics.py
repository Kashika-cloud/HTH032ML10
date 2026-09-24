"""Economic impact and financial savings quantification module for PeakWise AI.

Evaluates demand charge savings, avoided peak breach penalties, Time-of-Use (TOU)
energy cost differences, and rebound thermodynamic overhead to calculate net daily
savings and annualized financial projections.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    BASE_ENERGY_RATE,
    BREACH_PENALTY_PER_KW,
    DAYS_PER_MONTH,
    DAYS_PER_YEAR,
    DEMAND_CHARGE_PER_KW,
    PEAK_CAP_KW,
    TOU_HOURLY_RATES,
)

logger = logging.getLogger("peakwise.economics")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def evaluate_economics(
    schedule_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    cap: float = PEAK_CAP_KW,
    demand_charge_per_kw: float = DEMAND_CHARGE_PER_KW,
    breach_penalty_per_kw: float = BREACH_PENALTY_PER_KW,
    tou_rates: Optional[List[float]] = None,
    base_energy_rate: float = BASE_ENERGY_RATE,
    days_per_month: float = DAYS_PER_MONTH,
    days_per_year: float = DAYS_PER_YEAR,
) -> Dict[str, Any]:
    """Calculate before vs after economic impact and cost savings.

    Args:
        schedule_df: 24h schedule DataFrame with [baseline_total_kw, scheduled_total_kw].
        moves_df: Table of load shifts with [moved_kw, served_kw].
        cap: Peak power cap threshold in kW.
        demand_charge_per_kw: Monthly demand charge tariff ($/kW-month).
        breach_penalty_per_kw: Penalty rate for exceeding peak cap ($/kW).
        tou_rates: Hourly electricity rates ($/kWh).
        base_energy_rate: Baseline volumetric rate ($/kWh).
        days_per_month: Days per billing month for prorating demand charge (default 30).
        days_per_year: Days per year for annual extrapolation (default 365).

    Returns:
        Dict[str, Any]: Detailed breakdown of demand charges, penalties, energy costs,
            net daily savings, and annualized extrapolated estimates.
    """
    rates = tou_rates if tou_rates is not None else TOU_HOURLY_RATES
    y_base = schedule_df["baseline_total_kw"].values.astype(float)
    y_sched = schedule_df["scheduled_total_kw"].values.astype(float)

    # 1. Peak Demand Metrics
    peak_base = float(np.max(y_base))
    peak_sched = float(np.max(y_sched))
    peak_reduction = max(0.0, peak_base - peak_sched)

    # 2. Demand Charges (Daily Prorated from Monthly Billing Cycle)
    daily_demand_rate = demand_charge_per_kw / days_per_month
    demand_charge_base = peak_base * daily_demand_rate
    demand_charge_sched = peak_sched * daily_demand_rate
    demand_charge_savings = peak_reduction * daily_demand_rate

    # 3. Peak Cap Breach Penalties
    overshoot_base = max(0.0, peak_base - cap)
    overshoot_sched = max(0.0, peak_sched - cap)
    breach_penalty_base = overshoot_base * breach_penalty_per_kw
    breach_penalty_sched = overshoot_sched * breach_penalty_per_kw
    breach_penalty_avoided = max(0.0, breach_penalty_base - breach_penalty_sched)

    # 4. TOU Energy Expenditure
    tou_energy_base = float(np.sum(y_base * np.array(rates)))
    tou_energy_sched = float(np.sum(y_sched * np.array(rates)))
    tou_energy_delta = tou_energy_sched - tou_energy_base

    # 5. Rebound Energy Overhead Cost
    if not moves_df.empty:
        moved_kwh = float(moves_df["moved_kw"].sum())
        served_kwh = float(moves_df["served_kw"].sum())
        rebound_kwh = max(0.0, served_kwh - moved_kwh)
    else:
        moved_kwh = 0.0
        served_kwh = 0.0
        rebound_kwh = 0.0

    rebound_cost = rebound_kwh * base_energy_rate

    # 6. Net Daily Financial Savings
    # Savings = (Demand Charge Savings + Avoided Penalties) - (Additional TOU/Rebound Energy Cost)
    gross_daily_savings = demand_charge_savings + breach_penalty_avoided
    net_daily_savings = gross_daily_savings - max(0.0, tou_energy_delta)

    # 7. Annualized Extrapolated Savings
    # Explicitly labeled as an extrapolation assuming recurring weekday operational schedule
    annualized_demand_savings = demand_charge_savings * days_per_year
    annualized_penalties_avoided = breach_penalty_avoided * days_per_year
    annualized_net_savings = net_daily_savings * days_per_year

    report = {
        "peaks": {
            "cap_kw": round(cap, 2),
            "baseline_peak_kw": round(peak_base, 2),
            "scheduled_peak_kw": round(peak_sched, 2),
            "peak_reduction_kw": round(peak_reduction, 2),
            "baseline_breach_kw": round(overshoot_base, 2),
            "scheduled_breach_kw": round(overshoot_sched, 2),
        },
        "daily_economics": {
            "demand_charge_before_usd": round(demand_charge_base, 2),
            "demand_charge_after_usd": round(demand_charge_sched, 2),
            "demand_charge_savings_usd": round(demand_charge_savings, 2),
            "breach_penalty_before_usd": round(breach_penalty_base, 2),
            "breach_penalty_after_usd": round(breach_penalty_sched, 2),
            "breach_penalty_avoided_usd": round(breach_penalty_avoided, 2),
            "tou_energy_cost_before_usd": round(tou_energy_base, 2),
            "tou_energy_cost_after_usd": round(tou_energy_sched, 2),
            "tou_energy_delta_usd": round(tou_energy_delta, 2),
            "rebound_overhead_kwh": round(rebound_kwh, 2),
            "rebound_overhead_cost_usd": round(rebound_cost, 2),
            "gross_daily_savings_usd": round(gross_daily_savings, 2),
            "net_daily_savings_usd": round(net_daily_savings, 2),
        },
        "annualized_extrapolation": {
            "extrapolation_basis": "Assumption: 365 days recurring daily pattern (extrapolated)",
            "annual_demand_charge_savings_usd": round(annualized_demand_savings, 2),
            "annual_penalties_avoided_usd": round(annualized_penalties_avoided, 2),
            "annual_net_savings_usd": round(annualized_net_savings, 2),
        },
    }

    logger.info(
        "Economics evaluated: Peak reduction=%.2f kW, Daily net savings=$%.2f "
        "(Demand savings=$%.2f, Penalty avoided=$%.2f, Energy delta=$%.2f)",
        peak_reduction,
        net_daily_savings,
        demand_charge_savings,
        breach_penalty_avoided,
        tou_energy_delta,
    )
    return report
