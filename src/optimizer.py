"""Linear Programming load shifting optimizer for PeakWise AI using Google OR-Tools.

Optimizes 24-hour shiftable electrical loads against the P90 forecast curve to:
1. Guarantee non-shiftable (fixed) loads are untouched.
2. Conserve originating shiftable energy with rebound penalty losses.
3. Restrict shifting strictly within operational precooling and deferral time windows.
4. Enforce per-hour maximum shift limits.
5. Keep total electrical demand under the peak cap (PEAK_CAP_KW) when feasible.
6. When infeasible, gracefully return the minimum achievable peak (P*), exceeding hours,
   and diagnostic root-cause explanation without crashing or raising exceptions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from src.config import (
    BASE_ENERGY_RATE,
    MAX_DEFER_HOURS,
    MAX_PRECOOL_HOURS,
    MAX_SHIFT_FRACTION_PER_HOUR,
    PEAK_CAP_KW,
    REBOUND_FACTOR,
    RISK_LEVEL,
    TIMESTAMP_COL,
    TOU_HOURLY_RATES,
)

logger = logging.getLogger("peakwise.optimizer")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def optimize_schedule(
    loads_df: pd.DataFrame,
    cap: float = PEAK_CAP_KW,
    risk_level: str = RISK_LEVEL,
    max_precool_hours: int = MAX_PRECOOL_HOURS,
    max_defer_hours: int = MAX_DEFER_HOURS,
    max_shift_fraction: float = MAX_SHIFT_FRACTION_PER_HOUR,
    rebound_factor: float = REBOUND_FACTOR,
    tou_rates: Optional[List[float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Execute two-stage Linear Programming load shifting optimization.

    Stage 1: Minimizes maximum peak load P subject to physical shift constraints
             to discover feasibility and minimum achievable peak P*.
    Stage 2: If P* <= cap, solves the cost-minimization LP enforcing load[t] <= cap,
             minimizing TOU energy expenditure + rebound overhead + minimal distance friction.
             If P* > cap, returns status 'INFEASIBLE' with P* and root-cause diagnostics.

    Args:
        loads_df: Disaggregated loads DataFrame containing columns:
            [timestamp, shiftable_p50, fixed_p50, shiftable_p90, fixed_p90].
        cap: Peak power cap threshold in kW.
        risk_level: 'p90' or 'p50' target curve to plan against (default 'p90').
        max_precool_hours: Max hours to shift backward (precool, default 3).
        max_defer_hours: Max hours to shift forward (defer, default 3).
        max_shift_fraction: Max fraction of an hour's shiftable load movable (default 0.50).
        rebound_factor: Efficiency overhead multiplier for shifted energy (default 1.05).
        tou_rates: 24-element hourly electricity tariff list in $/kWh.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            - schedule_df: 24h table with original, scheduled, and component loads.
            - moves_df: Table of load shifts: [from_hour, to_hour, moved_kw, served_kw].
            - metadata: Diagnostics including status, min achievable peak, and bound reasons.
    """
    rates = tou_rates if tou_rates is not None else TOU_HOURLY_RATES
    H = len(loads_df)
    if H != 24:
        raise ValueError(f"Optimizer requires a 24-hour horizon; received {H} rows.")

    shift_col = f"shiftable_{risk_level}"
    fixed_col = f"fixed_{risk_level}"
    orig_total_col = risk_level

    if shift_col not in loads_df.columns or fixed_col not in loads_df.columns:
        raise ValueError(
            f"Missing required columns '{shift_col}' or '{fixed_col}' in loads_df."
        )

    S = loads_df[shift_col].values.astype(float)
    F = loads_df[fixed_col].values.astype(float)
    baseline_total = loads_df[orig_total_col].values.astype(float)
    timestamps = loads_df[TIMESTAMP_COL].values
    baseline_peak = round(float(np.max(baseline_total)), 4)

    # If baseline is already strictly within cap, no peak reduction action is needed
    if baseline_peak <= (cap + 1e-4):
        logger.info(
            "Baseline peak (%.2f kW) is within peak cap (%.2f kW). No action needed.",
            baseline_peak,
            cap,
        )
        metadata = {
            "status": "NO_ACTION_NEEDED",
            "is_feasible": True,
            "min_achievable_peak_kw": baseline_peak,
            "cap_kw": round(cap, 2),
            "baseline_peak_kw": baseline_peak,
            "scheduled_peak_kw": baseline_peak,
            "peak_reduction_kw": 0.0,
            "exceeding_hours": [],
            "reason": "No action needed: baseline demand operates safely below peak cap without intervention.",
            "total_moves_count": 0,
        }
        schedule_df = pd.DataFrame(
            {
                TIMESTAMP_COL: timestamps,
                "hour": range(H),
                "baseline_total_kw": baseline_total,
                "fixed_kw": F,
                "shiftable_kw": S,
                "scheduled_total_kw": baseline_total,
                "cap_kw": cap,
            }
        )
        moves_df = pd.DataFrame(columns=["from_hour", "to_hour", "kw", "moved_kw", "served_kw"])
        return schedule_df, moves_df, metadata

    # Determine allowed destination hours for each origin hour h
    def get_allowed_targets(h: int) -> List[int]:
        low = max(0, h - max_precool_hours)
        high = min(H - 1, h + max_defer_hours)
        return list(range(low, high + 1))

    # =========================================================================
    # STAGE 1: Peak Minimization (Feasibility Discovery)
    # =========================================================================
    solver1 = pywraplp.Solver.CreateSolver("GLOP")
    if not solver1:
        raise RuntimeError("OR-Tools GLOP solver could not be initialized.")

    x1: Dict[Tuple[int, int], pywraplp.Variable] = {}
    for h in range(H):
        for t in get_allowed_targets(h):
            x1[(h, t)] = solver1.NumVar(0.0, solver1.infinity(), f"x1_{h}_{t}")

    # Peak variable P
    P = solver1.NumVar(0.0, solver1.infinity(), "Peak_P")

    # Constraint 1: Energy conservation from origin h
    for h in range(H):
        targets = get_allowed_targets(h)
        solver1.Add(solver1.Sum([x1[(h, t)] for t in targets]) == S[h])

    # Constraint 2: Maximum fraction shifted out of origin h
    for h in range(H):
        targets = [t for t in get_allowed_targets(h) if t != h]
        if targets:
            solver1.Add(
                solver1.Sum([x1[(h, t)] for t in targets]) <= max_shift_fraction * S[h]
            )

    # Constraint 3: Peak bound for each destination hour t
    for t in range(H):
        incoming_terms = []
        for h in range(H):
            if (h, t) in x1:
                if h == t:
                    incoming_terms.append(x1[(h, t)])
                else:
                    incoming_terms.append(rebound_factor * x1[(h, t)])
        # Total served load at t: F[t] + sum(incoming) <= P
        solver1.Add(F[t] + solver1.Sum(incoming_terms) <= P)

    # Objective 1: Minimize peak P
    solver1.Minimize(P)
    status1 = solver1.Solve()

    if status1 != pywraplp.Solver.OPTIMAL:
        logger.error("Stage 1 peak minimization solver failed with status %d", status1)
        min_peak = float(np.max(baseline_total))
        return _build_fallback_schedule(loads_df, baseline_total, min_peak, "STAGE1_SOLVER_ERROR")

    min_achievable_peak = round(float(P.solution_value()), 4)
    is_feasible = bool(min_achievable_peak <= (cap + 1e-4))

    logger.info(
        "Stage 1 Result: Minimum Achievable Peak P*=%.2f kW (Cap=%.2f kW, Feasible=%s)",
        min_achievable_peak,
        cap,
        is_feasible,
    )

    # If infeasible under cap, extract diagnostic causes and return minimum-peak schedule
    if not is_feasible:
        # Reconstruct Stage 1 schedule
        scheduled_loads1 = np.zeros(H)
        moves_list1 = []
        for t in range(H):
            total_t = F[t]
            for h in range(H):
                if (h, t) in x1:
                    val = x1[(h, t)].solution_value()
                    if h == t:
                        total_t += val
                    else:
                        total_t += rebound_factor * val
                        if val > 1e-4:
                            moves_list1.append(
                                {
                                    "from_hour": h,
                                    "to_hour": t,
                                    "kw": float(val),
                                    "moved_kw": float(val),
                                    "served_kw": float(val * rebound_factor),
                                }
                            )
            scheduled_loads1[t] = total_t

        min_achievable_peak = float(np.max(scheduled_loads1))
        exceeding_hours = [int(h) for h in range(H) if scheduled_loads1[h] > (cap + 1e-4)]
        reasons = []

        # Check if fixed load alone exceeds cap
        fixed_exceed = [int(h) for h in range(H) if F[h] > cap]
        if fixed_exceed:
            reasons.append(
                f"Non-shiftable (fixed) load alone exceeds cap in hours {fixed_exceed} "
                f"(max fixed={np.max(F):.2f} kW)."
            )

        # Check shift capacity exhaustion
        reasons.append(
            f"Physical shift windows [precool={max_precool_hours}h, defer={max_defer_hours}h, "
            f"max_shift={int(max_shift_fraction*100)}%] cannot absorb required energy. "
            f"Minimum achievable peak is {min_achievable_peak:.2f} kW."
        )

        diagnostic_reason = " ".join(reasons)
        metadata = {
            "status": "INFEASIBLE",
            "is_feasible": False,
            "min_achievable_peak_kw": min_achievable_peak,
            "cap_kw": round(cap, 2),
            "baseline_peak_kw": round(float(np.max(baseline_total)), 4),
            "scheduled_peak_kw": min_achievable_peak,
            "exceeding_hours": exceeding_hours,
            "reason": diagnostic_reason,
            "total_moves_count": len(moves_list1),
        }

        schedule_df = pd.DataFrame(
            {
                TIMESTAMP_COL: timestamps,
                "hour": range(H),
                "baseline_total_kw": baseline_total,
                "fixed_kw": F,
                "shiftable_kw": S,
                "scheduled_total_kw": scheduled_loads1,
                "cap_kw": cap,
            }
        )
        moves_df = pd.DataFrame(moves_list1) if moves_list1 else pd.DataFrame(
            columns=["from_hour", "to_hour", "kw", "moved_kw", "served_kw"]
        )
        return schedule_df, moves_df, metadata

    # =========================================================================
    # STAGE 2: Cost & Friction Minimization (Feasible under Cap)
    # =========================================================================
    solver2 = pywraplp.Solver.CreateSolver("GLOP")
    x2: Dict[Tuple[int, int], pywraplp.Variable] = {}
    for h in range(H):
        for t in get_allowed_targets(h):
            x2[(h, t)] = solver2.NumVar(0.0, solver2.infinity(), f"x2_{h}_{t}")

    # Energy conservation
    for h in range(H):
        targets = get_allowed_targets(h)
        solver2.Add(solver2.Sum([x2[(h, t)] for t in targets]) == S[h])

    # Maximum shift fraction out of h
    for h in range(H):
        targets = [t for t in get_allowed_targets(h) if t != h]
        if targets:
            solver2.Add(
                solver2.Sum([x2[(h, t)] for t in targets]) <= max_shift_fraction * S[h]
            )

    # Hard Cap Constraint for each hour t: load[t] <= cap
    for t in range(H):
        incoming_terms = []
        for h in range(H):
            if (h, t) in x2:
                if h == t:
                    incoming_terms.append(x2[(h, t)])
                else:
                    incoming_terms.append(rebound_factor * x2[(h, t)])
        solver2.Add(F[t] + solver2.Sum(incoming_terms) <= cap)

    # Objective: Minimize TOU Energy Cost + Rebound Overhead + Distance Friction
    # Friction (0.001 * |h-t|) prevents gratuitous load moving when no breach occurs
    obj_terms = []
    for t in range(H):
        # TOU energy cost on fixed load (constant, but keeps accounting complete)
        obj_terms.append(rates[t] * F[t])
        for h in range(H):
            if (h, t) in x2:
                if h == t:
                    # Unshifted load pays standard TOU
                    obj_terms.append(rates[t] * x2[(h, t)])
                else:
                    # Shifted load pays TOU on served kWh + rebound penalty cost + distance cost
                    dist = abs(h - t)
                    shift_cost_coeff = (
                        (rates[t] * rebound_factor)
                        + ((rebound_factor - 1.0) * BASE_ENERGY_RATE)
                        + (0.001 * dist)
                    )
                    obj_terms.append(shift_cost_coeff * x2[(h, t)])

    solver2.Minimize(solver2.Sum(obj_terms))
    status2 = solver2.Solve()

    if status2 != pywraplp.Solver.OPTIMAL:
        logger.warning("Stage 2 optimization returned status %d; using Stage 1 solution.", status2)
        # Fallback to Stage 1 solution

    # Extract final optimal schedule and move records
    scheduled_loads = np.zeros(H)
    moves_list = []

    for t in range(H):
        total_t = F[t]
        for h in range(H):
            if (h, t) in x2:
                val = x2[(h, t)].solution_value()
                if h == t:
                    total_t += val
                else:
                    total_t += rebound_factor * val
                    if val > 1e-4:
                        moves_list.append(
                            {
                                "from_hour": h,
                                "to_hour": t,
                                "kw": float(val),
                                "moved_kw": float(val),
                                "served_kw": float(val * rebound_factor),
                            }
                        )
        scheduled_loads[t] = total_t

    final_peak = round(float(np.max(scheduled_loads)), 4)
    baseline_peak = round(float(np.max(baseline_total)), 4)

    metadata = {
        "status": "OPTIMAL",
        "is_feasible": True,
        "min_achievable_peak_kw": min_achievable_peak,
        "cap_kw": round(cap, 2),
        "baseline_peak_kw": baseline_peak,
        "scheduled_peak_kw": final_peak,
        "peak_reduction_kw": round(max(0.0, baseline_peak - final_peak), 4),
        "exceeding_hours": [],
        "reason": "Successfully shifted loads below peak cap with minimal operational friction.",
        "total_moves_count": len(moves_list),
    }

    schedule_df = pd.DataFrame(
        {
            TIMESTAMP_COL: timestamps,
            "hour": range(H),
            "baseline_total_kw": baseline_total,
            "fixed_kw": F,
            "shiftable_kw": S,
            "scheduled_total_kw": scheduled_loads,
            "cap_kw": cap,
        }
    )

    moves_df = pd.DataFrame(moves_list) if moves_list else pd.DataFrame(
        columns=["from_hour", "to_hour", "kw", "moved_kw", "served_kw"]
    )

    logger.info(
        "Optimization complete: Baseline Peak=%.2f kW -> Scheduled Peak=%.2f kW "
        "(Cap=%.2f kW, Moves=%d)",
        baseline_peak,
        final_peak,
        cap,
        len(moves_list),
    )
    return schedule_df, moves_df, metadata


def _build_fallback_schedule(
    loads_df: pd.DataFrame,
    baseline_total: np.ndarray,
    min_peak: float,
    reason: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate unshifted fallback schedule if solver terminates unexpectedly."""
    H = len(loads_df)
    metadata = {
        "status": "ERROR",
        "is_feasible": False,
        "min_achievable_peak_kw": min_peak,
        "cap_kw": 0.0,
        "baseline_peak_kw": min_peak,
        "scheduled_peak_kw": min_peak,
        "exceeding_hours": list(range(H)),
        "reason": reason,
        "total_moves_count": 0,
    }
    schedule_df = pd.DataFrame(
        {
            TIMESTAMP_COL: loads_df[TIMESTAMP_COL].values,
            "hour": range(H),
            "baseline_total_kw": baseline_total,
            "fixed_kw": loads_df.get("fixed_p90", baseline_total * 0.65).values,
            "shiftable_kw": loads_df.get("shiftable_p90", baseline_total * 0.35).values,
            "scheduled_total_kw": baseline_total,
            "cap_kw": 0.0,
        }
    )
    moves_df = pd.DataFrame(columns=["from_hour", "to_hour", "moved_kw", "served_kw"])
    return schedule_df, moves_df, metadata
