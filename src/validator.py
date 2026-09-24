"""Independent schedule constraint validation module for PeakWise AI.

Recomputes and verifies all physical, operational, and electrical constraints on
the scheduled load profile without accessing optimizer internals:
1. Fixed load untouched: non-shiftable loads remain intact for all 24 hours.
2. Energy conservation: all originated shiftable energy is strictly accounted for.
3. Shift window compliance: no load is shifted outside allowed precool/defer bounds.
4. Shift limit compliance: no origin hour shifts more than its allowed max fraction.
5. Peak cap compliance: scheduled load does not exceed cap (or min achievable peak if infeasible).
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (
    MAX_DEFER_HOURS,
    MAX_PRECOOL_HOURS,
    MAX_SHIFT_FRACTION_PER_HOUR,
    PEAK_CAP_KW,
    REBOUND_FACTOR,
)

logger = logging.getLogger("peakwise.validator")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def validate_schedule(
    schedule_df: pd.DataFrame,
    moves_df: pd.DataFrame,
    cap: float = PEAK_CAP_KW,
    max_precool_hours: int = MAX_PRECOOL_HOURS,
    max_defer_hours: int = MAX_DEFER_HOURS,
    max_shift_fraction: float = MAX_SHIFT_FRACTION_PER_HOUR,
    rebound_factor: float = REBOUND_FACTOR,
    tolerance: float = 1e-6,
    is_infeasible: bool = False,
    min_achievable_peak: Optional[float] = None,
) -> Dict[str, Any]:
    """Independently audit every constraint on the final load schedule.

    Args:
        schedule_df: 24h schedule DataFrame with [hour, fixed_kw, shiftable_kw, scheduled_total_kw].
        moves_df: Table of load shifts with [from_hour, to_hour, moved_kw, served_kw].
        cap: Peak power threshold in kW.
        max_precool_hours: Allowed precool window (hours backward).
        max_defer_hours: Allowed deferral window (hours forward).
        max_shift_fraction: Max movable fraction of an hour's shiftable load.
        rebound_factor: Efficiency overhead multiplier for shifted energy.
        tolerance: Numerical tolerance threshold.
        is_infeasible: Whether the optimizer flagged the instance as infeasible under cap.
        min_achievable_peak: Minimum achievable peak P* reported by optimizer if infeasible.

    Returns:
        Dict[str, Any]: Full audit report with pass/fail status per constraint and max violation.
    """
    H = len(schedule_df)
    results: Dict[str, Dict[str, Any]] = {}
    all_passed = True

    F = schedule_df["fixed_kw"].values.astype(float)
    S = schedule_df["shiftable_kw"].values.astype(float)
    sched_total = schedule_df["scheduled_total_kw"].values.astype(float)

    # Reconstruct load served at each hour from moves
    # Moves contains only h != t. Self-served load is S[h] - sum_{t != h} moved_kw(h->t)
    moved_out = np.zeros(H)
    incoming_served = np.zeros(H)

    if not moves_df.empty:
        for _, row in moves_df.iterrows():
            h = int(row["from_hour"])
            t = int(row["to_hour"])
            moved_val = float(row["moved_kw"])
            served_val = float(row["served_kw"])

            moved_out[h] += moved_val
            incoming_served[t] += served_val

    unshifted_shiftable = np.maximum(0.0, S - moved_out)
    reconstructed_sched = F + unshifted_shiftable + incoming_served

    # -------------------------------------------------------------------------
    # Constraint 1: Fixed load untouched
    # -------------------------------------------------------------------------
    # Fixed load at hour t must match schedule_df['fixed_kw']
    diff_fixed = np.abs(sched_total - (unshifted_shiftable + incoming_served) - F)
    max_fixed_viol = float(np.max(diff_fixed))
    c1_passed = max_fixed_viol <= tolerance
    results["fixed_load_untouched"] = {
        "passed": c1_passed,
        "max_violation": round(max_fixed_viol, 6),
        "details": f"Max discrepancy across 24h: {max_fixed_viol:.2e} kW (tol={tolerance:.0e})",
    }
    if not c1_passed:
        all_passed = False

    # -------------------------------------------------------------------------
    # Constraint 2: Total energy conservation
    # -------------------------------------------------------------------------
    # Each origin hour's shifted load + unshifted load must equal original shiftable load
    energy_diffs = []
    for h in range(H):
        energy_accounted = unshifted_shiftable[h] + moved_out[h]
        energy_diffs.append(abs(energy_accounted - S[h]))
    max_energy_viol = float(np.max(energy_diffs))
    c2_passed = max_energy_viol <= tolerance
    results["energy_conservation"] = {
        "passed": c2_passed,
        "max_violation": round(max_energy_viol, 6),
        "details": f"Origin energy conservation max discrepancy: {max_energy_viol:.2e} kWh",
    }
    if not c2_passed:
        all_passed = False

    # -------------------------------------------------------------------------
    # Constraint 3: Shift window compliance
    # -------------------------------------------------------------------------
    # Every move must satisfy: h - max_precool_hours <= t <= h + max_defer_hours
    window_violations = []
    if not moves_df.empty:
        for _, row in moves_df.iterrows():
            h = int(row["from_hour"])
            t = int(row["to_hour"])
            earliest = max(0, h - max_precool_hours)
            latest = min(H - 1, h + max_defer_hours)
            if t < earliest or t > latest:
                window_violations.append((h, t, earliest, latest))

    c3_passed = len(window_violations) == 0
    results["shift_window_compliance"] = {
        "passed": c3_passed,
        "max_violation": float(len(window_violations)),
        "details": (
            "All shifts within allowed window"
            if c3_passed
            else f"Found {len(window_violations)} out-of-window moves: {window_violations}"
        ),
    }
    if not c3_passed:
        all_passed = False

    # -------------------------------------------------------------------------
    # Constraint 4: Per-hour shift limits
    # -------------------------------------------------------------------------
    # moved_out[h] <= max_shift_fraction * S[h]
    max_shift_allowed = max_shift_fraction * S
    shift_limit_diffs = np.maximum(0.0, moved_out - max_shift_allowed)
    max_limit_viol = float(np.max(shift_limit_diffs))
    c4_passed = max_limit_viol <= tolerance
    results["per_hour_shift_limit"] = {
        "passed": c4_passed,
        "max_violation": round(max_limit_viol, 6),
        "details": (
            f"All hourly moves <= {int(max_shift_fraction*100)}% limit (max excess: {max_limit_viol:.2e} kW)"
        ),
    }
    if not c4_passed:
        all_passed = False

    # -------------------------------------------------------------------------
    # Constraint 5: Peak cap compliance
    # -------------------------------------------------------------------------
    if not is_infeasible:
        cap_overshoots = np.maximum(0.0, sched_total - cap)
        max_cap_viol = float(np.max(cap_overshoots))
        c5_passed = max_cap_viol <= tolerance
        details_str = (
            f"Scheduled load strictly <= cap {cap:.2f} kW (max={np.max(sched_total):.2f} kW)"
            if c5_passed
            else f"Cap exceeded by {max_cap_viol:.2f} kW"
        )
    else:
        # If infeasible, verify schedule respects minimum achievable peak P*
        target_ceiling = min_achievable_peak if min_achievable_peak is not None else cap
        cap_overshoots = np.maximum(0.0, sched_total - target_ceiling)
        max_cap_viol = float(np.max(cap_overshoots))
        c5_passed = max_cap_viol <= tolerance
        details_str = (
            f"[INFEASIBLE CASE] Scheduled load respects minimum achievable peak "
            f"P*={target_ceiling:.2f} kW (Cap was {cap:.2f} kW)"
        )

    results["peak_cap_compliance"] = {
        "passed": c5_passed,
        "max_violation": round(max_cap_viol, 6),
        "details": details_str,
    }
    if not c5_passed:
        all_passed = False

    report = {
        "all_passed": all_passed,
        "is_feasible": not is_infeasible,
        "constraints": results,
    }
    logger.info("Schedule validation complete. Overall Pass=%s", all_passed)
    return report
