"""Peak detection and breach risk quantification module for PeakWise AI.

Analyzes 24-hour ahead probabilistic forecasts (P10, P50, P90) against the peak
power threshold (PEAK_CAP_KW) to identify potential breaches, estimate continuous
breach probabilities via quantile interpolation, classify operational risk, and
compute expected load overshoots.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.config import PEAK_CAP_KW, TIMESTAMP_COL

logger = logging.getLogger("peakwise.peak")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def estimate_breach_probability(
    cap: float,
    p10: float,
    p50: float,
    p90: float,
) -> float:
    """Estimate the empirical probability P(load > cap) by interpolating quantiles.

    Uses a piecewise linear CDF constructed from the predicted quantiles:
      (0.10 -> p10), (0.50 -> p50), (0.90 -> p90).

    Args:
        cap: Target demand threshold in kW.
        p10: 10th percentile load forecast in kW.
        p50: 50th percentile (median) load forecast in kW.
        p90: 90th percentile load forecast in kW.

    Returns:
        float: Estimated breach probability in [0.0, 1.0].
    """
    p10_safe = max(0.0, float(p10))
    p50_safe = max(p10_safe, float(p50))
    p90_safe = max(p50_safe, float(p90))

    if cap <= p10_safe:
        if p10_safe > 0:
            cdf_val = 0.10 * max(0.0, cap / p10_safe)
        else:
            cdf_val = 0.0
    elif cap <= p50_safe:
        denom = max(1e-4, p50_safe - p10_safe)
        cdf_val = 0.10 + 0.40 * ((cap - p10_safe) / denom)
    elif cap <= p90_safe:
        denom = max(1e-4, p90_safe - p50_safe)
        cdf_val = 0.50 + 0.40 * ((cap - p50_safe) / denom)
    else:
        # cap > p90
        spread = max(1.0, p90_safe - p50_safe)
        overshoot_ratio = (cap - p90_safe) / spread
        cdf_val = 0.90 + 0.095 * min(1.0, overshoot_ratio)

    cdf_val = float(np.clip(cdf_val, 0.001, 0.999))
    prob_breach = 1.0 - cdf_val
    return round(prob_breach, 4)


def classify_risk_level(breach_prob: float, overshoot_kw: float) -> str:
    """Classify hourly operational breach risk into LOW, MED, or HIGH.

    Args:
        breach_prob: Estimated breach probability.
        overshoot_kw: Expected load overshoot in kW above cap.

    Returns:
        str: 'LOW', 'MED', or 'HIGH'.
    """
    if breach_prob >= 0.50 or overshoot_kw > 5.0:
        return "HIGH"
    elif breach_prob >= 0.15 or overshoot_kw > 0.0:
        return "MED"
    return "LOW"


def detect(
    forecast_df: pd.DataFrame,
    cap: float = PEAK_CAP_KW,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Detect peak breaches, compute breach probabilities, and summarize daily risk.

    Args:
        forecast_df: 24-hour forecast DataFrame with columns [timestamp, p10, p50, p90].
        cap: Peak power cap threshold in kW (default from src.config).

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - Hourly analysis DataFrame containing:
                ['timestamp', 'p50', 'p90', 'cap', 'breach', 'breach_probability',
                 'risk_level', 'expected_overshoot_kw']
            - Daily summary dictionary containing overall metrics, peak hour, and status.
    """
    df = forecast_df.copy()
    if not {TIMESTAMP_COL, "p10", "p50", "p90"}.issubset(df.columns):
        raise ValueError(
            f"forecast_df must contain ['{TIMESTAMP_COL}', 'p10', 'p50', 'p90']. "
            f"Found: {list(df.columns)}"
        )

    df["cap"] = round(cap, 2)
    df["breach"] = df["p90"] > cap
    df["expected_overshoot_kw"] = np.maximum(0.0, df["p90"] - cap).round(4)

    probs = [
        estimate_breach_probability(cap, row["p10"], row["p50"], row["p90"])
        for _, row in df.iterrows()
    ]
    df["breach_probability"] = probs

    risks = [
        classify_risk_level(prob, over)
        for prob, over in zip(df["breach_probability"], df["expected_overshoot_kw"])
    ]
    df["risk_level"] = risks

    # Extract daily peak diagnostics
    p90_max_idx = int(df["p90"].argmax())
    p90_peak = float(df["p90"].iloc[p90_max_idx])
    p50_peak = float(df["p50"].max())
    peak_timestamp = df[TIMESTAMP_COL].iloc[p90_max_idx]
    peak_hour = int(pd.to_datetime(peak_timestamp).hour)

    breach_hours_count = int(df["breach"].sum())
    max_overshoot_kw = float(df["expected_overshoot_kw"].max())

    if max_overshoot_kw > 5.0 or (p50_peak > cap):
        overall_risk = "HIGH"
    elif breach_hours_count > 0:
        overall_risk = "MED"
    else:
        overall_risk = "LOW"

    summary: Dict[str, Any] = {
        "cap_kw": round(cap, 2),
        "daily_peak_p90_kw": round(p90_peak, 4),
        "daily_peak_p50_kw": round(p50_peak, 4),
        "peak_hour": peak_hour,
        "peak_timestamp": str(peak_timestamp),
        "has_breach": breach_hours_count > 0,
        "breach_hours_count": breach_hours_count,
        "max_overshoot_kw": round(max_overshoot_kw, 4),
        "overall_risk_level": overall_risk,
    }

    logger.info(
        "Peak detection complete: Peak P90=%.2f kW at hour %02d:00, "
        "Breach hours=%d, Max overshoot=%.2f kW, Risk=%s",
        p90_peak,
        peak_hour,
        breach_hours_count,
        max_overshoot_kw,
        overall_risk,
    )

    out_cols = [
        TIMESTAMP_COL,
        "p50",
        "p90",
        "cap",
        "breach",
        "breach_probability",
        "risk_level",
        "expected_overshoot_kw",
    ]
    return df[out_cols], summary
