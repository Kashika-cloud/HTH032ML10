"""Load disaggregation module for PeakWise AI.

EXPLICIT MODELING ASSUMPTION:
The ASHRAE Great Energy Predictor III dataset records whole-facility aggregate
electricity consumption without submetered end-use breakdown. Therefore, the
disaggregation of total power demand into shiftable and non-shiftable (fixed)
components is an engineering domain assumption.

In commercial and educational manufacturing scenarios:
- Shiftable loads (typically 20% to 45% of total demand):
  HVAC chillers, thermal energy storage, ventilation precooling/setback,
  process water pumps, and electric vehicle / forklift battery charging.
- Non-shiftable / Fixed loads (typically 55% to 80% of total demand):
  Critical production equipment, rubber molding presses, curing machinery,
  baseline safety systems, emergency lighting, and IT servers.
"""

import logging
from typing import Optional

import pandas as pd

from src.config import (
    DEFAULT_SHIFTABLE_PROFILE_EDUCATION,
    SHIFTABLE_PROFILES_BY_USE,
    TIMESTAMP_COL,
)

logger = logging.getLogger("peakwise.loads")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_shiftable_share_for_hour(
    hour: int,
    primary_use: str = "Education",
) -> float:
    """Retrieve the assumed shiftable load fraction for a given hour and building use.

    Args:
        hour: Hour of the day (0 to 23).
        primary_use: Primary building use category.

    Returns:
        float: Fraction of total load classified as shiftable (between 0.0 and 1.0).
    """
    profile = SHIFTABLE_PROFILES_BY_USE.get(primary_use, DEFAULT_SHIFTABLE_PROFILE_EDUCATION)
    h_idx = int(hour) % 24
    return profile[h_idx]


def split(
    forecast_df: pd.DataFrame,
    primary_use: str = "Education",
    custom_shiftable_share: Optional[float] = None,
) -> pd.DataFrame:
    """Disaggregate 24-hour load forecasts into shiftable and non-shiftable kW.

    Args:
        forecast_df: Forecast DataFrame containing ['timestamp', 'p10', 'p50', 'p90'].
        primary_use: Facility primary use category (default 'Education').
        custom_shiftable_share: Optional constant override for shiftable fraction.

    Returns:
        pd.DataFrame: Enriched DataFrame containing:
            ['timestamp', 'p10', 'p50', 'p90', 'shiftable_share',
             'shiftable_p50', 'fixed_p50', 'shiftable_p90', 'fixed_p90']
    """
    df = forecast_df.copy()
    if not {TIMESTAMP_COL, "p50", "p90"}.issubset(df.columns):
        raise ValueError(
            f"forecast_df must contain ['{TIMESTAMP_COL}', 'p50', 'p90']. "
            f"Found: {list(df.columns)}"
        )

    ts = pd.to_datetime(df[TIMESTAMP_COL])
    hours = ts.dt.hour

    if custom_shiftable_share is not None:
        if not (0.0 <= custom_shiftable_share <= 1.0):
            raise ValueError(
                f"custom_shiftable_share must be in [0, 1], got {custom_shiftable_share}"
            )
        shares = [custom_shiftable_share] * len(df)
    else:
        shares = [get_shiftable_share_for_hour(h, primary_use) for h in hours]

    df["shiftable_share"] = shares

    # Disaggregate P50
    df["shiftable_p50"] = (df["p50"] * df["shiftable_share"]).round(4)
    df["fixed_p50"] = (df["p50"] - df["shiftable_p50"]).round(4)

    # Disaggregate P90
    df["shiftable_p90"] = (df["p90"] * df["shiftable_share"]).round(4)
    df["fixed_p90"] = (df["p90"] - df["shiftable_p90"]).round(4)

    # Canonical columns for active risk profile (P90)
    df["shiftable_kw"] = df["shiftable_p90"]
    df["fixed_kw"] = df["fixed_p90"]

    logger.info(
        "Disaggregated loads for '%s': P90 total mean=%.2f kW, "
        "shiftable mean=%.2f kW, fixed mean=%.2f kW.",
        primary_use,
        df["p90"].mean(),
        df["shiftable_p90"].mean(),
        df["fixed_p90"].mean(),
    )
    return df
