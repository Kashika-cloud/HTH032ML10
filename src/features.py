"""Feature engineering module for 24-hour ahead energy forecasting.

CRITICAL DESIGN REQUIREMENT: ZERO DATA LEAKAGE.
For any forecast horizon of H=24 hours ahead, at inference time origin T,
the latest available target observation is at or before T (i.e. target at t-24).
Therefore:
1. Minimum lag for any meter reading feature must be >= 24 hours.
2. All rolling statistics are strictly calculated on series shifted by >= 24 hours.
3. No current or future meter readings (t, t-1, ..., t-23) are ever accessed.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    FORECAST_HORIZON,
    LAG_HOURS,
    MIN_ALLOWED_LAG,
    PEAK_CAP_PERCENTILE,
    ROLLING_WINDOWS,
    TARGET_COL,
    TIMESTAMP_COL,
    WEATHER_COLS,
)

logger = logging.getLogger("peakwise.features")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def add_calendar_features(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
) -> pd.DataFrame:
    """Extract deterministic calendar and cyclical features from timestamp.

    Features generated:
    - hour (0-23)
    - day_of_week (0-6, Monday=0)
    - month (1-12)
    - is_weekend (0 or 1)
    - hour_sin, hour_cos (cyclical 24-hour representations)
    - day_of_week_sin, day_of_week_cos (cyclical weekly representations)
    - month_sin, month_cos (cyclical annual representations)

    Args:
        df: Input DataFrame containing the timestamp column.
        timestamp_col: Name of the datetime column.

    Returns:
        pd.DataFrame: DataFrame with calendar features added.
    """
    df_out = df.copy()
    ts = pd.to_datetime(df_out[timestamp_col])

    df_out["hour"] = ts.dt.hour.astype(int)
    df_out["day_of_week"] = ts.dt.dayofweek.astype(int)
    df_out["month"] = ts.dt.month.astype(int)
    df_out["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)

    # Cyclical hour encoding (24h period)
    df_out["hour_sin"] = np.sin(2.0 * np.pi * df_out["hour"] / 24.0)
    df_out["hour_cos"] = np.cos(2.0 * np.pi * df_out["hour"] / 24.0)

    # Cyclical day of week encoding (7-day period)
    df_out["day_of_week_sin"] = np.sin(2.0 * np.pi * df_out["day_of_week"] / 7.0)
    df_out["day_of_week_cos"] = np.cos(2.0 * np.pi * df_out["day_of_week"] / 7.0)

    # Cyclical month encoding (12-month period)
    df_out["month_sin"] = np.sin(2.0 * np.pi * (df_out["month"] - 1.0) / 12.0)
    df_out["month_cos"] = np.cos(2.0 * np.pi * (df_out["month"] - 1.0) / 12.0)

    logger.debug("Added 10 calendar and cyclical features.")
    return df_out


def add_weather_features(
    df: pd.DataFrame,
    weather_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Ensure weather features and add physical thermodynamic interactions.

    Adds:
    - temp_dew_diff: Dew point depression (air_temperature - dew_temperature),
      an indicator of relative humidity directly impacting HVAC cooling load.

    Args:
        df: Input DataFrame.
        weather_cols: List of base weather columns. If None, uses WEATHER_COLS.

    Returns:
        pd.DataFrame: DataFrame with enriched weather features.
    """
    df_out = df.copy()
    base_cols = weather_cols if weather_cols is not None else WEATHER_COLS

    for col in base_cols:
        if col not in df_out.columns:
            logger.warning("Weather column '%s' not found in DataFrame.", col)

    if "air_temperature" in df_out.columns and "dew_temperature" in df_out.columns:
        df_out["temp_dew_diff"] = df_out["air_temperature"] - df_out["dew_temperature"]
        logger.debug("Added derived weather feature 'temp_dew_diff'.")

    return df_out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lags: Optional[List[int]] = None,
    min_lag: int = MIN_ALLOWED_LAG,
) -> pd.DataFrame:
    """Create lagged target features strictly respecting the 24-hour forecast horizon.

    To eliminate data leakage, every lag MUST be >= min_lag (default 24 hours).
    A lag of 24 at time t uses meter reading at t-24. At origin T predicting
    up to T+24, t-24 <= T, ensuring all inputs are known in advance.

    Args:
        df: Input DataFrame.
        target_col: Name of target column to lag.
        lags: List of lag periods in hours. Default from src.config [24, 48, 168].
        min_lag: Strict minimum allowed lag offset (default 24).

    Returns:
        pd.DataFrame: DataFrame with lag features.

    Raises:
        ValueError: If any lag is smaller than min_lag (leakage violation).
    """
    df_out = df.copy()
    lag_list = lags if lags is not None else LAG_HOURS

    for lag in lag_list:
        if lag < min_lag:
            error_msg = (
                f"Data leakage detected! Requested lag {lag} is smaller than "
                f"minimum required horizon lag {min_lag}. Lags must be >= {min_lag}."
            )
            logger.critical(error_msg)
            raise ValueError(error_msg)

        col_name = f"lag_{lag}"
        df_out[col_name] = df_out[target_col].shift(lag)
        logger.debug("Added feature '%s' (shift=%d).", col_name, lag)

    return df_out


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    base_shift: int = MIN_ALLOWED_LAG,
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Create rolling statistics based ONLY on historical values shifted by >= 24h.

    To eliminate data leakage, the target series is FIRST shifted by base_shift
    (24 hours). Rolling aggregations (mean, std, min, max) are then calculated
    exclusively on this shifted series.

    For example, for window=24 with base_shift=24:
    Feature at time t aggregates meter readings from t-47 to t-24 inclusive.
    No observation from t-23 through t is ever accessed.

    Args:
        df: Input DataFrame.
        target_col: Name of target column.
        base_shift: Base shift offset in hours (default 24).
        windows: List of rolling window sizes in hours (default [24, 168]).

    Returns:
        pd.DataFrame: DataFrame with rolling statistical features.

    Raises:
        ValueError: If base_shift is smaller than MIN_ALLOWED_LAG.
    """
    if base_shift < MIN_ALLOWED_LAG:
        error_msg = (
            f"Data leakage violation! Rolling base_shift={base_shift} must be >= "
            f"MIN_ALLOWED_LAG={MIN_ALLOWED_LAG}."
        )
        logger.critical(error_msg)
        raise ValueError(error_msg)

    df_out = df.copy()
    win_list = windows if windows is not None else ROLLING_WINDOWS

    # Pre-shift the target series to strictly prevent any future lookahead
    shifted_target = df_out[target_col].shift(base_shift)

    for w in win_list:
        df_out[f"rolling_mean_{w}"] = shifted_target.rolling(window=w).mean()
        df_out[f"rolling_std_{w}"] = shifted_target.rolling(window=w).std()
        df_out[f"rolling_min_{w}"] = shifted_target.rolling(window=w).min()
        df_out[f"rolling_max_{w}"] = shifted_target.rolling(window=w).max()
        logger.debug(
            "Added rolling features for window=%d (shifted by %d hours).",
            w,
            base_shift,
        )

    return df_out


def compute_peak_cap(
    df: pd.DataFrame,
    percentile: float = PEAK_CAP_PERCENTILE,
    target_col: str = TARGET_COL,
    timestamp_col: str = TIMESTAMP_COL,
) -> float:
    """Compute the historical daily peak load and calculate the peak cap percentile.

    This function derives the peak threshold dynamically from the historical
    distribution rather than hardcoding it:
    1. Group observations by calendar date (24 hours per day).
    2. Extract the maximum demand (peak) for each day.
    3. Calculate the specified quantile (e.g. 0.92 for 92nd percentile).

    Args:
        df: Clean DataFrame containing historical meter readings.
        percentile: Percentile threshold for peak cap (default 0.92 from config).
        target_col: Name of electricity meter reading column.
        timestamp_col: Name of datetime column.

    Returns:
        float: Computed peak cap threshold in kW (rounded to 4 decimal places).

    Raises:
        ValueError: If DataFrame is empty or percentile is outside (0, 1).
    """
    if df.empty:
        raise ValueError("Cannot compute peak cap on an empty DataFrame.")
    if not (0.0 < percentile < 1.0):
        raise ValueError(f"Percentile must be between 0 and 1, got {percentile}")

    ts = pd.to_datetime(df[timestamp_col])
    daily_peaks = df.groupby(ts.dt.date)[target_col].max()

    peak_cap = float(daily_peaks.quantile(percentile))
    logger.info(
        "Computed historical daily peak cap at %.2f percentile: %.4f kW across %d days.",
        percentile,
        peak_cap,
        len(daily_peaks),
    )
    return round(peak_cap, 4)


def compute_peak_cap_summary(
    df: pd.DataFrame,
    percentile: float = PEAK_CAP_PERCENTILE,
    target_col: str = TARGET_COL,
    timestamp_col: str = TIMESTAMP_COL,
) -> Dict[str, Any]:
    """Compute comprehensive diagnostic statistics for daily peak load and cap.

    Args:
        df: Input clean DataFrame.
        percentile: Percentile threshold for peak cap.
        target_col: Name of target column.
        timestamp_col: Name of timestamp column.

    Returns:
        Dict[str, Any]: Detailed peak diagnostics.
    """
    ts = pd.to_datetime(df[timestamp_col])
    daily_peaks = df.groupby(ts.dt.date)[target_col].max()
    peak_cap = float(daily_peaks.quantile(percentile))

    breach_days = int((daily_peaks > peak_cap).sum())
    breach_hours = int((df[target_col] > peak_cap).sum())

    summary = {
        "days_count": len(daily_peaks),
        "percentile": percentile,
        "peak_cap_kw": round(peak_cap, 4),
        "daily_peak_min": round(float(daily_peaks.min()), 4),
        "daily_peak_mean": round(float(daily_peaks.mean()), 4),
        "daily_peak_median": round(float(daily_peaks.median()), 4),
        "daily_peak_max": round(float(daily_peaks.max()), 4),
        "daily_peak_std": round(float(daily_peaks.std()), 4),
        "breach_days_count": breach_days,
        "breach_days_pct": round((breach_days / len(daily_peaks)) * 100.0, 2),
        "breach_hours_count": breach_hours,
        "breach_hours_pct": round((breach_hours / len(df)) * 100.0, 2),
    }
    return summary


def get_feature_column_names(df: pd.DataFrame) -> List[str]:
    """Identify the model input feature columns, excluding identifiers and target.

    Args:
        df: Feature-engineered DataFrame.

    Returns:
        List[str]: List of column names designated as model inputs.
    """
    non_feature_cols = {
        TIMESTAMP_COL,
        TARGET_COL,
        "building_id",
        "meter",
        "site_id",
        "primary_use",
        "square_feet",
        "year_built",
        "floor_count",
    }
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    return feature_cols


def create_feature_pipeline(
    df: pd.DataFrame,
    drop_na: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Execute the full end-to-end feature engineering pipeline.

    Steps:
    1. Calendar & Cyclical features (hour, day_of_week, month, is_weekend, sin/cos)
    2. Weather features & interactions (dew point depression)
    3. Lag features (lag_24, lag_48, lag_168)
    4. Rolling stats on shifted target (rolling_mean/std/min/max for 24h and 168h)
    5. Drop initial NaN burn-in rows caused by 168h lag (if drop_na=True)

    Args:
        df: Clean input DataFrame from src.data.load_clean_data.
        drop_na: Whether to drop rows with NaN values resulting from lag burn-in.

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - Enriched DataFrame with all features and target
            - List of feature column names for model training
    """
    logger.info("Executing feature engineering pipeline (drop_na=%s)...", drop_na)

    df_feats = add_calendar_features(df)
    df_feats = add_weather_features(df_feats)
    df_feats = add_lag_features(df_feats)
    df_feats = add_rolling_features(df_feats)

    feature_cols = get_feature_column_names(df_feats)

    if drop_na:
        initial_rows = len(df_feats)
        df_feats = df_feats.dropna(subset=feature_cols).reset_index(drop=True)
        dropped_rows = initial_rows - len(df_feats)
        logger.info(
            "Dropped %d burn-in NaN rows (max lag=168h); %d rows remain.",
            dropped_rows,
            len(df_feats),
        )

    logger.info(
        "Feature pipeline complete: shape=%s, num_features=%d",
        df_feats.shape,
        len(feature_cols),
    )
    return df_feats, feature_cols
