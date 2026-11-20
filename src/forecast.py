"""Forecasting inference module for PeakWise AI.

This module provides the primary inference API for 24-hour ahead probabilistic
energy forecasting using pre-trained LightGBM quantile models (P10, P50, P90).

CRITICAL CONSTRAINTS:
1. This is the ONLY module that reads and loads model files from models/.
2. Cached model objects are maintained at the module level.
3. Exposes predict(building_id, start_ts, weather_override=None) returning
   DataFrame[timestamp, p10, p50, p90] for the next 24 hours.
4. Additive weather overrides are applied prior to feature reconstruction so
   thermodynamic interactions (e.g. temp_dew_diff) update dynamically.
5. Quantile monotonicity (p10 <= p50 <= p90) and non-negative bounds are enforced.
6. Reuses src.features for feature engineering without logic duplication.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


from src.data import load_clean_data
from src.features import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_weather_features,
)

logger = logging.getLogger("peakwise.forecast")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Module-level model and metadata cache
_MODELS_CACHE: Optional[Dict[str, Any]] = None
_FEATURE_COLS_CACHE: Optional[List[str]] = None


def load_models(
    models_dir: Optional[Path | str] = None,
    force_reload: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    """Load and cache pre-trained LightGBM models and feature column definitions.

    Reads q10.joblib, q50.joblib, q90.joblib, and feature_list.json from the
    specified models directory. Results are cached in module memory.

    Args:
        models_dir: Directory containing saved model artifacts. If None, uses
            MODELS_DIR from src.config.
        force_reload: If True, invalidates existing cache and reloads from disk.

    Returns:
        Tuple[Dict[str, Any], List[str]]:
            - Dictionary mapping quantile names ("q10", "q50", "q90") to model objects.
            - List of expected input feature column names.

    Raises:
        FileNotFoundError: If any model file or feature_list.json is missing.
    """
    global _MODELS_CACHE, _FEATURE_COLS_CACHE

    if _MODELS_CACHE is not None and _FEATURE_COLS_CACHE is not None and not force_reload:
        return _MODELS_CACHE, _FEATURE_COLS_CACHE

    path = Path(models_dir) if models_dir is not None else MODELS_DIR
    logger.info("Loading forecast models from %s...", path)

    q10_file = path / "q10.joblib"
    q50_file = path / "q50.joblib"
    q90_file = path / "q90.joblib"
    fcols_file = path / "feature_list.json"

    for file_path in [q10_file, q50_file, q90_file, fcols_file]:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required model artifact not found: {file_path}. "
                "Ensure train.py has been executed to generate models."
            )

    models = {
        "q10": joblib.load(q10_file),
        "q50": joblib.load(q50_file),
        "q90": joblib.load(q90_file),
    }

    with open(fcols_file, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    _MODELS_CACHE = models
    _FEATURE_COLS_CACHE = feature_cols

    logger.info("Successfully loaded q10, q50, q90 models and %d features.", len(feature_cols))
    return _MODELS_CACHE, _FEATURE_COLS_CACHE


def apply_weather_override(
    df: pd.DataFrame,
    start_ts: pd.Timestamp,
    horizon_hours: int = FORECAST_HORIZON,
    weather_override: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Apply additive deltas to weather features within the 24-hour forecast window.

    Must be called BEFORE feature engineering so derived weather thermodynamics
    (e.g., dew point depression temp_dew_diff) update dynamically.

    Args:
        df: Input clean DataFrame.
        start_ts: Beginning timestamp of the 24-hour prediction window.
        horizon_hours: Forecast window duration in hours (default 24).
        weather_override: Optional dictionary of additive offsets, e.g.
            {"air_temperature": 5.0, "dew_temperature": 2.0}.

    Returns:
        pd.DataFrame: DataFrame copy with weather overrides applied to the forecast window.
    """
    df_out = df.copy()
    if not weather_override:
        return df_out

    end_ts = start_ts + pd.Timedelta(hours=horizon_hours - 1)
    mask = (df_out[TIMESTAMP_COL] >= start_ts) & (df_out[TIMESTAMP_COL] <= end_ts)

    if not mask.any():
        logger.warning(
            "Forecast window [%s to %s] not found in dataset; weather override not applied.",
            start_ts,
            end_ts,
        )
        return df_out

    for param, delta in weather_override.items():
        if param in df_out.columns:
            df_out.loc[mask, param] = df_out.loc[mask, param] + delta
            logger.info(
                "Applied weather override on [%s - %s]: %s += %.2f",
                start_ts,
                end_ts,
                param,
                delta,
            )
        else:
            logger.warning("Weather override parameter '%s' not present in DataFrame.", param)

    return df_out


def build_forecast_features(
    clean_df: pd.DataFrame,
    start_ts: pd.Timestamp,
    feature_cols: List[str],
    horizon_hours: int = FORECAST_HORIZON,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct leakage-free features for the 24-hour prediction window.

    Reuses feature engineering routines from src.features. Guarantees that
    only information available prior to the forecast window is used.

    Args:
        clean_df: Cleaned dataset (with any weather overrides already applied).
        start_ts: Forecast origin timestamp (inclusive start).
        feature_cols: List of expected input feature columns for the models.
        horizon_hours: Forecast horizon in hours (default 24).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - X: Feature matrix of shape (24, len(feature_cols)) ready for inference.
            - window_df: Corresponding DataFrame slice containing timestamps.

    Raises:
        ValueError: If fewer than 24 consecutive hourly timestamps exist for the window.
    """
    # Build complete feature representations using shared library functions
    df_feats = add_calendar_features(clean_df)
    df_feats = add_weather_features(df_feats)
    df_feats = add_lag_features(df_feats)
    df_feats = add_rolling_features(df_feats)

    end_ts = start_ts + pd.Timedelta(hours=horizon_hours - 1)
    mask = (df_feats[TIMESTAMP_COL] >= start_ts) & (df_feats[TIMESTAMP_COL] <= end_ts)
    window_df = df_feats.loc[mask].copy().reset_index(drop=True)

    if len(window_df) != horizon_hours:
        raise ValueError(
            f"Forecast window [{start_ts} to {end_ts}] has {len(window_df)} hours, "
            f"expected exactly {horizon_hours} hours."
        )

    # Validate all feature columns exist
    missing = [c for c in feature_cols if c not in window_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for inference: {missing}")

    X = window_df[feature_cols].copy()
    return X, window_df


def enforce_quantile_constraints(
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enforce quantile monotonicity (p10 <= p50 <= p90) and non-negative power bounds.

    Args:
        p10: Raw P10 predictions (kW).
        p50: Raw P50 predictions (kW).
        p90: Raw P90 predictions (kW).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Monotonically consistent, non-negative (p10, p50, p90) arrays.
    """
    matrix = np.column_stack([p10, p50, p90])
    # Clip electrical power at 0 kW (facilities do not generate power into the grid in this MVP)
    matrix = np.clip(matrix, 0.0, None)
    # Sort per row across quantile dimension to guarantee monotonicity
    matrix.sort(axis=1)

    return matrix[:, 0], matrix[:, 1], matrix[:, 2]


def predict(
    building_id: int = BUILDING_ID,
    start_ts: pd.Timestamp | str = "2016-12-15 00:00:00",
    weather_override: Optional[Dict[str, float]] = None,
    models_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Generate 24-hour ahead quantile forecasts (P10, P50, P90).

    This is the primary public entry point for all downstream optimization,
    peak breach detection, and dashboard visualization.

    Args:
        building_id: Target facility building ID (default: 1301 from config).
        start_ts: Starting timestamp for the 24-hour forecast horizon (inclusive).
        weather_override: Optional dictionary of additive offsets applied to weather
            variables (e.g. {"air_temperature": 5.0}).
        models_dir: Directory where trained models are saved.

    Returns:
        pd.DataFrame: Forecast table with columns:
            ['timestamp', 'p10', 'p50', 'p90'] for the next 24 consecutive hours.
    """
    ts_start = pd.to_datetime(start_ts)
    models, feature_cols = load_models(models_dir=models_dir)

    # Ingest clean historical series for the building
    clean_df = load_clean_data(building_id=building_id, meter_type=METER_TYPE)

    # Apply additive weather override prior to feature derivation
    clean_df = apply_weather_override(
        clean_df,
        start_ts=ts_start,
        horizon_hours=FORECAST_HORIZON,
        weather_override=weather_override,
    )

    # Build features cleanly
    X, window_df = build_forecast_features(
        clean_df=clean_df,
        start_ts=ts_start,
        feature_cols=feature_cols,
        horizon_hours=FORECAST_HORIZON,
    )

    # Run inference across models
    raw_p10 = models["q10"].predict(X)
    raw_p50 = models["q50"].predict(X)
    raw_p90 = models["q90"].predict(X)

    # Enforce p10 <= p50 <= p90 and >= 0 kW
    p10, p50, p90 = enforce_quantile_constraints(raw_p10, raw_p50, raw_p90)

    forecast_df = pd.DataFrame(
        {
            TIMESTAMP_COL: window_df[TIMESTAMP_COL].values,
            "p10": np.round(p10, 4),
            "p50": np.round(p50, 4),
            "p90": np.round(p90, 4),
        }
    )

    logger.info(
        "Generated 24h forecast for [%s to %s]: p50 min=%.2f, max=%.2f kW; p90 max=%.2f kW.",
        forecast_df[TIMESTAMP_COL].iloc[0],
        forecast_df[TIMESTAMP_COL].iloc[-1],
        forecast_df["p50"].min(),
        forecast_df["p50"].max(),
        forecast_df["p90"].max(),
    )
    return forecast_df
