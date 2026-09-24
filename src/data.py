"""Data ingestion, validation, and preprocessing module for PeakWise AI.

This module is responsible for loading the preprocessed ASHRAE parquet dataset,
validating column schema, filtering by building and meter type, sorting chronologically,
and preparing a clean, contiguous time series for downstream feature engineering.
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.config import (
    BUILDING_ID,
    METER_TYPE,
    PROCESSED_DATA_PATH,
    REQUIRED_RAW_COLUMNS,
    TIMESTAMP_COL,
    WEATHER_COLS,
)

logger = logging.getLogger("peakwise.data")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_raw_parquet(file_path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load the raw parquet dataset from disk.

    Args:
        file_path: Optional path to the parquet file. If None, uses
            PROCESSED_DATA_PATH from src.config.

    Returns:
        pd.DataFrame: Loaded raw dataset.

    Raises:
        FileNotFoundError: If the specified parquet file does not exist.
    """
    path = Path(file_path) if file_path is not None else PROCESSED_DATA_PATH
    if not path.is_file():
        logger.error("Dataset not found at path: %s", path)
        raise FileNotFoundError(f"Parquet dataset not found at: {path}")

    logger.info("Loading dataset from: %s", path)
    df = pd.read_parquet(path)
    logger.info("Loaded parquet with shape: %s", df.shape)
    return df


def validate_columns(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None,
) -> None:
    """Validate that all required columns exist in the DataFrame.

    Args:
        df: Input DataFrame to check.
        required_cols: List of required column names. If None, uses
            REQUIRED_RAW_COLUMNS from src.config.

    Raises:
        ValueError: If any required column is missing.
    """
    cols_to_check = required_cols if required_cols is not None else REQUIRED_RAW_COLUMNS
    missing_cols = [c for c in cols_to_check if c not in df.columns]

    if missing_cols:
        error_msg = f"Missing required columns in dataset: {missing_cols}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug("All required columns validated: %s", cols_to_check)


def filter_by_building_and_meter(
    df: pd.DataFrame,
    building_id: int = BUILDING_ID,
    meter_type: Optional[int] = METER_TYPE,
) -> pd.DataFrame:
    """Filter DataFrame to the target building ID and specific meter type.

    Args:
        df: Input DataFrame containing building and meter data.
        building_id: Target building ID (default 1301).
        meter_type: Target meter type (0 = electricity). If None, keeps all meters.

    Returns:
        pd.DataFrame: Filtered DataFrame copy.

    Raises:
        ValueError: If filtering results in an empty DataFrame.
    """
    mask = df["building_id"] == building_id
    if meter_type is not None:
        mask = mask & (df["meter"] == meter_type)

    filtered = df.loc[mask].copy()

    if filtered.empty:
        error_msg = (
            f"No records found for building_id={building_id} "
            f"and meter_type={meter_type}."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(
        "Filtered to building_id=%d, meter_type=%s: %d records retained.",
        building_id,
        str(meter_type),
        len(filtered),
    )
    return filtered


def clean_and_sort_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = TIMESTAMP_COL,
) -> pd.DataFrame:
    """Ensure timestamp format, check uniqueness, and sort chronologically.

    Args:
        df: Input DataFrame.
        timestamp_col: Column name containing datetime values.

    Returns:
        pd.DataFrame: Chronologically sorted DataFrame with reset index.

    Raises:
        ValueError: If duplicate timestamps are found for the filtered series.
    """
    df_clean = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_clean[timestamp_col]):
        df_clean[timestamp_col] = pd.to_datetime(df_clean[timestamp_col])

    # Check duplicates
    duplicate_count = int(df_clean[timestamp_col].duplicated().sum())
    if duplicate_count > 0:
        logger.warning(
            "Found %d duplicate timestamps; dropping subsequent duplicates.",
            duplicate_count,
        )
        df_clean = df_clean.drop_duplicates(subset=[timestamp_col], keep="first")

    # Sort strictly ascending
    df_clean = df_clean.sort_values(timestamp_col).reset_index(drop=True)
    logger.debug(
        "Timestamps sorted chronologically from %s to %s (%d records).",
        df_clean[timestamp_col].min(),
        df_clean[timestamp_col].max(),
        len(df_clean),
    )
    return df_clean


def clean_weather_features(
    df: pd.DataFrame,
    weather_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Impute missing weather observations using time-aware linear interpolation.

    Performs linear interpolation with forward-fill and back-fill boundary handling
    to guarantee a complete, contiguous weather series without introducing lookahead
    bias into future periods.

    Args:
        df: Input DataFrame with weather columns.
        weather_cols: List of weather column names. If None, uses WEATHER_COLS from src.config.

    Returns:
        pd.DataFrame: DataFrame with clean, non-null weather features.
    """
    df_clean = df.copy()
    cols = weather_cols if weather_cols is not None else WEATHER_COLS
    cols_present = [c for c in cols if c in df_clean.columns]

    initial_nulls = int(df_clean[cols_present].isnull().sum().sum())
    if initial_nulls > 0:
        logger.info(
            "Imputing %d missing weather values across %d columns using linear interpolation.",
            initial_nulls,
            len(cols_present),
        )
        df_clean[cols_present] = (
            df_clean[cols_present]
            .interpolate(method="linear", limit_direction="both")
            .bfill()
            .ffill()
        )

    remaining_nulls = int(df_clean[cols_present].isnull().sum().sum())
    if remaining_nulls > 0:
        logger.warning("Remaining nulls in weather columns: %d", remaining_nulls)
    else:
        logger.debug("Weather columns successfully cleaned with 0 nulls.")

    return df_clean


def load_clean_data(
    file_path: Optional[Path | str] = None,
    building_id: int = BUILDING_ID,
    meter_type: Optional[int] = METER_TYPE,
) -> pd.DataFrame:
    """Load, validate, filter, and clean data for the target building.

    This function acts as the primary data ingestion pipeline for Phase 1.

    Args:
        file_path: Optional path to raw parquet file.
        building_id: Target building ID (default: 1301 from config).
        meter_type: Target meter type (default: 0 = electricity from config).

    Returns:
        pd.DataFrame: Clean, sorted, validated DataFrame ready for feature engineering.
    """
    logger.info("Initiating data loading and cleaning pipeline...")
    raw_df = load_raw_parquet(file_path=file_path)
    validate_columns(raw_df)
    filtered_df = filter_by_building_and_meter(
        raw_df,
        building_id=building_id,
        meter_type=meter_type,
    )
    sorted_df = clean_and_sort_timestamps(filtered_df)
    clean_df = clean_weather_features(sorted_df)

    logger.info(
        "Data pipeline complete: %d rows, %d columns, range %s to %s.",
        len(clean_df),
        clean_df.shape[1],
        clean_df[TIMESTAMP_COL].min(),
        clean_df[TIMESTAMP_COL].max(),
    )
    return clean_df
