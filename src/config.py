"""Configuration module for PeakWise AI.

This module serves as the single source of truth for all project parameters,
hyperparameters, paths, and domain assumptions across all pipeline phases.
Do NOT hardcode these constants across other modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# -----------------------------------------------------------------------------
# Base Directories & Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH: Path = DATA_DIR / "processed" / "ashrae_processed.parquet"
MODELS_DIR: Path = PROJECT_ROOT / "models"


# -----------------------------------------------------------------------------
# Target & Building Identification
# -----------------------------------------------------------------------------
BUILDING_ID: int = 1301
METER_TYPE: int = 0  # 0 = Electricity (ASHRAE standard)
TARGET_COL: str = "meter_reading"
TIMESTAMP_COL: str = "timestamp"


# -----------------------------------------------------------------------------
# Forecasting Horizon & Quantiles
# -----------------------------------------------------------------------------
FORECAST_HORIZON: int = 24  # 24-hour ahead hourly forecast
QUANTILES: List[float] = [0.1, 0.5, 0.9]


# -----------------------------------------------------------------------------
# Peak Cap & Demand Charge Economics
# -----------------------------------------------------------------------------
PEAK_CAP_PERCENTILE: float = 0.92  # 92nd percentile of historical daily peaks

def compute_historical_peak_cap(
    parquet_path: Path = PROCESSED_DATA_PATH,
    building_id: int = BUILDING_ID,
    meter_type: int = METER_TYPE,
    percentile: float = PEAK_CAP_PERCENTILE,
) -> float:
    """Compute peak cap as the N-th percentile of historical daily peak loads."""
    try:
        if Path(parquet_path).is_file():
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            mask = (df["building_id"] == building_id) & (df["meter"] == meter_type)
            bldg = df.loc[mask]
            if not bldg.empty and "meter_reading" in bldg.columns:
                daily_peaks = bldg.groupby(bldg["timestamp"].dt.date)["meter_reading"].max()
                return round(float(daily_peaks.quantile(percentile)), 2)
    except Exception:
        pass
    return 147.55  # Fallback default (computed 92nd percentile)

PEAK_CAP_KW: float = compute_historical_peak_cap()  # Default computed 92nd percentile peak cap (kW)
DEMAND_CHARGE_PER_KW: float = 15.00  # $ / kW monthly billing peak demand charge
DEMAND_CHARGE_RATE: float = DEMAND_CHARGE_PER_KW  # Alias for compatibility
BREACH_PENALTY_PER_KW: float = 35.00  # $ / kW penalty for breaching peak cap
BASE_ENERGY_RATE: float = 0.08  # $ / kWh baseline volumetric energy rate

DAYS_PER_MONTH: float = 30.0
DAYS_PER_YEAR: float = 365.0

# Time of Use (TOU) Rates by Hour of Day ($/kWh)
# Peak: 12 PM - 6 PM ($0.18/kWh), Off-Peak ($0.08/kWh)
TOU_PEAK_HOURS: List[int] = list(range(12, 18))
TOU_PEAK_RATE: float = 0.18
TOU_OFF_PEAK_RATE: float = 0.08
TOU_HOURLY_RATES: List[float] = [
    TOU_PEAK_RATE if h in TOU_PEAK_HOURS else TOU_OFF_PEAK_RATE
    for h in range(24)
]
TOU_RATES: List[float] = TOU_HOURLY_RATES  # Canonical alias

# -----------------------------------------------------------------------------
# Optimization & Load Shifting Parameters
# -----------------------------------------------------------------------------
RISK_LEVEL: str = "p90"  # Target curve to optimize against ('p50' or 'p90')
MAX_DEFER_HOURS: int = 3  # Maximum hours a load can be deferred forward (t > h)
MAX_PRECOOL_HOURS: int = 3  # Maximum hours a load can be pre-served / pre-cooled (t < h)
MAX_SHIFT_FRACTION_PER_HOUR: float = 0.50  # Up to 50% of an hour's shiftable load can be moved
MAX_SHIFT_PCT_PER_HOUR: float = MAX_SHIFT_FRACTION_PER_HOUR  # Alias
REBOUND_FACTOR: float = 1.05  # 5% energy overhead incurred by shifting load (efficiency loss)
REBOUND_PENALTY: float = REBOUND_FACTOR  # Alias

# Shiftable Load Share by Hour of Day
# In Education facilities: higher shiftable potential during daytime HVAC operation
DEFAULT_SHIFTABLE_PROFILE_EDUCATION: List[float] = [
    0.20, 0.20, 0.20, 0.20, 0.20, 0.25,  # 00:00 - 05:00 (Night base)
    0.35, 0.40, 0.45, 0.45, 0.45, 0.45,  # 06:00 - 11:00 (Morning ramp & class hours)
    0.45, 0.45, 0.45, 0.45, 0.40, 0.35,  # 12:00 - 17:00 (Afternoon peak)
    0.30, 0.25, 0.25, 0.20, 0.20, 0.20,  # 18:00 - 23:00 (Evening setback)
]

DEFAULT_SHIFTABLE_PROFILE_MANUFACTURING: List[float] = [
    0.30, 0.30, 0.30, 0.30, 0.30, 0.35,
    0.40, 0.45, 0.45, 0.45, 0.45, 0.45,
    0.45, 0.45, 0.45, 0.45, 0.40, 0.40,
    0.35, 0.35, 0.30, 0.30, 0.30, 0.30,
]

SHIFTABLE_PROFILES_BY_USE = {
    "Education": DEFAULT_SHIFTABLE_PROFILE_EDUCATION,
    "Manufacturing": DEFAULT_SHIFTABLE_PROFILE_MANUFACTURING,
    "default": DEFAULT_SHIFTABLE_PROFILE_EDUCATION,
}
SHIFTABLE_SHARE = SHIFTABLE_PROFILES_BY_USE  # Canonical alias

# Non-shiftable load share (remainder of total load)
NON_SHIFTABLE_SHARE = {
    use: [round(1.0 - val, 4) for val in profile]
    for use, profile in SHIFTABLE_PROFILES_BY_USE.items()
}
NON_SHIFTABLE = NON_SHIFTABLE_SHARE  # Canonical alias


# -----------------------------------------------------------------------------
# Data Columns Specification
# -----------------------------------------------------------------------------
REQUIRED_RAW_COLUMNS: List[str] = [
    "building_id",
    "meter",
    "timestamp",
    "meter_reading",
    "air_temperature",
    "dew_temperature",
    "cloud_coverage",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
]

WEATHER_COLS: List[str] = [
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
]


# -----------------------------------------------------------------------------
# Feature Engineering Parameters (Leakage-Free 24h Horizon)
# -----------------------------------------------------------------------------
# Lags must be >= FORECAST_HORIZON (24) so that no future/current values within
# the 24-hour prediction window are accessed.
LAG_HOURS: List[int] = [24, 48, 168]
ROLLING_WINDOWS: List[int] = [24, 168]
MIN_ALLOWED_LAG: int = FORECAST_HORIZON  # Strict lower bound on lag offset


# -----------------------------------------------------------------------------
# Manufacturing Load Assumptions (Domain Model for Later Phases)
# -----------------------------------------------------------------------------
NON_SHIFTABLE_LOADS: List[str] = [
    "Rubber molding presses",
    "Curing machines",
    "Critical production equipment",
    "Safety systems",
]

SHIFTABLE_LOADS: List[str] = [
    "HVAC/chiller",
    "Air compressor",
    "Cooling/water pumps",
    "Forklift battery charging",
]


# -----------------------------------------------------------------------------
# Structured Configuration Class
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectConfig:
    """Immutable configuration container for PeakWise AI."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    processed_data_path: Path = PROCESSED_DATA_PATH
    models_dir: Path = MODELS_DIR

    building_id: int = BUILDING_ID
    meter_type: int = METER_TYPE
    target_col: str = TARGET_COL
    timestamp_col: str = TIMESTAMP_COL

    forecast_horizon: int = FORECAST_HORIZON
    quantiles: List[float] = field(default_factory=lambda: list(QUANTILES))

    peak_cap_percentile: float = PEAK_CAP_PERCENTILE
    peak_cap_kw: float = PEAK_CAP_KW
    demand_charge_rate: float = DEMAND_CHARGE_RATE
    base_energy_rate: float = BASE_ENERGY_RATE
    rebound_penalty: float = REBOUND_PENALTY
    max_shift_pct_per_hour: float = MAX_SHIFT_PCT_PER_HOUR

    required_raw_columns: List[str] = field(default_factory=lambda: list(REQUIRED_RAW_COLUMNS))
    weather_cols: List[str] = field(default_factory=lambda: list(WEATHER_COLS))
    lag_hours: List[int] = field(default_factory=lambda: list(LAG_HOURS))
    rolling_windows: List[int] = field(default_factory=lambda: list(ROLLING_WINDOWS))
    min_allowed_lag: int = MIN_ALLOWED_LAG

    non_shiftable_loads: List[str] = field(default_factory=lambda: list(NON_SHIFTABLE_LOADS))
    shiftable_loads: List[str] = field(default_factory=lambda: list(SHIFTABLE_LOADS))


# Singleton default config instance
CONFIG: ProjectConfig = ProjectConfig()
