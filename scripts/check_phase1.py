"""Phase 1 Validation Script for PeakWise AI (HTH-ML-08).

Validates:
1. Selected building and meter type
2. Chronological ordering & timestamp continuity
3. Missing values & NaN counts across raw, clean, and feature data
4. Calculated daily peak cap (92nd percentile)
5. Zero data leakage (perturbation test, lag alignment, rolling alignment, target exclusion)
6. Feature table shape, columns, and data integrity

Outputs clear PASS / FAIL summary table and exits with status code 0 on success.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path when running script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.config import (
    BUILDING_ID,
    FORECAST_HORIZON,
    LAG_HOURS,
    METER_TYPE,
    PEAK_CAP_PERCENTILE,
    REQUIRED_RAW_COLUMNS,
    ROLLING_WINDOWS,
    TARGET_COL,
    TIMESTAMP_COL,
    WEATHER_COLS,
)
from src.data import load_clean_data, load_raw_parquet
from src.features import (
    compute_peak_cap,
    compute_peak_cap_summary,
    create_feature_pipeline,
    get_feature_column_names,
)


class Phase1Validator:
    """Systematic validation runner for Phase 1 requirements."""

    def __init__(self) -> None:
        self.results: List[Tuple[str, bool, str]] = []
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.clean_df: pd.DataFrame = pd.DataFrame()
        self.features_df: pd.DataFrame = pd.DataFrame()
        self.features_df_all: pd.DataFrame = pd.DataFrame()
        self.feature_cols: List[str] = []
        self.peak_cap: float = 0.0

    def record(self, test_name: str, passed: bool, detail: str) -> None:
        """Record check result."""
        self.results.append((test_name, passed, detail))

    def run_check_1_building_selection(self) -> None:
        """Check 1: Selected Building and Meter Type."""
        try:
            self.raw_df = load_raw_parquet()
            self.clean_df = load_clean_data()

            b_ids = self.clean_df["building_id"].unique().tolist()
            meters = self.clean_df["meter"].unique().tolist()

            is_correct_bldg = (b_ids == [BUILDING_ID])
            is_correct_meter = (meters == [METER_TYPE])

            passed = is_correct_bldg and is_correct_meter
            detail = (
                f"Building ID={b_ids} (expected [{BUILDING_ID}]), "
                f"Meter={meters} (expected [{METER_TYPE}])"
            )
            self.record("Check 1: Building & Meter Selection", passed, detail)
        except Exception as e:
            self.record("Check 1: Building & Meter Selection", False, f"Error: {e}")

    def run_check_2_chronological_ordering(self) -> None:
        """Check 2: Chronological Ordering & Timestamp Continuity."""
        try:
            ts = self.clean_df[TIMESTAMP_COL]
            is_datetime = pd.api.types.is_datetime64_any_dtype(ts)
            is_monotonic = bool(ts.is_monotonic_increasing)
            duplicates = int(ts.duplicated().sum())

            diffs = ts.diff().dropna()
            all_1h_step = bool((diffs == pd.Timedelta(hours=1)).all())

            passed = is_datetime and is_monotonic and (duplicates == 0) and all_1h_step
            detail = (
                f"Monotonic={is_monotonic}, Duplicates={duplicates}, "
                f"Strict 1h intervals={all_1h_step}, Total hours={len(ts)} "
                f"({ts.min()} to {ts.max()})"
            )
            self.record("Check 2: Chronological Ordering & Continuity", passed, detail)
        except Exception as e:
            self.record("Check 2: Chronological Ordering & Continuity", False, f"Error: {e}")

    def run_check_3_nan_counts(self) -> None:
        """Check 3: Missing Value Handling & NaN Counts."""
        try:
            raw_nulls = int(self.raw_df.isnull().sum().sum())
            clean_nulls = int(self.clean_df[REQUIRED_RAW_COLUMNS].isnull().sum().sum())

            # Generate features with drop_na=False and drop_na=True
            self.features_df_all, _ = create_feature_pipeline(self.clean_df, drop_na=False)
            self.features_df, self.feature_cols = create_feature_pipeline(self.clean_df, drop_na=True)

            feature_nulls_dropped = int(self.features_df[self.feature_cols].isnull().sum().sum())
            # Exact burn-in: target shifted by FORECAST_HORIZON (24), then rolling max window (168): 24 + 168 - 1 = 191
            burn_in_threshold = FORECAST_HORIZON + max(ROLLING_WINDOWS) - 1

            # After burn-in, there should be zero nulls even in features_df_all
            post_burn_in_nulls = int(
                self.features_df_all.iloc[burn_in_threshold:][self.feature_cols].isnull().sum().sum()
            )

            passed = (clean_nulls == 0) and (feature_nulls_dropped == 0) and (post_burn_in_nulls == 0)
            detail = (
                f"Clean DF nulls={clean_nulls}, "
                f"Feature DF (drop_na=True) nulls={feature_nulls_dropped}, "
                f"Post burn-in (>={burn_in_threshold}h) nulls={post_burn_in_nulls}"
            )
            self.record("Check 3: NaN Counts & Handling", passed, detail)
        except Exception as e:
            self.record("Check 3: NaN Counts & Handling", False, f"Error: {e}")

    def run_check_4_calculated_peak_cap(self) -> None:
        """Check 4: Calculated Peak Cap (92nd Percentile)."""
        try:
            self.peak_cap = compute_peak_cap(self.clean_df, percentile=PEAK_CAP_PERCENTILE)
            summary = compute_peak_cap_summary(self.clean_df, percentile=PEAK_CAP_PERCENTILE)

            # Verification assertions
            valid_range = summary["daily_peak_min"] < self.peak_cap < summary["daily_peak_max"]
            correct_days = summary["days_count"] == 366  # Leap year 2016
            expected_p92 = 147.55  # Mathematical 92nd percentile for ASHRAE bldg 1301 meter 0
            matches_expected = np.isclose(self.peak_cap, expected_p92, atol=1e-2)

            passed = valid_range and correct_days and matches_expected
            detail = (
                f"Computed Peak Cap={self.peak_cap:.2f} kW (92nd percentile across "
                f"{summary['days_count']} days, min={summary['daily_peak_min']:.1f} kW, "
                f"max={summary['daily_peak_max']:.1f} kW, "
                f"breach days={summary['breach_days_count']} ({summary['breach_days_pct']}%))"
            )
            self.record("Check 4: Calculated Peak Cap", passed, detail)
        except Exception as e:
            self.record("Check 4: Calculated Peak Cap", False, f"Error: {e}")

    def run_check_5_leakage_validation(self) -> None:
        """Check 5: Strict Zero Data Leakage Validation."""
        try:
            leakage_errors: List[str] = []

            # 5A: Target Exclusion Check
            if TARGET_COL in self.feature_cols:
                leakage_errors.append(f"Target '{TARGET_COL}' directly present in feature matrix!")

            # 5B: Direct Lag Alignment Check
            for i in [200, 500, 1000, 5000]:
                for lag in [24, 48, 168]:
                    actual_feat_val = self.features_df_all[f"lag_{lag}"].iloc[i]
                    expected_target_val = self.clean_df[TARGET_COL].iloc[i - lag]
                    if not np.isclose(actual_feat_val, expected_target_val):
                        leakage_errors.append(
                            f"Lag alignment mismatch at row {i}, lag {lag}: "
                            f"{actual_feat_val} != {expected_target_val}"
                        )

            # 5C: Rolling Window Alignment Check
            # rolling_mean_24 at row i must equal mean of target from i-47 to i-24
            for i in [200, 500, 1000]:
                actual_roll = self.features_df_all["rolling_mean_24"].iloc[i]
                expected_roll = self.clean_df[TARGET_COL].iloc[i - 47 : i - 23].mean()
                if not np.isclose(actual_roll, expected_roll):
                    leakage_errors.append(
                        f"Rolling alignment mismatch at row {i}: {actual_roll} != {expected_roll}"
                    )

            # 5D: Perturbation Sensitivity Test (Mathematical Proof of No Leakage)
            # Perturb target at index k. Rows j < k + 24 MUST remain strictly identical.
            k = 500
            perturbed_df = self.clean_df.copy()
            perturbed_df.loc[k, TARGET_COL] += 5000.0

            feats_orig, _ = create_feature_pipeline(self.clean_df, drop_na=False)
            feats_pert, _ = create_feature_pipeline(perturbed_df, drop_na=False)

            # Inspect all feature columns for rows 0 to k + 23 (i.e. < k + 24)
            pre_horizon_slice_orig = feats_orig.iloc[: k + 24][self.feature_cols].values
            pre_horizon_slice_pert = feats_pert.iloc[: k + 24][self.feature_cols].values

            diff = np.abs(np.nan_to_num(pre_horizon_slice_orig) - np.nan_to_num(pre_horizon_slice_pert))
            max_diff_pre_horizon = float(np.max(diff))

            if max_diff_pre_horizon > 1e-6:
                leakage_errors.append(
                    f"Perturbation leakage detected! Max difference before row {k+24} is {max_diff_pre_horizon}."
                )

            # At row k + 24, lag_24 MUST change by exactly 5000.0 (confirms sensitivity)
            lag24_diff_at_horizon = abs(
                feats_pert["lag_24"].iloc[k + 24] - feats_orig["lag_24"].iloc[k + 24]
            )
            if not np.isclose(lag24_diff_at_horizon, 5000.0):
                leakage_errors.append(
                    f"Sensitivity check failed at row {k+24}: diff={lag24_diff_at_horizon}, expected 5000.0"
                )

            passed = len(leakage_errors) == 0
            detail = (
                f"Perturbation max diff before T+24: {max_diff_pre_horizon:.1e} (threshold < 1e-6), "
                f"Lag & rolling alignments verified, Target excluded from X."
            )
            if not passed:
                detail = f"Failures: {'; '.join(leakage_errors)}"

            self.record("Check 5: Zero Data Leakage Validation", passed, detail)
        except Exception as e:
            self.record("Check 5: Zero Data Leakage Validation", False, f"Error: {e}")

    def run_check_6_feature_shape_and_schema(self) -> None:
        """Check 6: Feature Table Shape, Required Columns & Dtypes."""
        try:
            expected_shape = (8593, 39)
            actual_shape = self.features_df.shape
            shape_match = (actual_shape == expected_shape)

            expected_required_features = [
                "hour", "day_of_week", "month", "is_weekend",
                "hour_sin", "hour_cos",
                "air_temperature", "dew_temperature", "cloud_coverage",
                "precip_depth_1_hr", "sea_level_pressure", "wind_direction", "wind_speed",
                "lag_24", "lag_168",
                "rolling_mean_24", "rolling_std_24",
                "rolling_mean_168", "rolling_std_168",
            ]
            missing_features = [f for f in expected_required_features if f not in self.feature_cols]

            passed = shape_match and (len(missing_features) == 0)
            detail = (
                f"Shape={actual_shape} (expected {expected_shape}), "
                f"Features count={len(self.feature_cols)}, "
                f"Missing expected features={missing_features}"
            )
            self.record("Check 6: Feature Table Shape & Schema", passed, detail)
        except Exception as e:
            self.record("Check 6: Feature Table Shape & Schema", False, f"Error: {e}")

    def run_all(self) -> bool:
        """Run all Phase 1 checks and print clean report."""
        print("=" * 80)
        print("PEAKWISE AI (HTH-ML-08) - PHASE 1 VALIDATION RUNNER")
        print("=" * 80)

        self.run_check_1_building_selection()
        self.run_check_2_chronological_ordering()
        self.run_check_3_nan_counts()
        self.run_check_4_calculated_peak_cap()
        self.run_check_5_leakage_validation()
        self.run_check_6_feature_shape_and_schema()

        all_passed = True
        print("\n" + "-" * 80)
        print(f"{'CHECK NAME':<45} | {'STATUS':<8} | {'DETAILS'}")
        print("-" * 80)

        for name, passed, detail in self.results:
            status_str = "PASS" if passed else "FAIL"
            print(f"{name:<45} | {status_str:<8} | {detail}")
            if not passed:
                all_passed = False

        print("-" * 80)
        print("\n" + "=" * 80)
        if all_passed:
            print("OVERALL RESULT: [PASS] - All Phase 1 requirements met successfully!")
        else:
            print("OVERALL RESULT: [FAIL] - One or more Phase 1 requirements failed.")
        print("=" * 80)
        return all_passed


def main() -> None:
    validator = Phase1Validator()
    success = validator.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
