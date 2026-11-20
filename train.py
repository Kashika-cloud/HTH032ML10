"""Model training and evaluation pipeline for PeakWise AI (Phase 2).

Trains LightGBM quantile regression models (P10, P50, P90) and a reference L2 model
using strict chronological time-based splitting:
- Test set: Final 4 weeks (28 days = 672 hours)
- Validation set: 2 weeks prior to test set (14 days = 336 hours) for early stopping
- Train set: All preceding data (~46 weeks)

Includes:
- Baselines: (a) Seasonal-naive (lag_168), (b) Naive yesterday (lag_24)
- Weather forecast noise simulation during training
- 24h rolling-origin backtest across every test day
- Comprehensive error scoring: MAE, RMSE, CV-RMSE, MAPE, daily-max error, peak-hour recall
- P90 coverage and peak-cap breach precision/recall
- Model artifact export: q10.joblib, q50.joblib, q90.joblib, feature_list.json,
  metrics.json, and feature_importance.csv
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import (
    BUILDING_ID,
    FORECAST_HORIZON,
    MODELS_DIR,
    PEAK_CAP_KW,
    QUANTILES,
    TARGET_COL,
    TIMESTAMP_COL,
    WEATHER_COLS,
)
from src.data import load_clean_data
from src.features import create_feature_pipeline

logger = logging.getLogger("peakwise.train")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def split_data_chronologically(
    df: pd.DataFrame,
    test_days: int = 28,
    val_days: int = 14,
    timestamp_col: str = TIMESTAMP_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform strict time-based train / validation / test splitting.

    Args:
        df: Feature-engineered DataFrame.
        test_days: Number of days allocated to test period (default 28).
        val_days: Number of days allocated to validation period (default 14).
        timestamp_col: Name of datetime column.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (train_df, val_df, test_df) partitioned strictly by date.
    """
    max_ts = df[timestamp_col].max()
    test_start = max_ts - pd.Timedelta(days=test_days) + pd.Timedelta(hours=1)
    val_start = test_start - pd.Timedelta(days=val_days)

    train_df = df[df[timestamp_col] < val_start].copy()
    val_df = df[(df[timestamp_col] >= val_start) & (df[timestamp_col] < test_start)].copy()
    test_df = df[df[timestamp_col] >= test_start].copy()

    logger.info(
        "Chronological split complete: Train=%d hrs (%s to %s), "
        "Val=%d hrs (%s to %s), Test=%d hrs (%s to %s)",
        len(train_df),
        train_df[timestamp_col].min(),
        train_df[timestamp_col].max(),
        len(val_df),
        val_df[timestamp_col].min(),
        val_df[timestamp_col].max(),
        len(test_df),
        test_df[timestamp_col].min(),
        test_df[timestamp_col].max(),
    )
    return train_df, val_df, test_df


def inject_weather_noise(
    train_df: pd.DataFrame,
    weather_cols: List[str],
    seed: int = 42,
    noise_ratio: float = 0.05,
) -> pd.DataFrame:
    """Add small Gaussian noise to weather features in training to simulate forecast uncertainty.

    Args:
        train_df: Training DataFrame.
        weather_cols: List of base weather columns.
        seed: Random seed for reproducibility.
        noise_ratio: Proportion of standard deviation used as noise scale.

    Returns:
        pd.DataFrame: Training DataFrame with perturbed weather features.
    """
    df_out = train_df.copy()
    rng = np.random.default_rng(seed)

    for col in weather_cols:
        if col in df_out.columns:
            std_val = float(df_out[col].std())
            scale = max(0.05, std_val * noise_ratio)
            noise = rng.normal(loc=0.0, scale=scale, size=len(df_out))
            df_out[col] = df_out[col] + noise

    if "air_temperature" in df_out.columns and "dew_temperature" in df_out.columns:
        if "temp_dew_diff" in df_out.columns:
            df_out["temp_dew_diff"] = df_out["air_temperature"] - df_out["dew_temperature"]

    logger.info("Injected weather forecast uncertainty noise into training split.")
    return df_out


def train_quantile_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    alpha: float,
    seed: int = 42,
) -> lgb.LGBMRegressor:
    """Train a single LightGBM quantile regression model with early stopping.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        alpha: Target quantile level (e.g. 0.1, 0.5, 0.9).
        seed: Random seed.

    Returns:
        lgb.LGBMRegressor: Trained model instance.
    """
    logger.info("Training LightGBM Quantile model (alpha=%.2f)...", alpha)
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    logger.info(
        "Model alpha=%.2f trained. Best iteration: %d",
        alpha,
        getattr(model, "best_iteration_", model.n_estimators),
    )
    return model


def train_l2_reference_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = 42,
) -> lgb.LGBMRegressor:
    """Train a standard L2 (mean squared error) LightGBM model for baseline reference.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        seed: Random seed.

    Returns:
        lgb.LGBMRegressor: Trained L2 reference model.
    """
    logger.info("Training LightGBM L2 reference model...")
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    logger.info(
        "L2 reference model trained. Best iteration: %d",
        getattr(model, "best_iteration_", model.n_estimators),
    )
    return model


def compute_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute standard point forecast metrics for a given true vs predicted series.

    Args:
        y_true: True meter readings (kW).
        y_pred: Predicted values (kW).

    Returns:
        Dict[str, float]: MAE, RMSE, CV-RMSE, and MAPE metrics.
    """
    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mean_true = float(np.mean(y_true))
    cv_rmse = float((rmse / mean_true) * 100.0) if mean_true > 0 else 0.0
    mape = float(np.mean(np.abs(diff) / (y_true + 1e-5)) * 100.0)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "cv_rmse": round(cv_rmse, 2),
        "mape": round(mape, 2),
    }


def run_rolling_origin_backtest(
    test_df: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
    peak_cap_kw: float = PEAK_CAP_KW,
    timestamp_col: str = TIMESTAMP_COL,
    target_col: str = TARGET_COL,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Execute rolling-origin 24-hour evaluation across all test days.

    Evaluates:
    - Global MAE, RMSE, CV-RMSE, MAPE
    - Daily-max absolute error (|max_true - max_pred|) averaged across days
    - Peak-hour recall (fraction of days predicted peak hour is within +/- 1h of true peak)
    - P90 coverage (fraction of hours y_true <= y_p90)
    - P90 breach detection precision, recall, and F1 at PEAK_CAP_KW

    Args:
        test_df: Test set DataFrame containing timestamp and target.
        predictions: Dictionary mapping model names to array of predicted values.
        peak_cap_kw: Peak demand cap threshold in kW.
        timestamp_col: Name of datetime column.
        target_col: Name of target column.

    Returns:
        Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
            - Comparison metrics dictionary for each model.
            - Diagnostic summary for P90 coverage and breach detection.
    """
    logger.info("Executing rolling-origin 24h backtest across test days...")
    y_true_all = test_df[target_col].values
    dates = test_df[timestamp_col].dt.date.unique()

    model_metrics: Dict[str, Dict[str, float]] = {}

    for model_name, y_pred_all in predictions.items():
        base_metrics = compute_window_metrics(y_true_all, y_pred_all)

        daily_max_errors = []
        peak_hour_hits = []

        for d in dates:
            mask = test_df[timestamp_col].dt.date == d
            t_true = y_true_all[mask]
            t_pred = y_pred_all[mask]

            if len(t_true) == 0:
                continue

            # Daily maximum absolute error
            max_true = np.max(t_true)
            max_pred = np.max(t_pred)
            daily_max_errors.append(abs(max_true - max_pred))

            # Peak hour recall within +/- 1 hour
            h_true = int(np.argmax(t_true))
            h_pred = int(np.argmax(t_pred))
            peak_hit = 1.0 if abs(h_true - h_pred) <= 1 else 0.0
            peak_hour_hits.append(peak_hit)

        base_metrics["daily_max_error"] = round(float(np.mean(daily_max_errors)), 4)
        base_metrics["peak_hour_recall"] = round(float(np.mean(peak_hour_hits)) * 100.0, 2)
        model_metrics[model_name] = base_metrics

    # P90 Quantile Specific Diagnostics
    p90_preds = predictions.get("q90", None)
    p90_diagnostics: Dict[str, Any] = {}

    if p90_preds is not None:
        coverage = float(np.mean(y_true_all <= p90_preds) * 100.0)

        # Daily breach evaluation
        true_daily_breaches = []
        pred_daily_breaches = []

        for d in dates:
            mask = test_df[timestamp_col].dt.date == d
            t_true = y_true_all[mask]
            t_pred = p90_preds[mask]
            true_daily_breaches.append(bool(np.max(t_true) > peak_cap_kw))
            pred_daily_breaches.append(bool(np.max(t_pred) > peak_cap_kw))

        t_b = np.array(true_daily_breaches)
        p_b = np.array(pred_daily_breaches)
        tp_daily = int(np.sum(t_b & p_b))
        fp_daily = int(np.sum(~t_b & p_b))
        fn_daily = int(np.sum(t_b & ~p_b))
        tn_daily = int(np.sum(~t_b & ~p_b))

        prec_daily = (tp_daily / (tp_daily + fp_daily) * 100.0) if (tp_daily + fp_daily) > 0 else 0.0
        rec_daily = (tp_daily / (tp_daily + fn_daily) * 100.0) if (tp_daily + fn_daily) > 0 else 0.0
        f1_daily = (
            2 * prec_daily * rec_daily / (prec_daily + rec_daily)
            if (prec_daily + rec_daily) > 0
            else 0.0
        )

        # Hourly breach evaluation
        hourly_true_b = y_true_all > peak_cap_kw
        hourly_pred_b = p90_preds > peak_cap_kw
        tp_hr = int(np.sum(hourly_true_b & hourly_pred_b))
        fp_hr = int(np.sum(~hourly_true_b & hourly_pred_b))
        fn_hr = int(np.sum(hourly_true_b & ~hourly_pred_b))
        prec_hr = (tp_hr / (tp_hr + fp_hr) * 100.0) if (tp_hr + fp_hr) > 0 else 0.0
        rec_hr = (tp_hr / (tp_hr + fn_hr) * 100.0) if (tp_hr + fn_hr) > 0 else 0.0

        p90_diagnostics = {
            "target_coverage_pct": 90.0,
            "actual_coverage_pct": round(coverage, 2),
            "peak_cap_kw": peak_cap_kw,
            "daily_breaches": {
                "true_breaches": int(np.sum(t_b)),
                "predicted_breaches": int(np.sum(p_b)),
                "tp": tp_daily,
                "fp": fp_daily,
                "fn": fn_daily,
                "tn": tn_daily,
                "precision_pct": round(prec_daily, 2),
                "recall_pct": round(rec_daily, 2),
                "f1_pct": round(f1_daily, 2),
            },
            "hourly_breaches": {
                "true_breaches": int(np.sum(hourly_true_b)),
                "predicted_breaches": int(np.sum(hourly_pred_b)),
                "precision_pct": round(prec_hr, 2),
                "recall_pct": round(rec_hr, 2),
            },
        }

    return model_metrics, p90_diagnostics


def extract_feature_importance(
    model: lgb.LGBMRegressor,
    feature_names: List[str],
) -> pd.DataFrame:
    """Extract feature importance (split and gain) from trained LightGBM model.

    Args:
        model: Trained LightGBM model.
        feature_names: Ordered list of feature column names.

    Returns:
        pd.DataFrame: Feature importance table sorted by gain descending.
    """
    booster = model.booster_
    gains = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")

    df_imp = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_gain": gains,
            "importance_split": splits,
        }
    ).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    return df_imp


def print_comparison_report(
    model_metrics: Dict[str, Dict[str, float]],
    p90_diagnostics: Dict[str, Any],
    top_features_df: pd.DataFrame,
) -> None:
    """Print clean terminal comparison table and top feature importances."""
    print("\n" + "=" * 95)
    print("PEAKWISE AI (HTH-ML-08) - PHASE 2 BACKTEST & MODEL EVALUATION REPORT")
    print("=" * 95)

    header = f"{'MODEL':<24} | {'MAE (kW)':<9} | {'RMSE (kW)':<9} | {'CV-RMSE':<8} | {'MAPE':<7} | {'DAILY-MAX ERR':<13} | {'PEAK RECALL'}"
    print(header)
    print("-" * 95)

    display_names = {
        "seasonal_naive_168": "Seasonal Naive (lag-168)",
        "naive_yesterday_24": "Naive Yesterday (lag-24)",
        "lightgbm_l2": "LightGBM (L2 Reference)",
        "lightgbm_q50": "LightGBM (P50 Quantile)",
        "lightgbm_q10": "LightGBM (P10 Quantile)",
        "lightgbm_q90": "LightGBM (P90 Quantile)",
    }

    for key, metrics in model_metrics.items():
        name = display_names.get(key, key)
        print(
            f"{name:<24} | "
            f"{metrics['mae']:<9.2f} | "
            f"{metrics['rmse']:<9.2f} | "
            f"{metrics['cv_rmse']:<7.2f}% | "
            f"{metrics['mape']:<6.2f}% | "
            f"{metrics['daily_max_error']:<13.2f} | "
            f"{metrics['peak_hour_recall']:<5.1f}%"
        )

    print("-" * 95)

    # P90 Diagnostics
    if p90_diagnostics:
        print("\nP90 QUANTILE & PEAK BREACH DETECTION (PEAK CAP = 147.55 kW):")
        print(f"  • P90 Hourly Coverage: {p90_diagnostics['actual_coverage_pct']}% (Target: ~90%)")
        db = p90_diagnostics["daily_breaches"]
        print(
            f"  • Daily Breach Detection: Precision={db['precision_pct']}%, "
            f"Recall={db['recall_pct']}%, F1={db['f1_pct']}% "
            f"(True Breaches={db['true_breaches']}, Predicted Breaches={db['predicted_breaches']})"
        )

    # Top 15 Features by Gain
    print("\nTOP 15 FEATURES BY GAIN (from P50 Quantile Model):")
    print("-" * 65)
    print(f"{'RANK':<5} | {'FEATURE NAME':<35} | {'IMPORTANCE GAIN':<15}")
    print("-" * 65)
    for i, row in top_features_df.head(15).iterrows():
        print(f"{i+1:<5} | {row['feature']:<35} | {row['importance_gain']:<15.2f}")
    print("-" * 65)
    print("=" * 95 + "\n")


def main() -> None:
    """Execute complete Phase 2 training, backtest, and artifact generation."""
    logger.info("Starting Phase 2 Training Pipeline for BUILDING_ID=%d...", BUILDING_ID)

    # 1. Load Clean Dataset & Generate Feature Pipeline
    clean_df = load_clean_data(building_id=BUILDING_ID)
    features_df, feature_cols = create_feature_pipeline(clean_df, drop_na=True)

    # 2. Chronological Split (Train / Val / Test)
    train_df, val_df, test_df = split_data_chronologically(features_df, test_days=28, val_days=14)

    # 3. Simulate Weather Forecast Uncertainty Noise in Training Split
    train_df = inject_weather_noise(train_df, weather_cols=WEATHER_COLS, seed=42)

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]

    # 4. Train Quantile Models (alpha=0.1, 0.5, 0.9) and L2 Reference Model
    models: Dict[str, lgb.LGBMRegressor] = {
        "q10": train_quantile_model(X_train, y_train, X_val, y_val, alpha=0.1, seed=42),
        "q50": train_quantile_model(X_train, y_train, X_val, y_val, alpha=0.5, seed=42),
        "q90": train_quantile_model(X_train, y_train, X_val, y_val, alpha=0.9, seed=42),
        "l2": train_l2_reference_model(X_train, y_train, X_val, y_val, seed=42),
    }

    # 5. Generate Predictions on Test Set
    predictions = {
        "seasonal_naive_168": test_df["lag_168"].values,
        "naive_yesterday_24": test_df["lag_24"].values,
        "lightgbm_l2": models["l2"].predict(X_test),
        "lightgbm_q50": models["q50"].predict(X_test),
        "lightgbm_q10": models["q10"].predict(X_test),
        "lightgbm_q90": models["q90"].predict(X_test),
        "q90": models["q90"].predict(X_test),  # For p90 diagnostic calculation
    }

    # 6. Execute 24-Hour Rolling-Origin Backtest
    model_metrics, p90_diagnostics = run_rolling_origin_backtest(
        test_df=test_df,
        predictions=predictions,
        peak_cap_kw=PEAK_CAP_KW,
    )

    # 7. Extract Feature Importances from P50 Model
    df_importance = extract_feature_importance(models["q50"], feature_cols)

    # 8. Save Artifacts to models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model artifacts to %s...", MODELS_DIR)

    joblib.dump(models["q10"], MODELS_DIR / "q10.joblib")
    joblib.dump(models["q50"], MODELS_DIR / "q50.joblib")
    joblib.dump(models["q90"], MODELS_DIR / "q90.joblib")
    joblib.dump(models["l2"], MODELS_DIR / "l2.joblib")

    with open(MODELS_DIR / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    metrics_payload = {
        "building_id": BUILDING_ID,
        "test_period": {
            "start": str(test_df[TIMESTAMP_COL].min()),
            "end": str(test_df[TIMESTAMP_COL].max()),
            "hours": len(test_df),
            "days": len(test_df[TIMESTAMP_COL].dt.date.unique()),
        },
        "models": {k: v for k, v in model_metrics.items() if k != "q90"},
        "p90_diagnostics": p90_diagnostics,
    }

    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    df_importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    logger.info("Saved all artifacts: models, feature_list.json, metrics.json, feature_importance.csv.")

    # 9. Print Terminal Summary
    print_comparison_report(
        model_metrics={k: v for k, v in model_metrics.items() if k != "q90"},
        p90_diagnostics=p90_diagnostics,
        top_features_df=df_importance,
    )


if __name__ == "__main__":
    main()
