# PeakWise AI: Peak-Constrained Energy Optimization Advisor (HTH-ML-08)

An end-to-end Machine Learning and Mathematical Optimization decision support system designed to minimize commercial/industrial electricity peak demand charges. 

PeakWise AI predicts 24-hour ahead hourly electricity demand with probabilistic confidence bounds, detects likely threshold breaches, recommends an optimal physical load-shift schedule using Linear Programming, and independently audits all physical and operational constraints while quantifying tariff cost savings.

---

## 1. Problem Statement

Commercial and industrial facilities frequently face steep **demand charges** (often $15–$35/kW each month) assessed based on their single highest 15-to-60-minute power spike. Exceeding a contract peak cap triggers penalty surcharges and permanently raises monthly billing baselines.

Traditional rule-based building energy management systems (BEMS) operate reactively or blindly shed load, disrupting critical processes. **PeakWise AI** provides proactive, risk-aware advisory:
1. **Forecasts** hourly electricity consumption 24 hours ahead using LightGBM quantile regression ($P_{10}, P_{50}, P_{90}$).
2. **Quantifies** continuous peak breach probability via piecewise quantile CDF interpolation.
3. **Disaggregates** loads into shiftable and non-shiftable components.
4. **Optimizes** an hourly dispatch plan with Google OR-Tools Linear Programming (GLOP) to shave peaks below the cap while minimizing Time-of-Use (TOU) energy costs and thermodynamic rebound penalties.
5. **Audits** the final dispatch plan using an independent validator (zero tolerance).
6. **Delivers** an interactive Streamlit operations center with live weather-responsive re-forecasting.

---

## 2. System Architecture

```
                                  +---------------------------------------+
                                  |   ASHRAE Great Energy Predictor III   |
                                  |  (Building 1301 - Electricity Meter)  |
                                  +---------------------------------------+
                                                      |
                                                      v
                                        +---------------------------+
                                        |      src/features.py      |
                                        | (Leakage-Free 24h Lags &  |
                                        | Weather Thermodynamics)   |
                                        +---------------------------+
                                                      |
                                                      v
                                        +---------------------------+
                                        |      src/forecast.py      |
                                        | (LightGBM Quantiles:      |
                                        |  P10, P50, P90 Models)    |
                                        +---------------------------+
                                                      |
                                     +----------------+----------------+
                                     |                                 |
                                     v                                 v
                       +---------------------------+     +---------------------------+
                       |        src/peak.py        |     |        src/loads.py       |
                       |  (Continuous Breach Risk  |     |   (Domain Disaggregation: |
                       |   CDF & Alerts Engine)    |     |    Shiftable vs. Fixed)   |
                       +---------------------------+     +---------------------------+
                                     |                                 |
                                     +----------------+----------------+
                                                      |
                                                      v
                                        +---------------------------+
                                        |     src/optimizer.py      |
                                        | (OR-Tools Two-Stage LP:   |
                                        |  Stage 1: Min Peak P*     |
                                        |  Stage 2: Cost & Friction)|
                                        +---------------------------+
                                                      |
                                     +----------------+----------------+
                                     |                                 |
                                     v                                 v
                       +---------------------------+     +---------------------------+
                       |     src/validator.py      |     |     src/economics.py      |
                       |  (Independent Constraint  |     |   (Demand Charge Savings, |
                       |   Audit: Zero Tolerance)  |     |   Avoided Penalties, TOU) |
                       +---------------------------+     +---------------------------+
                                     |                                 |
                                     +----------------+----------------+
                                                      |
                                     +----------------+----------------+
                                     |                                 |
                                     v                                 v
                       +---------------------------+     +---------------------------+
                       |        run_demo.py        |     |   app/streamlit_app.py    |
                       |   (CLI 6-Step Pipeline &  |     | (Interactive Dashboard &  |
                       |   Realized 28d Backtest)  |     | Live Weather Re-Forecast) |
                       +---------------------------+     +---------------------------+
```

---

## 3. Key Assumptions & Domain Configuration

All assumptions and parameters are maintained in [`src/config.py`](src/config.py) as the single source of truth:

1. **Facility Profile**:
   * **Target Facility**: ASHRAE Building 1301 (`primary_use: Education`, `square_feet: 62,242`, `site_id: 14`).
   * **Meter**: Meter 0 (Electricity).
   * **Peak Cap ($PEAK\_CAP\_KW$)**: Default is dynamically computed from historical data as the 92nd percentile of daily maximum loads (**$147.55\text{ kW}$**), adjustable via CLI or UI.

2. **Load Disaggregation (Explicit Engineering Assumption)**:
   * *The ASHRAE Great Energy Predictor III dataset records whole-building electricity without submetering.*
   * **Shiftable Load Share ($20\% - 45\%$)**: HVAC chillers, ventilation precooling/setback, process pumps, and EV/forklift battery charging. Peak daytime share reaches $45\%$ during working class hours.
   * **Non-Shiftable / Fixed Base Load ($55\% - 80\%$)**: Critical manufacturing machines, rubber molding presses, curing equipment, baseline safety systems, and IT servers.
   * Fixed load is strictly untouched during all optimization steps.

3. **Operational Constraints**:
   * **Max Deferral Horizon**: $+3\text{ hours}$ forward ($t \le h + 3$).
   * **Max Precooling Horizon**: $-3\text{ hours}$ backward ($t \ge h - 3$).
   * **Max Shift Fraction Per Hour**: Up to $50\%$ of an hour's shiftable load may be moved.
   * **Thermodynamic Rebound Overhead**: $1.05\times$ ($5\%$ energy efficiency penalty on shifted energy).

4. **Economic Tariff Structure**:
   * **Monthly Demand Charge**: $\$15.00\text{ / kW-month}$ (prorated to daily equivalent across 30 days: $\$0.50\text{ / kW-day}$).
   * **Breach Penalty**: $\$35.00\text{ / kW}$ on peak demand exceeding the contract cap.
   * **Time-of-Use (TOU) Rates**: Peak hours (12:00 PM – 6:00 PM) at $\$0.18\text{ / kWh}$; off-peak hours at $\$0.08\text{ / kWh}$.
   * **Baseline Volumetric Rate**: $\$0.08\text{ / kWh}$.

---

## 4. How to Run

### Environment Setup
Python 3.11 with pip:
```bash
pip install -r requirements.txt
```

### 1. Train Quantile Forecast Models
Runs rolling-origin time-based validation, saves LightGBM models to `models/`, and generates `models/metrics.json`:
```bash
python train.py
```

### 2. Run CLI Advisor Demonstration
Run a single-day 6-step advisory execution:
```bash
# Known breach day with calibrated cap:
python run_demo.py --date 2016-12-15 --cap 135.0

# No-breach day (baseline operates safely under cap):
python run_demo.py --date 2016-12-15 --cap 147.55

# Infeasible forced-cap scenario (gracefully reports P* and diagnostic cause):
python run_demo.py --date 2016-12-15 --cap 80.0

# Live heatwave sensitivity (+5 deg C):
python run_demo.py --date 2016-12-15 --cap 135.0 --heatwave 5.0
```

### 3. Run Rolling Backtest Across Realized Actuals
Evaluates the full 28-day test set (Dec 4 – Dec 31, 2016), planning strictly from the forecast and dispatching against unmetered actuals:
```bash
python run_demo.py --backtest --cap 135.0
```

### 4. Launch Interactive Streamlit Application
Launches the full interactive decision support advisor:
```bash
streamlit run app/streamlit_app.py
```

---

## 5. Model Validation & Benchmark Metrics

Evaluated across the 28-day holdout test period (`2016-12-04 00:00:00` to `2016-12-31 23:00:00`, 672 consecutive hourly steps):

| Model Architecture | Objective | MAE (kW) | RMSE (kW) | CV-RMSE (%) | Daily Max Error (kW) | Peak-Hour Recall (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LightGBM Point (L2)** | Mean Regression | **6.34** | **8.21** | **6.88%** | 10.91 | **42.86%** |
| **LightGBM Quantile (Q50)** | Median Regression | 6.62 | 8.53 | 7.16% | 11.82 | 35.71% |
| **LightGBM Quantile (Q90)** | Upper Risk Bound | 9.19 | 11.24 | 9.42% | **8.05** | 21.43% |
| **Yesterday Naive (Lag 24)**| 24h Baseline | 8.46 | 11.69 | 9.80% | 8.09 | 17.86% |
| **Seasonal Naive (Lag 168)**| 7-Day Baseline | 8.80 | 12.35 | 10.35% | 10.95 | 28.57% |

*LightGBM achieves a **6.88% CV-RMSE**, substantially outperforming the ASHRAE Guideline 14 industry threshold of $< 25\%$ for hourly calibrated models.*

### 28-Day Realized Backtest Performance (Cap = 135 kW)
* **Total Days Evaluated**: 28 Days
* **Days with Realized Breaches**: 13 Days
* **Total Realized Net Savings**: **$1,186.55**
* **Average Daily Realized Savings**: **$42.38 / day**
* **Annualized Realized Extrapolation**: **$15,467.53 / year**

---

## 6. Limitations & Future Roadmap

1. **Submetering Data**: Submetered data would replace the engineering shiftable-load profile assumptions with empirical HVAC/chiller circuit measurements.
2. **Thermal Inertia Simulation**: Integrating an EnergyPlus or RC equivalent thermal circuit would allow dynamic indoor comfort verification during precooling and setback.
3. **Multi-Facility Coordination**: Extending the OR-Tools LP formulation from a single facility to an aggregated virtual power plant (VPP) portfolio.
4. **Stochastic Rolling-Horizon Dispatch**: Transitioning from a 24-hour day-ahead static solve to an hourly rolling MPC (Model Predictive Control) horizon.