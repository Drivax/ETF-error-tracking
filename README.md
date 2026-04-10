# ETF Tracking Error Prediction & Arbitrage Detection

## Project Objective
This project builds an end-to-end machine learning system to predict ETF tracking error against benchmark indices and detect potential statistical arbitrage opportunities in near real time.

The platform is designed for bank risk management and trading workflows. It includes:
- Robust market data collection for ETFs and benchmarks.
- Feature engineering for daily and intraday tracking behavior.
- A predictive model for forward tracking error.
- A signal engine for abnormal ETF-index dislocations.
- Explainability outputs with SHAP and counterfactual analysis.
- A Streamlit dashboard for analysts and risk managers.

## Business & Regulatory Context
Tracking error is a core risk indicator for index-tracking products. In a bank environment, it matters for:
- Product governance: verifying that ETF behavior remains consistent with mandate and disclosures.
- Market risk: monitoring sudden divergence from benchmark dynamics.
- Trading surveillance: detecting temporary dislocations that can become arbitrage opportunities.
- Client reporting: explaining observed deviations with transparent drivers.

From a regulatory perspective, the pipeline supports model risk management principles:
- Traceability: deterministic preprocessing and explicit model artifacts.
- Explainability: SHAP feature attributions and what-if counterfactuals.
- Monitoring readiness: clear metrics and threshold-based signal logic.
- Auditability: reproducible notebooks and modular source code.

## Dataset
### Data Sources
Data is retrieved from Yahoo Finance through `yfinance`, using real listed instruments and benchmark proxies:
- `SPY` vs `^GSPC` (S&P 500 index)
- `QQQ` vs `^NDX` (Nasdaq-100 index)
- `FEZ` vs `^STOXX50E` (Euro Stoxx 50)
- `URTH` vs `ACWI` (MSCI World proxy)

Daily and intraday bars are supported.

### Features Collected
For each ETF-benchmark pair:
- Open, High, Low, Close, Volume
- Arithmetic and log returns
- Forward return alignment for target construction
- Relative spread and rolling volatility descriptors

### Preprocessing
Main preprocessing steps:
- Timestamp alignment between ETF and benchmark.
- Missing value handling with forward fill and conservative drop rules.
- Return computation after sorted, timezone-safe index normalization.
- Pair-wise panel assembly with explicit `pair` identifier.

## Methodology
### Feature Engineering
The feature pipeline is implemented in `src/features.py` and includes:
- Lagged ETF and benchmark returns.
- Rolling volatility (ETF, benchmark, and difference).
- Rolling beta estimate from covariance/variance.
- Price ratio and ratio z-score.
- Rolling tracking error statistics.
- Volume pressure and micro-trend indicators.

Feature windows are intentionally heterogeneous (short, medium, long) to capture both short-lived dislocations and persistent drift.

### Model Architecture
The prediction model is implemented in `src/models.py`:
- Main estimator: `HistGradientBoostingRegressor`.
- Input: engineered features at time $t$.
- Output: expected tracking error over horizon $h$.
- Scaling: robust scaling to reduce sensitivity to heavy tails.
- Validation: time-based split to respect causality.

Why this architecture:
- Fast training and inference.
- Strong non-linear fit without deep learning complexity.
- Compatible with SHAP explainability.

### Tracking Error Modeling
The model predicts forward realized tracking error estimated from ETF-benchmark return differences. The target can be configured for daily or intraday horizons.

The framework supports:
- Pair-specific modeling.
- Unified cross-pair modeling (single model with `pair` feature encoding).
- Rolling retrain process for production refresh.

### Arbitrage Detection Logic
Arbitrage detection is implemented in `src/arbitrage_detector.py` and combines:
- Residual dislocation: observed spread minus expected spread.
- Rolling z-score thresholds.
- Volatility regime filter.
- Optional mean-reversion half-life estimate.

Signal categories:
- `NORMAL`
- `WATCH`
- `ARBITRAGE_LONG_ETF_SHORT_BENCH`
- `ARBITRAGE_SHORT_ETF_LONG_BENCH`

This design is intentionally interpretable. Each signal can be decomposed into quantitative threshold checks.

## Key Equations
### Return Definitions
$$
r_t = \frac{P_t}{P_{t-1}} - 1
$$
This is the arithmetic return of an asset from $t-1$ to $t$.

$$
\ell_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$
This is the log return, useful for additive time aggregation.

### Instantaneous Tracking Difference
$$
d_t = r^{ETF}_t - r^{IDX}_t
$$
This measures one-period return divergence between ETF and benchmark.

### Realized Tracking Error (Rolling)
$$
TE_t^{(w)} = \sqrt{\frac{1}{w-1}\sum_{i=t-w+1}^{t} \left(d_i - \bar{d}_{t,w}\right)^2}
$$
This is the rolling standard deviation of return differences over window $w$.

### Forward Target Construction
$$
y_t = TE_{t+h}^{(w)}
$$
The supervised target is the tracking error observed at horizon $h$ in the future.

### Rolling Beta
$$
\beta_t^{(w)} = \frac{\operatorname{Cov}_w\left(r^{ETF}, r^{IDX}\right)}{\operatorname{Var}_w\left(r^{IDX}\right) + \varepsilon}
$$
This quantifies ETF sensitivity to benchmark moves in a rolling window.

### Spread Z-Score for Arbitrage
$$
z_t = \frac{s_t - \mu_t^{(w)}}{\sigma_t^{(w)} + \varepsilon}
$$
Here $s_t$ is a spread proxy (for example log-price ratio). Large $|z_t|$ suggests abnormal dislocation.

### Mean-Reversion Half-Life
Given an AR(1) estimate $x_t = \phi x_{t-1} + \eta_t$,
$$
	ext{HalfLife} = -\frac{\ln(2)}{\ln(|\phi|)}
$$
This approximates time needed for half of a shock to decay.

## Evaluation Metrics & Results
Model quality is evaluated with:
- MAE for absolute error magnitude.
- RMSE for tail-sensitive error.
- $R^2$ for explained variance.
- MAPE (with safe denominator) for relative error context.

Signal engine quality can be monitored with:
- Precision of arbitrage flags against realized mean-reversion outcomes.
- Hit ratio under execution delay assumptions.
- Average normalized spread capture.

Typical expected behavior (instrument and period dependent):
- Higher accuracy in stable volatility regimes.
- Wider forecast errors during stress or opening auction intervals.
- Better signal precision when z-score is combined with volatility filter.

## Repository Structure
```text
ETF-error-tracking/
├─ app.py
├─ config.py
├─ main.py
├─ predict.py
├─ README.md
├─ requirements.txt
├─ notebooks/
│  ├─ 01_data_collection.ipynb
│  ├─ 02_feature_engineering.ipynb
│  ├─ 03_model_training.ipynb
│  └─ 04_results_and_evaluation.ipynb
└─ src/
	├─ __init__.py
	├─ arbitrage_detector.py
	├─ data_loader.py
	├─ features.py
	├─ models.py
	└─ utils.py
```

## Installation and Execution
### 1) Create and activate environment
```bash
python -m venv .venv
```

Windows PowerShell:
```bash
.venv\Scripts\Activate.ps1
```

Linux/macOS:
```bash
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Train and persist model artifact
```bash
python predict.py --train --lookback-period 2y --interval 1d --horizon 1 --window 20
```

### 4) Run batch prediction from saved model
```bash
python predict.py --predict --model-path artifacts/te_model.joblib --lookback-period 6mo --interval 1d
```

### 5) Launch Streamlit dashboard
```bash
streamlit run app.py
```

### 6) Run notebooks
Open notebooks in order:
1. `notebooks/01_data_collection.ipynb`
2. `notebooks/02_feature_engineering.ipynb`
3. `notebooks/03_model_training.ipynb`
4. `notebooks/04_results_and_evaluation.ipynb`

## What This Project Demonstrates
This project demonstrates how to translate a market microstructure problem into a production-ready quantitative risk pipeline. It combines real market data ingestion, robust feature construction, explainable machine learning, and explicit arbitrage signal rules in a way that is auditable, fast to operate, and understandable by both model validators and trading stakeholders. The result is a practical blueprint for ETF tracking surveillance and dislocation monitoring in a banking environment.
