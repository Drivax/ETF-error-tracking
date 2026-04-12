"""Global project configuration for data, model, and signal settings."""

from pathlib import Path

# Directory settings
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ETF-benchmark mapping for live and historical analysis
PAIR_CONFIGS = {
    "SPY": "^GSPC",
    "QQQ": "^NDX",
    "FEZ": "^STOXX50E",
    "URTH": "ACWI",
}

# Sector mapping for portfolio-level exposure rollups.
ETF_SECTOR_MAP = {
    "SPY": "US Broad Equity",
    "QQQ": "US Technology Growth",
    "FEZ": "Europe Large Cap",
    "URTH": "Global Equity",
}

# Data retrieval defaults
DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"
DEFAULT_INTRADAY_PERIOD = "60d"
DEFAULT_INTRADAY_INTERVAL = "5m"

# Feature and model defaults
DEFAULT_HORIZON = 1
DEFAULT_WINDOW = 20
RANDOM_STATE = 42

# Arbitrage detector defaults
ZSCORE_ENTRY = 2.0
ZSCORE_EXIT = 0.5
ARBITRAGE_WINDOW = 60
VOLATILITY_FILTER_QUANTILE = 0.80

# Artifacts
MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "te_model.joblib"
LATEST_FEATURES_PATH = ARTIFACTS_DIR / "latest_features.csv"
LATEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "latest_predictions.csv"
