#!/usr/bin/env python
"""Quick test of walk-forward execution with debug output."""

import sys
print("Starting walk-forward debug test...", file=sys.stderr, flush=True)

from config import PAIR_CONFIGS, ARTIFACTS_DIR
from pathlib import Path

print("Loading data loader...", file=sys.stderr, flush=True)
from src.data_loader import MarketDataLoader
loader = MarketDataLoader()

print("Fetching market data (1y, 1d)...", file=sys.stderr, flush=True)
market_panel = loader.fetch_universe(PAIR_CONFIGS, period="1y", interval="1d")
print(f"Market panel shape: {market_panel.shape}", file=sys.stderr, flush=True)

print("Engineering features...", file=sys.stderr, flush=True)
from src.features import FeatureEngineer
engineer = FeatureEngineer(rolling_window=20, horizon=1)
feature_panel = engineer.transform_universe(market_panel)
print(f"Feature panel shape: {feature_panel.shape}", file=sys.stderr, flush=True)

print("Initializing walk-forward runner...", file=sys.stderr, flush=True)
from src.walk_forward import WalkForwardPaperTrader
runner = WalkForwardPaperTrader(
    model_random_state=42,
    confidence_threshold=0.70,
    entry_tracking_error=0.0005,
    max_notional=1_000_000.0,
    min_notional=100_000.0,
    transaction_cost_bps=3.0,
    slippage_bps=2.0,
    min_train_rows=150,
    retrain_every=15,
    execution_delay_bars=1,
    holding_bars=6,
)

print("Running walk-forward...", file=sys.stderr, flush=True)
try:
    result = runner.run(market_panel=market_panel, feature_panel=feature_panel, target_col="target_te")
    print("Walk-forward completed!", file=sys.stderr, flush=True)
    
    print("\nWalk-Forward KPIs")
    for metric, value in result["kpis"].items():
        if isinstance(value, float):
            print(f"{metric}: {value:.6f}")
        else:
            print(f"{metric}: {value}")
    
    print("\nThreshold Recalibration")
    for key, value in result["recalibration"].items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    
    # Save artifacts
    output_prefix = ARTIFACTS_DIR / "walk_forward_1d_1y"
    result["predictions"].to_csv(f"{output_prefix}_predictions.csv", index=False)
    result["paper_trades"].to_csv(f"{output_prefix}_paper_trades.csv", index=False)
    result["alerts"].to_csv(f"{output_prefix}_alerts.csv", index=False)
    
    print(f"\nArtifacts saved to {ARTIFACTS_DIR}")
    
except Exception as e:
    print(f"Error: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
