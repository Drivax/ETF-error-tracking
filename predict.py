"""Command-line training and inference workflow for ETF tracking error prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    ARBITRAGE_WINDOW,
    DEFAULT_HORIZON,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    DEFAULT_WINDOW,
    MODEL_ARTIFACT_PATH,
    PAIR_CONFIGS,
    VOLATILITY_FILTER_QUANTILE,
    ZSCORE_ENTRY,
    ZSCORE_EXIT,
)
from src.arbitrage_detector import ArbitrageDetector
from src.data_loader import MarketDataLoader
from src.features import FeatureEngineer
from src.models import TrackingErrorModel


def build_modeling_dataset(period: str, interval: str, horizon: int, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download data and return raw panel plus engineered feature set."""
    loader = MarketDataLoader()
    panel = loader.fetch_universe(PAIR_CONFIGS, period=period, interval=interval)

    engineer = FeatureEngineer(rolling_window=window, horizon=horizon)
    feature_df = engineer.transform_universe(panel)
    return panel, feature_df


def run_training(args: argparse.Namespace) -> None:
    """Train model and save artifact to disk."""
    _, feature_df = build_modeling_dataset(
        period=args.lookback_period,
        interval=args.interval,
        horizon=args.horizon,
        window=args.window,
    )

    model = TrackingErrorModel(random_state=args.random_state)
    result = model.train(feature_df, target_col="target_te", test_size=args.test_size)

    artifact_path = Path(args.model_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(artifact_path)

    print("Training complete")
    print(f"Model artifact: {artifact_path}")
    print(f"Train rows: {result.train_rows}")
    print(f"Test rows: {result.test_rows}")
    for metric, value in result.metrics.items():
        print(f"{metric}: {value:.6f}")


def run_inference(args: argparse.Namespace) -> None:
    """Load model, score latest observations, and print signal table."""
    model = TrackingErrorModel.load(args.model_path)

    panel, feature_df = build_modeling_dataset(
        period=args.lookback_period,
        interval=args.interval,
        horizon=args.horizon,
        window=args.window,
    )

    detector = ArbitrageDetector(
        window=ARBITRAGE_WINDOW,
        zscore_entry=ZSCORE_ENTRY,
        zscore_exit=ZSCORE_EXIT,
        volatility_filter_quantile=VOLATILITY_FILTER_QUANTILE,
    )

    rows: list[dict[str, str | float]] = []
    for pair_name, pair_features in feature_df.groupby("pair", observed=True):
        latest = pair_features.dropna().tail(1).copy()
        if latest.empty:
            continue

        prediction = float(model.predict(latest[model.numeric_columns + model.categorical_columns])[0])

        pair_panel = panel[panel["pair"] == pair_name]
        signal_snapshot = detector.latest_signal(pair_panel)

        row = {
            "pair": pair_name,
            "predicted_tracking_error": prediction,
            "signal": signal_snapshot["signal"],
            "spread_zscore": signal_snapshot["spread_zscore"],
            "spread_vol": signal_snapshot["spread_vol"],
            "half_life": signal_snapshot["half_life"],
        }
        rows.append(row)

    if not rows:
        print("No valid rows available for inference.")
        return

    result_df = pd.DataFrame(rows).sort_values("predicted_tracking_error", ascending=False)
    print(result_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training and inference workflows."""
    parser = argparse.ArgumentParser(description="ETF tracking error prediction CLI")

    parser.add_argument("--train", action="store_true", help="Train model and save artifact.")
    parser.add_argument("--predict", action="store_true", help="Load model and run inference.")
    parser.add_argument("--model-path", type=str, default=str(MODEL_ARTIFACT_PATH), help="Path to model artifact.")
    parser.add_argument("--lookback-period", type=str, default=DEFAULT_PERIOD, help="yfinance lookback period.")
    parser.add_argument("--interval", type=str, default=DEFAULT_INTERVAL, help="yfinance interval.")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Forecast horizon steps.")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Rolling window for TE features.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Temporal test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    if not args.train and not args.predict:
        parser.error("At least one action is required: --train and/or --predict")
    return args


def main() -> None:
    """Main CLI entrypoint."""
    args = parse_args()
    if args.train:
        run_training(args)
    if args.predict:
        run_inference(args)


if __name__ == "__main__":
    main()
