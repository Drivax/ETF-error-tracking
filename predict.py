"""Command-line workflow for batch and real-time ETF tracking error analytics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import (
    ARBITRAGE_WINDOW,
    DEFAULT_HORIZON,
    DEFAULT_INTERVAL,
    DEFAULT_INTRADAY_INTERVAL,
    DEFAULT_INTRADAY_PERIOD,
    DEFAULT_PERIOD,
    DEFAULT_WINDOW,
    MODEL_ARTIFACT_PATH,
    PAIR_CONFIGS,
)
from src.arbitrage_signal import ArbitrageSignalGenerator
from src.data_loader import MarketDataLoader
from src.explainability import TrackingErrorExplainer
from src.features import FeatureEngineer
from src.models import TrackingErrorModel
from src.real_time_predictor import RealTimeTrackingErrorPredictor


def build_market_and_features(
    period: str,
    interval: str,
    horizon: int,
    rolling_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download market panel and build feature panel."""
    loader = MarketDataLoader()
    market_panel = loader.fetch_universe(PAIR_CONFIGS, period=period, interval=interval)

    engineer = FeatureEngineer(rolling_window=rolling_window, horizon=horizon)
    feature_panel = engineer.transform_universe(market_panel)
    return market_panel, feature_panel


def run_training(args: argparse.Namespace) -> None:
    """Train model artifact on historical panel."""
    _, feature_panel = build_market_and_features(
        period=args.lookback_period,
        interval=args.interval,
        horizon=args.horizon,
        rolling_window=args.window,
    )

    model = TrackingErrorModel(random_state=args.random_state)
    result = model.train(feature_panel, target_col="target_te", test_size=args.test_size)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    print("Training complete")
    print(f"Model artifact: {model_path}")
    print(f"Train rows: {result.train_rows}")
    print(f"Test rows: {result.test_rows}")
    for metric, value in result.metrics.items():
        print(f"{metric}: {value:.6f}")


def run_batch_inference(args: argparse.Namespace) -> None:
    """Run standard end-of-period predictions for latest row per pair."""
    model = TrackingErrorModel.load(args.model_path)
    market_panel, feature_panel = build_market_and_features(
        period=args.lookback_period,
        interval=args.interval,
        horizon=args.horizon,
        rolling_window=args.window,
    )

    realtime = RealTimeTrackingErrorPredictor(
        model=model,
        rolling_window=args.window,
        horizon=args.horizon,
    )
    rt_features = realtime.build_feature_panel(market_panel)
    predictions = realtime.predict_latest(rt_features, confidence_level=args.confidence_level)

    if predictions.empty:
        print("No valid rows available for batch inference.")
        return

    print("Latest Predictions")
    print(predictions.to_string(index=False))


def run_realtime_mode(args: argparse.Namespace) -> None:
    """Run full real-time desk workflow in CLI mode.

    Includes intraday predictions, confidence intervals, SHAP diagnostics,
    counterfactual recommendations, and actionable arbitrage signals.
    """
    model = TrackingErrorModel.load(args.model_path)

    loader = MarketDataLoader()
    market_panel = loader.fetch_universe(
        PAIR_CONFIGS,
        period=args.intraday_period,
        interval=args.intraday_interval,
    )

    realtime = RealTimeTrackingErrorPredictor(
        model=model,
        rolling_window=args.window,
        horizon=args.horizon,
    )
    rt_features = realtime.build_feature_panel(market_panel)
    prediction_snapshot = realtime.predict_latest(
        feature_panel=rt_features,
        confidence_level=args.confidence_level,
        confidence_window=120,
    )

    if prediction_snapshot.empty:
        print("No valid rows available for real-time prediction.")
        return

    print("Real-Time Tracking Error Snapshot")
    print(prediction_snapshot.to_string(index=False))

    signal_generator = ArbitrageSignalGenerator(
        confidence_threshold=0.70,
        entry_tracking_error=0.0005,
        max_notional=args.risk_budget_notional,
        min_notional=100_000.0,
        transaction_cost_bps=args.execution_cost_bps,
        slippage_bps=args.slippage_bps,
        persistence_window=max(6, ARBITRAGE_WINDOW // 10),
        liquidity_window=12,
    )

    signals = signal_generator.generate_universe_signals(
        intraday_panel=market_panel,
        prediction_snapshot=prediction_snapshot,
    )

    if signals.empty:
        print("\nNo actionable arbitrage signals.")
    else:
        print("\nArbitrage Signal Panel")
        print(signals.to_string(index=False))

    explain_pair = args.explain_pair
    if explain_pair is None:
        explain_pair = str(prediction_snapshot.iloc[0]["pair"])

    pair_features = rt_features[rt_features["pair"] == explain_pair].dropna()
    if pair_features.empty:
        print(f"\nNo explainability rows available for pair={explain_pair}")
        return

    input_columns = model.numeric_columns + model.categorical_columns
    latest_observation = pair_features.tail(1)[input_columns]

    explainer = TrackingErrorExplainer(model)
    shap_df = explainer.explain_observation(latest_observation)

    print(f"\nTop SHAP Drivers for {explain_pair}")
    top_drivers = shap_df[["feature", "shap_value", "direction"]].head(10)
    print(top_drivers.to_string(index=False))

    counterfactuals = explainer.generate_counterfactuals(
        observation=latest_observation,
        num_counterfactuals=3,
        max_features_to_change=3,
    )

    print("\nCounterfactual Recommendations")
    for scenario in counterfactuals:
        print(f"- {scenario.scenario_name}")
        print(f"  original_prediction: {scenario.original_prediction:.8f}")
        print(f"  counterfactual_prediction: {scenario.counterfactual_prediction:.8f}")
        print(f"  improvement_bps: {scenario.improvement_bps:.2f}")
        print(f"  changed_features: {scenario.changed_features}")
        print(f"  narrative: {scenario.narrative}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ETF tracking error analytics CLI")

    parser.add_argument("--train", action="store_true", help="Train model and save artifact.")
    parser.add_argument("--predict", action="store_true", help="Run standard latest-row predictions.")
    parser.add_argument("--real-time", action="store_true", help="Run intraday real-time desk workflow.")

    parser.add_argument("--model-path", type=str, default=str(MODEL_ARTIFACT_PATH), help="Model artifact path.")
    parser.add_argument("--lookback-period", type=str, default=DEFAULT_PERIOD, help="Historical lookback for train/predict.")
    parser.add_argument("--interval", type=str, default=DEFAULT_INTERVAL, help="Bar interval for train/predict.")

    parser.add_argument("--intraday-period", type=str, default=DEFAULT_INTRADAY_PERIOD, help="Intraday lookback period for --real-time.")
    parser.add_argument("--intraday-interval", type=str, default=DEFAULT_INTRADAY_INTERVAL, help="Intraday bar interval for --real-time.")

    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="Forecast horizon in bars.")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Feature rolling window.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Temporal test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--confidence-level", type=float, default=0.95, help="Confidence level for prediction intervals.")

    parser.add_argument("--entry-zscore", type=float, default=2.0, help="Base entry threshold before calibration.")
    parser.add_argument("--exit-zscore", type=float, default=0.5, help="Exit threshold used in spread capture estimate.")
    parser.add_argument("--execution-cost-bps", type=float, default=3.0, help="Execution cost in basis points.")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Slippage assumption in basis points.")
    parser.add_argument("--holding-bars", type=int, default=6, help="Expected holding period for arbitrage in bars.")
    parser.add_argument("--risk-budget-notional", type=float, default=1_000_000.0, help="Risk budget for action sizing.")
    parser.add_argument("--explain-pair", type=str, default=None, help="Specific pair for SHAP and counterfactual output.")

    args = parser.parse_args()
    if not any([args.train, args.predict, args.real_time]):
        parser.error("At least one action is required: --train and/or --predict and/or --real-time")
    return args


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()

    if args.train:
        run_training(args)
    if args.predict:
        run_batch_inference(args)
    if args.real_time:
        run_realtime_mode(args)


if __name__ == "__main__":
    main()
