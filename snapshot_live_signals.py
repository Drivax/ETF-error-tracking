from __future__ import annotations

import json
from pathlib import Path

from config import (
    ARBITRAGE_WINDOW,
    ARTIFACTS_DIR,
    DEFAULT_HORIZON,
    DEFAULT_WINDOW,
    MODEL_ARTIFACT_PATH,
    PAIR_CONFIGS,
    REALTIME_CONFIDENCE_THRESHOLD,
    REALTIME_ENTRY_TRACKING_ERROR,
    REALTIME_MIN_EXPECTED_PROFIT_BPS,
)
from src.arbitrage_signal import ArbitrageSignalGenerator
from src.data_loader import MarketDataLoader
from src.models import TrackingErrorModel
from src.real_time_predictor import RealTimeTrackingErrorPredictor


def main() -> None:
    model = TrackingErrorModel.load(MODEL_ARTIFACT_PATH)
    loader = MarketDataLoader()
    market_panel = loader.fetch_universe(PAIR_CONFIGS, period="3d", interval="5m")

    realtime = RealTimeTrackingErrorPredictor(
        model=model,
        rolling_window=DEFAULT_WINDOW,
        horizon=DEFAULT_HORIZON,
    )
    rt_features = realtime.build_feature_panel(market_panel)
    prediction_snapshot = realtime.predict_latest(
        feature_panel=rt_features,
        confidence_level=0.95,
        confidence_window=120,
    )

    signal_generator = ArbitrageSignalGenerator(
        confidence_threshold=REALTIME_CONFIDENCE_THRESHOLD,
        entry_tracking_error=REALTIME_ENTRY_TRACKING_ERROR,
        min_expected_profit_bps=REALTIME_MIN_EXPECTED_PROFIT_BPS,
        max_notional=1_000_000.0,
        min_notional=100_000.0,
        transaction_cost_bps=3.0,
        slippage_bps=2.0,
        persistence_window=max(6, ARBITRAGE_WINDOW // 10),
        liquidity_window=12,
    )
    signals = signal_generator.generate_universe_signals(
        intraday_panel=market_panel,
        prediction_snapshot=prediction_snapshot,
    )

    csv_path = Path(ARTIFACTS_DIR) / "live_threshold_snapshot.csv"
    json_path = Path(ARTIFACTS_DIR) / "live_threshold_snapshot.json"
    prediction_snapshot.to_csv(Path(ARTIFACTS_DIR) / "live_prediction_snapshot.csv", index=False)
    signals.to_csv(csv_path, index=False)

    payload = {
        "confidence_threshold": REALTIME_CONFIDENCE_THRESHOLD,
        "entry_tracking_error": REALTIME_ENTRY_TRACKING_ERROR,
        "min_expected_profit_bps": REALTIME_MIN_EXPECTED_PROFIT_BPS,
        "signal_count": int(len(signals)),
        "actionable_count": int((signals["action"] != "HOLD").sum()) if not signals.empty else 0,
        "signals_path": str(csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
