from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

from config import ARTIFACTS_DIR
from predict import build_market_and_features
from src.walk_forward import WalkForwardPaperTrader

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

status_path = Path(ARTIFACTS_DIR) / "intraday_short_status.json"
summary_path = Path(ARTIFACTS_DIR) / "intraday_short_summary.json"


def write_status(payload: dict[str, object]) -> None:
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    run_cfg = {
        "label": "5m_3d",
        "period": "3d",
        "interval": "5m",
        "holding_bars": 12,
        "min_train_rows": 120,
        "retrain_every": 120,
        "monitor_every": 200,
    }
    write_status({"stage": "building", "current": run_cfg})

    market_panel, feature_panel = build_market_and_features(
        period=run_cfg["period"],
        interval=run_cfg["interval"],
        horizon=1,
        rolling_window=20,
    )

    write_status(
        {
            "stage": "running",
            "current": {
                **run_cfg,
                "market_rows": int(len(market_panel)),
                "feature_rows": int(len(feature_panel)),
            },
        }
    )

    runner = WalkForwardPaperTrader(
        model_random_state=42,
        confidence_threshold=0.70,
        entry_tracking_error=0.0005,
        max_notional=1_000_000.0,
        min_notional=100_000.0,
        transaction_cost_bps=3.0,
        slippage_bps=2.0,
        min_train_rows=run_cfg["min_train_rows"],
        retrain_every=run_cfg["retrain_every"],
        execution_delay_bars=1,
        holding_bars=run_cfg["holding_bars"],
        drift_window=80,
        retrain_mae_ratio_trigger=1.45,
        retrain_mean_shift_trigger_sigma=2.5,
        monitor_every=run_cfg["monitor_every"],
    )
    result = runner.run(market_panel=market_panel, feature_panel=feature_panel, target_col="target_te")

    artifact_prefix = Path(ARTIFACTS_DIR) / "walk_forward_5m_3d"
    result["predictions"].to_csv(f"{artifact_prefix}_predictions.csv", index=False)
    result["paper_trades"].to_csv(f"{artifact_prefix}_paper_trades.csv", index=False)
    result["alerts"].to_csv(f"{artifact_prefix}_alerts.csv", index=False)

    summary = {
        "label": run_cfg["label"],
        "period": run_cfg["period"],
        "interval": run_cfg["interval"],
        "kpis": result["kpis"],
        "recalibration": result["recalibration"],
        "alert_count": int(len(result["alerts"])),
        "alert_rate": float(len(result["alerts"]) / max(len(result["predictions"]), 1)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_status({"stage": "finished", "summary": summary})


if __name__ == "__main__":
    main()
