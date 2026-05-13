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

STATUS_PATH = Path(ARTIFACTS_DIR) / "interval_comparison_status.json"
SUMMARY_PATH = Path(ARTIFACTS_DIR) / "interval_comparison_summary.json"

RUNS = [
    {
        "label": "1d_6mo",
        "period": "6mo",
        "interval": "1d",
        "holding_bars": 6,
        "min_train_rows": 120,
        "retrain_every": 20,
        "monitor_every": 20,
    },
    {
        "label": "5m_15d",
        "period": "15d",
        "interval": "5m",
        "holding_bars": 12,
        "min_train_rows": 180,
        "retrain_every": 60,
        "monitor_every": 100,
    },
]


def write_status(payload: dict[str, object]) -> None:
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def score_summary(summary: dict[str, object]) -> float:
    kpis = summary["kpis"]
    recal = summary["recalibration"]
    return (
        0.40 * float(kpis["trade_precision"])
        + 0.30 * float(kpis["hit_ratio_after_delay"])
        + 0.20 * max(float(kpis["realized_spread_capture"]), 0.0)
        + 0.10 * float(recal["objective_score"])
    )


def run_one(run_cfg: dict[str, object], completed: list[dict[str, object]]) -> dict[str, object]:
    label = str(run_cfg["label"])
    write_status({
        "stage": f"building_{label}",
        "completed": completed,
        "current": run_cfg,
    })

    market_panel, feature_panel = build_market_and_features(
        period=str(run_cfg["period"]),
        interval=str(run_cfg["interval"]),
        horizon=1,
        rolling_window=20,
    )

    write_status({
        "stage": f"running_{label}",
        "completed": completed,
        "current": {
            **run_cfg,
            "market_rows": int(len(market_panel)),
            "feature_rows": int(len(feature_panel)),
        },
    })

    runner = WalkForwardPaperTrader(
        model_random_state=42,
        confidence_threshold=0.70,
        entry_tracking_error=0.0005,
        max_notional=1_000_000.0,
        min_notional=100_000.0,
        transaction_cost_bps=3.0,
        slippage_bps=2.0,
        min_train_rows=int(run_cfg["min_train_rows"]),
        retrain_every=int(run_cfg["retrain_every"]),
        execution_delay_bars=1,
        holding_bars=int(run_cfg["holding_bars"]),
        drift_window=80,
        retrain_mae_ratio_trigger=1.45,
        retrain_mean_shift_trigger_sigma=2.5,
        monitor_every=int(run_cfg["monitor_every"]),
    )

    result = runner.run(market_panel=market_panel, feature_panel=feature_panel, target_col="target_te")

    artifact_prefix = Path(ARTIFACTS_DIR) / f"walk_forward_{label}"
    result["predictions"].to_csv(f"{artifact_prefix}_predictions.csv", index=False)
    result["paper_trades"].to_csv(f"{artifact_prefix}_paper_trades.csv", index=False)
    result["alerts"].to_csv(f"{artifact_prefix}_alerts.csv", index=False)

    summary = {
        "label": label,
        "period": str(run_cfg["period"]),
        "interval": str(run_cfg["interval"]),
        "kpis": result["kpis"],
        "recalibration": result["recalibration"],
        "alert_count": int(len(result["alerts"])),
        "alert_rate": float(len(result["alerts"]) / max(len(result["predictions"]), 1)),
        "robustness_score": 0.0,
    }
    summary["robustness_score"] = score_summary(summary)
    completed.append(summary)

    write_status({
        "stage": f"completed_{label}",
        "completed": completed,
        "current": None,
    })
    return summary


def main() -> None:
    completed: list[dict[str, object]] = []
    write_status({"stage": "starting", "completed": completed, "current": None})

    for run_cfg in RUNS:
        run_one(run_cfg, completed)

    ranked = sorted(completed, key=lambda item: float(item["robustness_score"]), reverse=True)
    summary = {
        "runs": completed,
        "winner": ranked[0],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_status({"stage": "finished", "completed": completed, "winner": ranked[0]})


if __name__ == "__main__":
    main()
