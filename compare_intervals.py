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


RUNS = [
    {
        "label": "1d_1y",
        "period": "1y",
        "interval": "1d",
        "holding_bars": 6,
        "min_train_rows": 150,
        "retrain_every": 15,
        "monitor_every": 10,
    },
    {
        "label": "5m_60d",
        "period": "60d",
        "interval": "5m",
        "holding_bars": 12,
        "min_train_rows": 300,
        "retrain_every": 30,
        "monitor_every": 25,
    },
]


def run_one(run_cfg: dict[str, object]) -> dict[str, object]:
    label = str(run_cfg["label"])
    period = str(run_cfg["period"])
    interval = str(run_cfg["interval"])
    print(f"running {label}...", flush=True)

    market_panel, feature_panel = build_market_and_features(
        period=period,
        interval=interval,
        horizon=1,
        rolling_window=20,
    )

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

    artifact_prefix = ARTIFACTS_DIR / f"walk_forward_{label}"
    result["predictions"].to_csv(f"{artifact_prefix}_predictions.csv", index=False)
    result["paper_trades"].to_csv(f"{artifact_prefix}_paper_trades.csv", index=False)
    result["alerts"].to_csv(f"{artifact_prefix}_alerts.csv", index=False)

    summary = {
        "label": label,
        "period": period,
        "interval": interval,
        "kpis": result["kpis"],
        "recalibration": result["recalibration"],
        "alert_count": int(len(result["alerts"])),
        "alert_rate": float(len(result["alerts"]) / max(len(result["predictions"]), 1)),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def pick_winner(summaries: list[dict[str, object]]) -> dict[str, object]:
    def score(summary: dict[str, object]) -> float:
        kpis = summary["kpis"]
        recal = summary["recalibration"]
        return (
            0.40 * float(kpis["trade_precision"])
            + 0.30 * float(kpis["hit_ratio_after_delay"])
            + 0.20 * max(float(kpis["realized_spread_capture"]), 0.0)
            + 0.10 * float(recal["objective_score"])
        )

    ranked = sorted(summaries, key=score, reverse=True)
    winner = dict(ranked[0])
    winner["robustness_score"] = score(ranked[0])
    return winner


def main() -> None:
    summaries = [run_one(run_cfg) for run_cfg in RUNS]
    winner = pick_winner(summaries)

    output = {
        "runs": summaries,
        "winner": winner,
    }
    output_path = Path(ARTIFACTS_DIR) / "interval_comparison_summary.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
