"""Walk-forward backtesting and paper-trading simulation for ETF arbitrage signals.

This module evaluates signal quality on unseen periods by iteratively retraining
on history and simulating delayed execution on future bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.arbitrage_signal import ArbitrageSignalGenerator
from src.models import TrackingErrorModel
from src.regime_detector import RegimeDetector


@dataclass(frozen=True)
class ThresholdRecommendation:
    """Recommended execution thresholds derived from forward simulation."""

    confidence_threshold: float
    min_expected_profit_bps: float
    objective_score: float
    evaluated_trades: int


class WalkForwardPaperTrader:
    """Walk-forward evaluator and paper-trading engine.

    The engine trains on an expanding window, predicts on unseen timestamps,
    simulates delayed execution, computes trading KPIs, and emits retrain alerts.
    """

    def __init__(
        self,
        model_random_state: int = 42,
        confidence_threshold: float = 0.70,
        entry_tracking_error: float = 0.0005,
        max_notional: float = 1_000_000.0,
        min_notional: float = 100_000.0,
        transaction_cost_bps: float = 3.0,
        slippage_bps: float = 2.0,
        min_train_rows: int = 400,
        retrain_every: int = 20,
        execution_delay_bars: int = 1,
        holding_bars: int = 6,
        drift_window: int = 80,
        retrain_mae_ratio_trigger: float = 1.45,
        retrain_mean_shift_trigger_sigma: float = 2.5,
    ) -> None:
        if min_train_rows < 120:
            raise ValueError("min_train_rows must be at least 120")
        if retrain_every < 1:
            raise ValueError("retrain_every must be >= 1")
        if execution_delay_bars < 0:
            raise ValueError("execution_delay_bars must be >= 0")
        if holding_bars < 1:
            raise ValueError("holding_bars must be >= 1")
        if drift_window < 30:
            raise ValueError("drift_window must be >= 30")

        self.model_random_state = model_random_state
        self.signal_generator = ArbitrageSignalGenerator(
            confidence_threshold=confidence_threshold,
            entry_tracking_error=entry_tracking_error,
            max_notional=max_notional,
            min_notional=min_notional,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )

        self.min_train_rows = min_train_rows
        self.retrain_every = retrain_every
        self.execution_delay_bars = execution_delay_bars
        self.holding_bars = holding_bars
        self.drift_window = drift_window
        self.retrain_mae_ratio_trigger = retrain_mae_ratio_trigger
        self.retrain_mean_shift_trigger_sigma = retrain_mean_shift_trigger_sigma

    @staticmethod
    def _drop_invalid(feature_panel: pd.DataFrame, target_col: str) -> pd.DataFrame:
        clean = feature_panel.dropna(subset=[target_col]).copy()
        return clean.sort_index()

    @staticmethod
    def _build_pair_lookup(market_panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
        lookup: dict[str, pd.DataFrame] = {}
        for pair, panel in market_panel.groupby("pair", observed=True):
            pair_panel = panel.sort_index().copy()
            pair_panel["tracking_diff_1"] = (
                pair_panel["etf_close"].pct_change() - pair_panel["benchmark_close"].pct_change()
            )
            pair_panel["spread"] = np.log(pair_panel["etf_close"] / pair_panel["benchmark_close"])
            lookup[str(pair)] = pair_panel
        return lookup

    @staticmethod
    def _resolve_future_timestamp(
        pair_panel: pd.DataFrame,
        current_timestamp: pd.Timestamp,
        bars_ahead: int,
    ) -> pd.Timestamp | None:
        if current_timestamp not in pair_panel.index:
            return None
        loc = pair_panel.index.get_indexer([current_timestamp])
        if len(loc) == 0 or loc[0] < 0:
            return None
        future_pos = int(loc[0] + bars_ahead)
        if future_pos >= len(pair_panel.index):
            return None
        return pd.Timestamp(pair_panel.index[future_pos])

    def _simulate_trade(
        self,
        signal_row: dict[str, Any],
        pair_panel: pd.DataFrame,
    ) -> dict[str, Any]:
        timestamp = pd.Timestamp(signal_row["timestamp"])
        action = str(signal_row["action"])
        confidence = float(signal_row["confidence"])
        notional = float(signal_row["notional"])
        estimated_profit_bps = float(signal_row["estimated_profit_bps"])

        if action == "HOLD" or notional <= 0:
            return {
                "executed": False,
                "hit": False,
                "realized_net_profit": 0.0,
                "realized_net_profit_bps": 0.0,
                "realized_spread_capture": 0.0,
                "entry_timestamp": timestamp,
                "exit_timestamp": timestamp,
                "confidence": confidence,
                "estimated_profit_bps": estimated_profit_bps,
            }

        entry_ts = self._resolve_future_timestamp(pair_panel, timestamp, self.execution_delay_bars)
        if entry_ts is None:
            return {
                "executed": False,
                "hit": False,
                "realized_net_profit": 0.0,
                "realized_net_profit_bps": 0.0,
                "realized_spread_capture": 0.0,
                "entry_timestamp": timestamp,
                "exit_timestamp": timestamp,
                "confidence": confidence,
                "estimated_profit_bps": estimated_profit_bps,
            }

        exit_ts = self._resolve_future_timestamp(pair_panel, entry_ts, self.holding_bars)
        if exit_ts is None:
            return {
                "executed": False,
                "hit": False,
                "realized_net_profit": 0.0,
                "realized_net_profit_bps": 0.0,
                "realized_spread_capture": 0.0,
                "entry_timestamp": entry_ts,
                "exit_timestamp": entry_ts,
                "confidence": confidence,
                "estimated_profit_bps": estimated_profit_bps,
            }

        current_diff = float(pair_panel.loc[entry_ts, "tracking_diff_1"])
        future_diff = float(pair_panel.loc[exit_ts, "tracking_diff_1"])

        # Capture is positive when divergence shrinks after entering the trade.
        spread_capture = abs(current_diff) - abs(future_diff)
        gross_profit_bps = spread_capture * 10000.0
        net_cost_bps = self.signal_generator.transaction_cost_bps + self.signal_generator.slippage_bps
        net_profit_bps = gross_profit_bps - net_cost_bps
        net_profit = notional * (net_profit_bps / 10000.0)

        if action == "CREATE":
            directional_hit = bool(current_diff > 0 and future_diff < current_diff)
        else:
            directional_hit = bool(current_diff < 0 and future_diff > current_diff)

        hit = bool(directional_hit and net_profit > 0)

        denom = max(abs(current_diff), 1e-8)
        realized_spread_capture = float(spread_capture / denom)

        return {
            "executed": True,
            "hit": hit,
            "realized_net_profit": float(net_profit),
            "realized_net_profit_bps": float(net_profit_bps),
            "realized_spread_capture": realized_spread_capture,
            "entry_timestamp": entry_ts,
            "exit_timestamp": exit_ts,
            "confidence": confidence,
            "estimated_profit_bps": estimated_profit_bps,
        }

    def _drift_and_regime_alerts(
        self,
        residual_frame: pd.DataFrame,
        pair_name: str,
        previous_regime: str | None,
    ) -> tuple[list[dict[str, Any]], bool, str | None]:
        alerts: list[dict[str, Any]] = []
        if len(residual_frame) < self.drift_window * 2:
            return alerts, False, previous_regime

        baseline = residual_frame.iloc[: self.drift_window]
        recent = residual_frame.iloc[-self.drift_window :]

        baseline_mae = float(np.mean(np.abs(baseline["residual"])))
        recent_mae = float(np.mean(np.abs(recent["residual"])))
        mae_ratio = recent_mae / max(baseline_mae, 1e-8)

        baseline_mean = float(baseline["residual"].mean())
        recent_mean = float(recent["residual"].mean())
        baseline_std = float(baseline["residual"].std(ddof=1) + 1e-8)
        mean_shift_sigma = abs(recent_mean - baseline_mean) / baseline_std

        drift_trigger = bool(
            mae_ratio >= self.retrain_mae_ratio_trigger
            or mean_shift_sigma >= self.retrain_mean_shift_trigger_sigma
        )

        if drift_trigger:
            alerts.append(
                {
                    "pair": pair_name,
                    "alert_type": "model_drift",
                    "severity": "high",
                    "message": (
                        f"Residual drift trigger: mae_ratio={mae_ratio:.2f}, "
                        f"mean_shift_sigma={mean_shift_sigma:.2f}."
                    ),
                    "retrain_trigger": True,
                }
            )

        regime_detector = RegimeDetector(rolling_window=24)
        regime_result = regime_detector.detect_regime(residual_frame["residual"])
        current_regime = str(regime_result["current_regime"])

        regime_changed = previous_regime is not None and current_regime != previous_regime
        severe_regime = current_regime in {"Stress", "High_Vol"}
        high_conf = float(regime_result["confidence"]) >= 0.65

        regime_trigger = bool(regime_changed and severe_regime and high_conf)
        if regime_trigger:
            alerts.append(
                {
                    "pair": pair_name,
                    "alert_type": "regime_shift",
                    "severity": "medium" if current_regime == "Stress" else "high",
                    "message": (
                        f"Regime changed {previous_regime} -> {current_regime} "
                        f"with confidence {float(regime_result['confidence']):.2f}."
                    ),
                    "retrain_trigger": current_regime == "High_Vol",
                }
            )

        return alerts, bool(drift_trigger or regime_trigger), current_regime

    @staticmethod
    def _compute_drawdown(pnl_series: pd.Series) -> float:
        if pnl_series.empty:
            return 0.0
        equity = pnl_series.cumsum()
        running_max = equity.cummax()
        drawdown = equity - running_max
        return float(drawdown.min())

    @staticmethod
    def recalibrate_thresholds(opportunity_frame: pd.DataFrame) -> ThresholdRecommendation:
        """Search confidence/profit thresholds that optimize forward KPIs."""
        if opportunity_frame.empty:
            return ThresholdRecommendation(
                confidence_threshold=0.70,
                min_expected_profit_bps=0.0,
                objective_score=0.0,
                evaluated_trades=0,
            )

        candidates_conf = np.arange(0.60, 0.96, 0.05)
        candidates_profit = np.arange(-5.0, 30.1, 5.0)

        best = ThresholdRecommendation(0.70, 0.0, -1.0, 0)
        for conf in candidates_conf:
            for min_profit in candidates_profit:
                selected = opportunity_frame[
                    (opportunity_frame["action"] != "HOLD")
                    & (opportunity_frame["confidence"] >= conf)
                    & (opportunity_frame["estimated_profit_bps"] >= min_profit)
                    & (opportunity_frame["executed"])
                ]
                if len(selected) < 20:
                    continue

                precision = float((selected["realized_net_profit_bps"] > 0).mean())
                hit_ratio = float(selected["hit"].mean())
                spread_capture = float(selected["realized_spread_capture"].mean())

                objective = 0.45 * precision + 0.35 * hit_ratio + 0.20 * max(spread_capture, 0.0)
                if objective > best.objective_score:
                    best = ThresholdRecommendation(
                        confidence_threshold=float(round(conf, 2)),
                        min_expected_profit_bps=float(round(min_profit, 2)),
                        objective_score=float(round(objective, 6)),
                        evaluated_trades=int(len(selected)),
                    )

        if best.objective_score < 0:
            return ThresholdRecommendation(0.70, 0.0, 0.0, 0)

        return best

    def run(
        self,
        market_panel: pd.DataFrame,
        feature_panel: pd.DataFrame,
        target_col: str = "target_te",
    ) -> dict[str, Any]:
        """Run walk-forward backtest and paper-trading simulation."""
        clean_features = self._drop_invalid(feature_panel, target_col=target_col)
        pair_lookup = self._build_pair_lookup(market_panel)

        if clean_features.empty:
            raise ValueError("Feature panel is empty after dropping invalid rows")

        model: TrackingErrorModel | None = None
        timestamps = sorted(clean_features.index.unique())

        prediction_rows: list[dict[str, Any]] = []
        paper_rows: list[dict[str, Any]] = []
        alert_rows: list[dict[str, Any]] = []

        residual_history_by_pair: dict[str, list[dict[str, Any]]] = {}
        regime_state_by_pair: dict[str, str | None] = {}

        steps_since_retrain = 0  # Start with 0 to delay first retrain until we have enough data
        force_retrain = False

        for idx in range(1, len(timestamps)):
            ts = pd.Timestamp(timestamps[idx])
            train_cutoff = pd.Timestamp(timestamps[idx - 1])

            train_df = clean_features.loc[clean_features.index <= train_cutoff].copy()
            if len(train_df) < self.min_train_rows:
                continue

            if model is None or steps_since_retrain >= self.retrain_every or force_retrain:
                model = TrackingErrorModel(random_state=self.model_random_state)
                # Use adaptive test_size based on training data size to ensure stable splits
                # Keep more data for training on small datasets
                if len(train_df) < 300:
                    adaptive_test_size = 0.10  # Use 10% test split for smaller datasets
                else:
                    adaptive_test_size = 0.20  # Use standard 20% for larger datasets
                model.train(train_df, target_col=target_col, test_size=adaptive_test_size)
                steps_since_retrain = 0
                force_retrain = False

            model_columns = model.numeric_columns + model.categorical_columns
            test_slice = clean_features.loc[clean_features.index == ts].copy()
            test_slice = test_slice.dropna(subset=model_columns + [target_col])
            if test_slice.empty:
                steps_since_retrain += 1
                continue

            predictions = model.predict(test_slice[model_columns])
            test_slice = test_slice.copy()
            test_slice["predicted_te"] = predictions
            test_slice["residual"] = test_slice[target_col] - test_slice["predicted_te"]

            for _, row in test_slice.iterrows():
                pair_name = str(row["pair"])
                pair_panel = pair_lookup.get(pair_name)
                if pair_panel is None:
                    continue

                pair_panel_until_now = pair_panel.loc[pair_panel.index <= ts].copy()
                if pair_panel_until_now.empty:
                    continue

                if pair_name not in residual_history_by_pair:
                    residual_history_by_pair[pair_name] = []
                    regime_state_by_pair[pair_name] = None

                # Regime-adaptive thresholds are estimated from historical residuals only.
                residual_frame = pd.DataFrame(residual_history_by_pair[pair_name])
                adaptive_thresholds: dict[str, float] | None = None
                regime_result: dict[str, Any] | None = None
                if len(residual_frame) >= 60:
                    detector = RegimeDetector(rolling_window=24)
                    regime_result = detector.detect_regime(residual_frame["residual"])
                    adaptive_thresholds = regime_result.get("adaptive_thresholds")

                signal = self.signal_generator.generate_signal(
                    pair_panel=pair_panel_until_now,
                    pair_name=pair_name,
                    predicted_tracking_error=float(row["predicted_te"]),
                    historical_tracking_error=pair_panel_until_now["tracking_diff_1"],
                    etf_price=float(pair_panel_until_now["etf_close"].iloc[-1]),
                    adaptive_thresholds=adaptive_thresholds,
                    regime_result=regime_result,
                )

                trade_result = self._simulate_trade(signal.to_dict(), pair_panel)
                prediction_rows.append(
                    {
                        "timestamp": ts,
                        "pair": pair_name,
                        "actual_te": float(row[target_col]),
                        "predicted_te": float(row["predicted_te"]),
                        "residual": float(row["residual"]),
                    }
                )

                row_payload = signal.to_dict() | trade_result
                paper_rows.append(row_payload)

                residual_history_by_pair[pair_name].append(
                    {
                        "timestamp": ts,
                        "actual_te": float(row[target_col]),
                        "predicted_te": float(row["predicted_te"]),
                        "residual": float(row["residual"]),
                    }
                )

                pair_residual_frame = pd.DataFrame(residual_history_by_pair[pair_name])
                pair_alerts, should_retrain, new_regime = self._drift_and_regime_alerts(
                    pair_residual_frame,
                    pair_name=pair_name,
                    previous_regime=regime_state_by_pair[pair_name],
                )

                regime_state_by_pair[pair_name] = new_regime
                if pair_alerts:
                    for alert in pair_alerts:
                        alert_rows.append(alert | {"timestamp": ts})
                if should_retrain:
                    force_retrain = True

            steps_since_retrain += 1

        predictions_df = pd.DataFrame(prediction_rows)
        paper_df = pd.DataFrame(paper_rows)
        alerts_df = pd.DataFrame(alert_rows)

        if predictions_df.empty:
            raise RuntimeError("Walk-forward run produced no predictions. Increase lookback or lower min_train_rows.")

        ml_mae = float(np.mean(np.abs(predictions_df["actual_te"] - predictions_df["predicted_te"])))
        ml_rmse = float(
            np.sqrt(np.mean(np.square(predictions_df["actual_te"] - predictions_df["predicted_te"])))
        )

        actionable = paper_df[(paper_df["action"] != "HOLD") & (paper_df["executed"])].copy()
        precision = float((actionable["realized_net_profit_bps"] > 0).mean()) if not actionable.empty else 0.0
        hit_ratio = float(actionable["hit"].mean()) if not actionable.empty else 0.0
        spread_capture = float(actionable["realized_spread_capture"].mean()) if not actionable.empty else 0.0

        pnl_series = actionable.sort_values("exit_timestamp")["realized_net_profit"] if not actionable.empty else pd.Series(dtype=float)
        max_drawdown = self._compute_drawdown(pnl_series)

        recommendation = self.recalibrate_thresholds(paper_df)

        kpis = {
            "ml_mae": ml_mae,
            "ml_rmse": ml_rmse,
            "trade_precision": precision,
            "hit_ratio_after_delay": hit_ratio,
            "realized_spread_capture": spread_capture,
            "max_drawdown": max_drawdown,
            "executed_trades": int(len(actionable)),
            "total_realized_pnl": float(actionable["realized_net_profit"].sum()) if not actionable.empty else 0.0,
        }

        recalibration = {
            "recommended_confidence_threshold": recommendation.confidence_threshold,
            "recommended_min_expected_profit_bps": recommendation.min_expected_profit_bps,
            "objective_score": recommendation.objective_score,
            "evaluated_trades": recommendation.evaluated_trades,
        }

        return {
            "kpis": kpis,
            "recalibration": recalibration,
            "predictions": predictions_df,
            "paper_trades": paper_df,
            "alerts": alerts_df,
        }
