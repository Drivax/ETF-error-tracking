"""Actionable arbitrage signal generation for ETF creation/redemption workflows.

Compared to pure z-score flags, this module produces desk-ready decisions with:
- Backtest-calibrated entry thresholds.
- Estimated net profitability after costs and slippage.
- Confidence score from historical hit ratio and prediction uncertainty.
- Recommended action size that scales with both edge and risk budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArbitrageSignalDecision:
    """One arbitrage recommendation for a pair at a timestamp."""

    timestamp: pd.Timestamp
    pair: str
    signal: str
    action: str
    signal_strength: float
    confidence_level: float
    estimated_gross_profit_bps: float
    estimated_net_profit_bps: float
    expected_holding_bars: int
    recommended_action_notional: float
    spread_zscore: float
    spread_volatility: float
    calibrated_entry_threshold: float

    def to_dict(self) -> dict[str, float | str | pd.Timestamp]:
        """Convert decision dataclass to dictionary for DataFrame assembly."""
        return asdict(self)


class ArbitrageSignalGenerator:
    """Generate calibrated creation/redemption signals from spread dislocations."""

    def __init__(
        self,
        rolling_window: int = 60,
        holding_bars: int = 6,
        execution_cost_bps: float = 3.0,
        slippage_bps: float = 2.0,
        base_entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        risk_budget_notional: float = 1_000_000.0,
    ) -> None:
        self.rolling_window = rolling_window
        self.holding_bars = holding_bars
        self.execution_cost_bps = execution_cost_bps
        self.slippage_bps = slippage_bps
        self.base_entry_zscore = base_entry_zscore
        self.exit_zscore = exit_zscore
        self.risk_budget_notional = risk_budget_notional

        # Threshold can be calibrated from historical data and then reused in production.
        self.calibrated_entry_zscore = base_entry_zscore
        self.calibration_summary: dict[str, float] = {}

    def _prepare_spread_frame(self, pair_panel: pd.DataFrame) -> pd.DataFrame:
        """Build spread and z-score statistics used by both calibration and live decisions."""
        required_columns = {"etf_close", "benchmark_close"}
        missing = required_columns.difference(pair_panel.columns)
        if missing:
            raise ValueError(f"Missing required price columns for arbitrage logic: {sorted(missing)}")

        frame = pair_panel.copy().sort_index()
        frame["spread"] = np.log(frame["etf_close"] / frame["benchmark_close"])
        frame["spread_change"] = frame["spread"].diff()

        rolling_mean = frame["spread"].rolling(self.rolling_window).mean()
        rolling_std = frame["spread"].rolling(self.rolling_window).std()
        frame["spread_zscore"] = (frame["spread"] - rolling_mean) / (rolling_std + 1e-8)
        frame["spread_volatility"] = frame["spread_change"].rolling(self.rolling_window).std()

        return frame

    def calibrate_threshold(self, history_panel: pd.DataFrame, minimum_samples: int = 150) -> dict[str, float]:
        """Calibrate entry threshold via simple historical net-capture backtest.

        We test a grid of z-score entry values. For each candidate threshold, we
        approximate gross spread capture by absolute spread reversion over a fixed
        holding horizon, then subtract explicit execution and slippage costs.
        """
        frame = self._prepare_spread_frame(history_panel)
        frame["future_spread"] = frame["spread"].shift(-self.holding_bars)
        frame["future_reversion_bps"] = (frame["spread"].abs() - frame["future_spread"].abs()) * 10000.0

        candidate_thresholds = np.arange(1.5, 3.1, 0.25)
        best_threshold = self.base_entry_zscore
        best_score = float("-inf")

        total_cost = self.execution_cost_bps + self.slippage_bps

        for threshold in candidate_thresholds:
            candidate = frame[frame["spread_zscore"].abs() >= threshold].dropna(subset=["future_reversion_bps"])
            if len(candidate) < minimum_samples:
                continue

            net_profit = candidate["future_reversion_bps"] - total_cost
            hit_ratio = float((net_profit > 0).mean())
            avg_net = float(net_profit.mean())

            # Composite score balances profitability and consistency.
            score = avg_net * hit_ratio
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        self.calibrated_entry_zscore = best_threshold
        self.calibration_summary = {
            "calibrated_entry_zscore": float(best_threshold),
            "calibration_score": float(best_score if np.isfinite(best_score) else 0.0),
            "holding_bars": float(self.holding_bars),
            "execution_cost_bps": float(self.execution_cost_bps),
            "slippage_bps": float(self.slippage_bps),
        }
        return self.calibration_summary

    def _estimate_gross_profit_bps(self, latest_zscore: float, latest_volatility: float) -> float:
        """Estimate gross capture potential in basis points from dislocation magnitude."""
        excess_dislocation = max(abs(latest_zscore) - self.exit_zscore, 0.0)
        gross_profit_bps = excess_dislocation * max(latest_volatility, 1e-8) * np.sqrt(self.holding_bars) * 10000.0
        return float(gross_profit_bps)

    def _compute_confidence(
        self,
        latest_zscore: float,
        uncertainty_sigma: float,
        predicted_tracking_error: float,
    ) -> float:
        """Build confidence score from edge magnitude and model uncertainty.

        Confidence is intentionally conservative: a strong dislocation can still lead
        to low confidence when prediction uncertainty is elevated.
        """
        edge_component = min(max((abs(latest_zscore) - self.calibrated_entry_zscore) / 2.0, 0.0), 1.0)

        # Uncertainty penalty compares predictive sigma to predicted level.
        scale = max(predicted_tracking_error, 1e-8)
        relative_uncertainty = uncertainty_sigma / scale
        uncertainty_component = max(0.0, 1.0 - min(relative_uncertainty, 1.0))

        confidence = 0.65 * edge_component + 0.35 * uncertainty_component
        return float(np.clip(confidence, 0.0, 1.0))

    def _recommended_notional(
        self,
        confidence_level: float,
        signal_strength: float,
        predicted_tracking_error: float,
    ) -> float:
        """Map signal quality and risk to a recommended action notional."""
        # Scale down size when predicted TE is high, because execution risk is higher.
        risk_penalty = 1.0 / (1.0 + predicted_tracking_error * 10000.0)
        notional = self.risk_budget_notional * confidence_level * signal_strength * risk_penalty
        return float(max(notional, 0.0))

    def generate_signal(
        self,
        pair_panel: pd.DataFrame,
        pair_name: str,
        predicted_tracking_error: float,
        uncertainty_sigma: float,
    ) -> ArbitrageSignalDecision:
        """Generate one actionable creation/redemption signal for the latest bar."""
        frame = self._prepare_spread_frame(pair_panel)
        latest = frame.dropna(subset=["spread_zscore", "spread_volatility"]).tail(1)
        if latest.empty:
            raise ValueError("Insufficient data to compute arbitrage signal.")

        row = latest.iloc[0]
        timestamp = latest.index[-1]
        zscore = float(row["spread_zscore"])
        spread_volatility = float(row["spread_volatility"])

        gross_profit_bps = self._estimate_gross_profit_bps(zscore, spread_volatility)
        net_profit_bps = gross_profit_bps - self.execution_cost_bps - self.slippage_bps

        edge_over_threshold = max(abs(zscore) - self.calibrated_entry_zscore, 0.0)
        signal_strength = float(np.clip(edge_over_threshold / 1.5, 0.0, 1.0))
        confidence_level = self._compute_confidence(zscore, uncertainty_sigma, predicted_tracking_error)

        if abs(zscore) < self.calibrated_entry_zscore or net_profit_bps <= 0:
            signal = "NO_ACTION"
            action = "Hold"
            signal_strength = 0.0
            recommended_notional = 0.0
        elif zscore < 0:
            # ETF cheap to benchmark proxy: buy ETF and short basket (creation style flow).
            signal = "CREATION_OPPORTUNITY"
            action = "Buy ETF / Sell Benchmark Basket"
            recommended_notional = self._recommended_notional(
                confidence_level=confidence_level,
                signal_strength=signal_strength,
                predicted_tracking_error=predicted_tracking_error,
            )
        else:
            # ETF rich to benchmark proxy: sell ETF and buy basket (redemption style flow).
            signal = "REDEMPTION_OPPORTUNITY"
            action = "Sell ETF / Buy Benchmark Basket"
            recommended_notional = self._recommended_notional(
                confidence_level=confidence_level,
                signal_strength=signal_strength,
                predicted_tracking_error=predicted_tracking_error,
            )

        return ArbitrageSignalDecision(
            timestamp=timestamp,
            pair=pair_name,
            signal=signal,
            action=action,
            signal_strength=float(signal_strength),
            confidence_level=float(confidence_level),
            estimated_gross_profit_bps=float(gross_profit_bps),
            estimated_net_profit_bps=float(net_profit_bps),
            expected_holding_bars=int(self.holding_bars),
            recommended_action_notional=float(recommended_notional),
            spread_zscore=float(zscore),
            spread_volatility=float(spread_volatility),
            calibrated_entry_threshold=float(self.calibrated_entry_zscore),
        )

    def generate_universe_signals(
        self,
        intraday_panel: pd.DataFrame,
        prediction_snapshot: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate latest arbitrage decisions for all pairs in the universe."""
        if prediction_snapshot.empty:
            return pd.DataFrame()

        rows: list[dict[str, float | str | pd.Timestamp]] = []
        for pair_name, pair_panel in intraday_panel.groupby("pair", observed=True):
            pair_prediction = prediction_snapshot[prediction_snapshot["pair"] == pair_name]
            if pair_prediction.empty:
                continue

            latest_prediction = pair_prediction.iloc[0]
            decision = self.generate_signal(
                pair_panel=pair_panel,
                pair_name=pair_name,
                predicted_tracking_error=float(latest_prediction["predicted_tracking_error"]),
                uncertainty_sigma=float(latest_prediction["uncertainty_sigma"]),
            )
            rows.append(decision.to_dict())

        if not rows:
            return pd.DataFrame()

        signal_df = pd.DataFrame(rows).sort_values(
            ["estimated_net_profit_bps", "confidence_level"],
            ascending=[False, False],
        )
        return signal_df.reset_index(drop=True)
