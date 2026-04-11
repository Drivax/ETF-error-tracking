"""Arbitrage signal generation for ETF creation and redemption decisions.

This module produces conservative, production-ready signals from predicted tracking
error. It is designed for desk usage where capital allocation must remain risk-aware.

Signal convention used by this implementation:
- Positive predicted tracking error (ETF premium) -> CREATE
- Negative predicted tracking error (ETF discount) -> REDEEM
- Low confidence or weak edge -> HOLD
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ArbitrageSignalOutput:
    """Structured signal output for one ETF pair at a given timestamp."""

    timestamp: pd.Timestamp
    pair: str
    action: str
    confidence: float
    recommended_shares: int
    notional: float
    estimated_profit: float
    estimated_profit_bps: float
    predicted_tracking_error: float
    liquidity_score: float
    persistence_score: float
    reason: str

    def to_dict(self) -> dict[str, float | str | int | pd.Timestamp]:
        """Serialize dataclass output into a dict for DataFrame usage."""
        return {
            "timestamp": self.timestamp,
            "pair": self.pair,
            "action": self.action,
            "confidence": self.confidence,
            "recommended_shares": self.recommended_shares,
            "notional": self.notional,
            "estimated_profit": self.estimated_profit,
            "estimated_profit_bps": self.estimated_profit_bps,
            "predicted_tracking_error": self.predicted_tracking_error,
            "liquidity_score": self.liquidity_score,
            "persistence_score": self.persistence_score,
            "reason": self.reason,
        }


class ArbitrageSignalGenerator:
    """Generate actionable and conservative CREATE/REDEEM/HOLD decisions.

    The confidence score combines:
    - Magnitude of predicted deviation
    - Persistence of deviation in recent realized observations
    - ETF liquidity proxy
    """

    def __init__(
        self,
        confidence_threshold: float = 0.70,
        entry_tracking_error: float = 0.0005,
        max_notional: float = 1_500_000.0,
        transaction_cost_bps: float = 3.0,
        slippage_bps: float = 2.0,
        min_notional: float = 100_000.0,
        persistence_window: int = 9,
        liquidity_window: int = 12,
    ) -> None:
        if not (0.50 <= confidence_threshold <= 0.99):
            raise ValueError("confidence_threshold must be between 0.50 and 0.99")
        if entry_tracking_error <= 0:
            raise ValueError("entry_tracking_error must be strictly positive")

        self.confidence_threshold = confidence_threshold
        self.entry_tracking_error = entry_tracking_error
        self.max_notional = max_notional
        self.min_notional = min_notional
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.persistence_window = persistence_window
        self.liquidity_window = liquidity_window

    @staticmethod
    def _clip_01(value: float) -> float:
        """Clip scalar to the [0, 1] interval."""
        return float(np.clip(value, 0.0, 1.0))

    def _deviation_magnitude_score(self, predicted_tracking_error: float) -> float:
        """Map TE magnitude to a confidence component in [0, 1]."""
        scaled = abs(predicted_tracking_error) / self.entry_tracking_error
        # Saturate around 3x threshold to avoid overconfidence on extreme points.
        return self._clip_01(scaled / 3.0)

    def _persistence_score(
        self,
        predicted_tracking_error: float,
        historical_tracking_error: pd.Series | None,
    ) -> float:
        """Estimate persistence using recent sign consistency and strength.

        A signal is more reliable when recent realized deviations share the same sign
        and are of non-trivial magnitude.
        """
        if historical_tracking_error is None or historical_tracking_error.dropna().empty:
            return 0.35

        recent = historical_tracking_error.dropna().tail(self.persistence_window)
        if recent.empty:
            return 0.35

        current_sign = np.sign(predicted_tracking_error)
        if current_sign == 0:
            return 0.0

        sign_agreement = float((np.sign(recent) == current_sign).mean())
        recent_strength = float(np.mean(np.abs(recent)))
        strength_score = self._clip_01(recent_strength / (2.0 * self.entry_tracking_error))
        return self._clip_01(0.7 * sign_agreement + 0.3 * strength_score)

    def _liquidity_score(self, pair_panel: pd.DataFrame) -> float:
        """Compute ETF liquidity score from recent dollar volume.

        We use rolling dollar volume as a practical execution proxy. The scale is
        intentionally conservative and saturates for very liquid ETFs.
        """
        required_columns = {"etf_close", "etf_volume"}
        if not required_columns.issubset(pair_panel.columns):
            return 0.4

        frame = pair_panel.dropna(subset=["etf_close", "etf_volume"]).copy()
        if frame.empty:
            return 0.4

        frame["dollar_volume"] = frame["etf_close"] * frame["etf_volume"]
        rolling_dv = frame["dollar_volume"].tail(self.liquidity_window).mean()

        # Normalize against 25 million USD as desk-friendly liquidity reference.
        liquidity = float(rolling_dv / 25_000_000.0)
        return self._clip_01(liquidity)

    def _confidence(
        self,
        predicted_tracking_error: float,
        historical_tracking_error: pd.Series | None,
        pair_panel: pd.DataFrame,
    ) -> tuple[float, float, float]:
        """Combine magnitude, persistence, and liquidity into final confidence."""
        magnitude = self._deviation_magnitude_score(predicted_tracking_error)
        persistence = self._persistence_score(predicted_tracking_error, historical_tracking_error)
        liquidity = self._liquidity_score(pair_panel)

        confidence = 0.50 * magnitude + 0.30 * persistence + 0.20 * liquidity
        return self._clip_01(confidence), persistence, liquidity

    @staticmethod
    def _estimate_profit(
        tracking_error: float,
        notional: float,
        transaction_cost_bps: float,
        slippage_bps: float,
    ) -> tuple[float, float]:
        """Estimate net expected profit from deviation capture after costs."""
        gross_profit = abs(tracking_error) * notional
        costs = ((transaction_cost_bps + slippage_bps) / 10000.0) * notional
        net_profit = gross_profit - costs
        net_profit_bps = (net_profit / max(notional, 1e-8)) * 10000.0
        return float(net_profit), float(net_profit_bps)

    def _recommended_notional(
        self,
        confidence: float,
        predicted_tracking_error: float,
    ) -> float:
        """Size notional conservatively according to confidence and edge quality."""
        if confidence < self.confidence_threshold:
            return 0.0

        edge_intensity = self._clip_01(abs(predicted_tracking_error) / (2.5 * self.entry_tracking_error))
        size_fraction = confidence * edge_intensity
        proposed_notional = self.max_notional * size_fraction

        if proposed_notional < self.min_notional:
            return 0.0
        return float(min(proposed_notional, self.max_notional))

    def generate_signal(
        self,
        pair_panel: pd.DataFrame,
        pair_name: str,
        predicted_tracking_error: float,
        historical_tracking_error: pd.Series | None = None,
        etf_price: float | None = None,
    ) -> ArbitrageSignalOutput:
        """Generate one conservative arbitrage signal for the latest timestamp.

        Parameters
        ----------
        pair_panel:
            Price/volume frame for one ETF-benchmark pair.
        pair_name:
            Human-readable pair identifier.
        predicted_tracking_error:
            Model-predicted tracking error (decimal form, e.g. 0.001 = 10 bps).
        historical_tracking_error:
            Optional recent realized tracking error series used for persistence scoring.
        etf_price:
            Optional ETF price override. If None, latest panel close is used.
        """
        if pair_panel.empty:
            raise ValueError("pair_panel must not be empty")

        timestamp = pair_panel.index[-1]
        if etf_price is None:
            if "etf_close" not in pair_panel.columns:
                raise ValueError("pair_panel must contain etf_close when etf_price is not provided")
            etf_price = float(pair_panel["etf_close"].iloc[-1])

        confidence, persistence_score, liquidity_score = self._confidence(
            predicted_tracking_error=predicted_tracking_error,
            historical_tracking_error=historical_tracking_error,
            pair_panel=pair_panel,
        )

        recommended_notional = self._recommended_notional(
            confidence=confidence,
            predicted_tracking_error=predicted_tracking_error,
        )

        estimated_profit, estimated_profit_bps = self._estimate_profit(
            tracking_error=predicted_tracking_error,
            notional=recommended_notional,
            transaction_cost_bps=self.transaction_cost_bps,
            slippage_bps=self.slippage_bps,
        )

        if confidence < self.confidence_threshold or recommended_notional <= 0 or estimated_profit <= 0:
            action = "HOLD"
            reason = (
                "Signal below confidence or profit threshold; no capital allocation "
                "recommended under conservative risk rules."
            )
            recommended_notional = 0.0
            recommended_shares = 0
            estimated_profit = 0.0
            estimated_profit_bps = 0.0
        else:
            action = "CREATE" if predicted_tracking_error > 0 else "REDEEM"
            recommended_shares = int(max(np.floor(recommended_notional / max(etf_price, 1e-6)), 0))

            deviation_direction = "premium" if predicted_tracking_error > 0 else "discount"
            reason = (
                f"Significant {deviation_direction} with persistence score {persistence_score:.2f}, "
                f"liquidity score {liquidity_score:.2f}, and confidence {confidence:.2f}."
            )

        return ArbitrageSignalOutput(
            timestamp=timestamp,
            pair=pair_name,
            action=action,
            confidence=float(confidence),
            recommended_shares=int(recommended_shares),
            notional=float(recommended_notional),
            estimated_profit=float(estimated_profit),
            estimated_profit_bps=float(estimated_profit_bps),
            predicted_tracking_error=float(predicted_tracking_error),
            liquidity_score=float(liquidity_score),
            persistence_score=float(persistence_score),
            reason=reason,
        )

    def generate_universe_signals(
        self,
        intraday_panel: pd.DataFrame,
        prediction_snapshot: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate current arbitrage signals across all pairs.

        This method is dashboard-friendly and computes each pair signal from the
        latest prediction plus recent realized tracking error persistence.
        """
        if prediction_snapshot.empty:
            return pd.DataFrame()

        rows: list[dict[str, float | str | int | pd.Timestamp]] = []
        for pair_name, pair_panel in intraday_panel.groupby("pair", observed=True):
            pair_prediction = prediction_snapshot[prediction_snapshot["pair"] == pair_name]
            if pair_prediction.empty:
                continue

            latest_prediction = pair_prediction.iloc[0]
            pair_panel_sorted = pair_panel.sort_index().copy()
            pair_panel_sorted["etf_ret"] = pair_panel_sorted["etf_close"].pct_change()
            pair_panel_sorted["benchmark_ret"] = pair_panel_sorted["benchmark_close"].pct_change()
            historical_te = (pair_panel_sorted["etf_ret"] - pair_panel_sorted["benchmark_ret"]).dropna()

            signal = self.generate_signal(
                pair_panel=pair_panel,
                pair_name=pair_name,
                predicted_tracking_error=float(latest_prediction["predicted_tracking_error"]),
                historical_tracking_error=historical_te,
                etf_price=float(pair_panel_sorted["etf_close"].iloc[-1]),
            )
            rows.append(signal.to_dict())

        if not rows:
            return pd.DataFrame()

        signal_df = pd.DataFrame(rows).sort_values(["confidence", "estimated_profit"], ascending=[False, False])
        return signal_df.reset_index(drop=True)
