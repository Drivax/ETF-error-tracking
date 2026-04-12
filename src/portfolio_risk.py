"""Portfolio-level risk aggregation for ETF tracking error monitoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioRiskSummary:
    """Top-line portfolio risk metrics."""

    total_predicted_te_pct: float
    portfolio_te_var_pct: float
    aggregate_arbitrage_risk_score: float
    actionable_weight_pct: float


class PortfolioRiskAggregator:
    """Aggregate ETF-level tracking error and arbitrage risk to portfolio level."""

    def __init__(self, var_confidence: float = 0.95, var_lookback: int = 240) -> None:
        if not (0.80 <= var_confidence < 1.0):
            raise ValueError("var_confidence must be in [0.80, 1.0)")
        if var_lookback < 30:
            raise ValueError("var_lookback must be >= 30")

        self.var_confidence = var_confidence
        self.var_lookback = var_lookback

    @staticmethod
    def normalize_weights(weight_map: dict[str, float]) -> dict[str, float]:
        """Return strictly positive ETF weights normalized to 1."""
        clean = {
            str(etf).strip().upper(): float(weight)
            for etf, weight in weight_map.items()
            if pd.notna(weight) and float(weight) > 0
        }
        if not clean:
            return {}

        total = float(sum(clean.values()))
        if total <= 0:
            return {}

        return {etf: value / total for etf, value in clean.items()}

    @staticmethod
    def _build_pair_weight_map(
        latest_predictions: pd.DataFrame,
        normalized_weights: dict[str, float],
    ) -> dict[str, float]:
        """Project ETF weights to pair-level weights from latest predictions."""
        if latest_predictions.empty:
            return {}

        if "pair" not in latest_predictions.columns:
            return {}

        pair_rows = latest_predictions[["pair"]].drop_duplicates().copy()
        if "etf_ticker" in latest_predictions.columns:
            pair_rows = pair_rows.merge(
                latest_predictions[["pair", "etf_ticker"]].drop_duplicates(),
                on="pair",
                how="left",
            )
        else:
            # Fallback: infer ETF ticker from pair format ETF_BENCHMARK.
            pair_rows["etf_ticker"] = pair_rows["pair"].astype(str).str.split("_", n=1).str[0]

        pair_rows["etf_ticker"] = pair_rows["etf_ticker"].astype(str).str.upper()
        pair_rows["weight"] = pair_rows["etf_ticker"].map(normalized_weights).fillna(0.0)

        return {
            str(row["pair"]): float(row["weight"])
            for _, row in pair_rows.iterrows()
            if float(row["weight"]) > 0
        }

    def build_portfolio_prediction_series(
        self,
        prediction_series: pd.DataFrame,
        latest_predictions: pd.DataFrame,
        normalized_weights: dict[str, float],
    ) -> pd.DataFrame:
        """Compute weighted portfolio predicted TE time series."""
        if prediction_series.empty or latest_predictions.empty or not normalized_weights:
            return pd.DataFrame(columns=["timestamp", "portfolio_predicted_te", "portfolio_sigma"])

        pair_weights = self._build_pair_weight_map(latest_predictions, normalized_weights)
        if not pair_weights:
            return pd.DataFrame(columns=["timestamp", "portfolio_predicted_te", "portfolio_sigma"])

        frame = prediction_series.copy()
        frame["pair_weight"] = frame["pair"].map(pair_weights).fillna(0.0)
        frame = frame[frame["pair_weight"] > 0].copy()
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "portfolio_predicted_te", "portfolio_sigma"])

        frame["weighted_te"] = frame["predicted_tracking_error"] * frame["pair_weight"]
        frame["weighted_sigma_sq"] = (frame["uncertainty_sigma"] * frame["pair_weight"]) ** 2

        grouped = (
            frame.groupby("timestamp", observed=True)
            .agg(
                portfolio_predicted_te=("weighted_te", "sum"),
                portfolio_sigma=("weighted_sigma_sq", lambda values: float(np.sqrt(np.sum(values)))),
            )
            .reset_index()
            .sort_values("timestamp")
        )

        return grouped.tail(self.var_lookback).reset_index(drop=True)

    def compute_var_pct(self, portfolio_series: pd.DataFrame) -> float:
        """Historical VaR on absolute portfolio TE in percent."""
        if portfolio_series.empty:
            return float("nan")

        abs_pct = (portfolio_series["portfolio_predicted_te"].abs() * 100.0).to_numpy(dtype=float)
        if len(abs_pct) < 15:
            return float("nan")

        return float(np.quantile(abs_pct, self.var_confidence, method="linear"))

    def compute_etf_contributions(
        self,
        live_overview: pd.DataFrame,
        normalized_weights: dict[str, float],
    ) -> pd.DataFrame:
        """Risk contribution by ETF as weighted absolute predicted TE."""
        if live_overview.empty or not normalized_weights:
            return pd.DataFrame()

        latest = (
            live_overview[["etf_ticker", "pair", "predicted_tracking_error", "uncertainty_sigma", "risk_bucket"]]
            .drop_duplicates(subset=["etf_ticker", "pair"])
            .copy()
        )
        latest["weight"] = latest["etf_ticker"].map(normalized_weights).fillna(0.0)
        latest = latest[latest["weight"] > 0].copy()
        if latest.empty:
            return pd.DataFrame()

        latest["weighted_abs_te"] = latest["weight"] * latest["predicted_tracking_error"].abs()
        latest["weighted_sigma"] = latest["weight"] * latest["uncertainty_sigma"]

        total_weighted_abs_te = float(latest["weighted_abs_te"].sum())
        latest["risk_contribution_pct"] = np.where(
            total_weighted_abs_te > 0,
            (latest["weighted_abs_te"] / total_weighted_abs_te) * 100.0,
            0.0,
        )

        latest = latest.sort_values("risk_contribution_pct", ascending=False).reset_index(drop=True)
        return latest

    @staticmethod
    def compute_sector_exposure(
        normalized_weights: dict[str, float],
        sector_map: dict[str, str],
    ) -> pd.DataFrame:
        """Aggregate ETF weights to sector exposures."""
        if not normalized_weights:
            return pd.DataFrame(columns=["sector", "exposure_pct"])

        rows = []
        for etf, weight in normalized_weights.items():
            sector = sector_map.get(etf, "Other / Unmapped")
            rows.append({"sector": sector, "weight": float(weight)})

        exposure = (
            pd.DataFrame(rows)
            .groupby("sector", observed=True, as_index=False)["weight"]
            .sum()
            .rename(columns={"weight": "exposure"})
        )
        exposure["exposure_pct"] = exposure["exposure"] * 100.0
        return exposure.sort_values("exposure_pct", ascending=False).reset_index(drop=True)

    def compute_arbitrage_aggregate(
        self,
        universe_signals: pd.DataFrame,
        normalized_weights: dict[str, float],
    ) -> tuple[float, float]:
        """Return aggregated arbitrage risk score and actionable weight share."""
        if universe_signals.empty or not normalized_weights:
            return 0.0, 0.0

        signals = universe_signals.copy()
        if "etf_ticker" not in signals.columns:
            # Pair format follows ETF_BENCHMARK in this project.
            signals["etf_ticker"] = signals["pair"].astype(str).str.split("_", n=1).str[0]

        signals["weight"] = signals["etf_ticker"].map(normalized_weights).fillna(0.0)
        signals = signals[signals["weight"] > 0].copy()
        if signals.empty:
            return 0.0, 0.0

        signals["actionable"] = signals["action"].isin(["CREATE", "REDEEM"])
        signals["risk_intensity"] = (
            signals["weight"]
            * signals["confidence"].clip(lower=0.0, upper=1.0)
            * signals["predicted_tracking_error"].abs()
            * 10_000.0
        )

        total_score = float(signals["risk_intensity"].sum())
        actionable_weight = float(signals.loc[signals["actionable"], "weight"].sum() * 100.0)
        return total_score, actionable_weight

    def summarize(
        self,
        live_overview: pd.DataFrame,
        portfolio_series: pd.DataFrame,
        universe_signals: pd.DataFrame,
        normalized_weights: dict[str, float],
    ) -> PortfolioRiskSummary:
        """Compute top-level portfolio risk summary metrics."""
        contributions = self.compute_etf_contributions(live_overview, normalized_weights)
        total_predicted_te_pct = float(contributions["weighted_abs_te"].sum() * 100.0) if not contributions.empty else 0.0
        var_pct = self.compute_var_pct(portfolio_series)
        arb_score, actionable_weight_pct = self.compute_arbitrage_aggregate(
            universe_signals=universe_signals,
            normalized_weights=normalized_weights,
        )

        return PortfolioRiskSummary(
            total_predicted_te_pct=total_predicted_te_pct,
            portfolio_te_var_pct=float(var_pct),
            aggregate_arbitrage_risk_score=arb_score,
            actionable_weight_pct=actionable_weight_pct,
        )
