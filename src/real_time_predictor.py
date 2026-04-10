"""Real-time and intraday tracking error prediction utilities.

This module turns the batch-style model into a near real-time forecaster that can be
called every 5-15 minutes during market hours. It provides:
1) Intraday feature refresh on rolling windows.
2) Prediction intervals around each point forecast.
3) Time series outputs for dashboard charting and CLI monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.features import FeatureEngineer
from src.models import TrackingErrorModel


@dataclass(frozen=True)
class RealTimePrediction:
    """Container for one intraday tracking error forecast."""

    timestamp: pd.Timestamp
    pair: str
    predicted_tracking_error: float
    ci_lower: float
    ci_upper: float
    uncertainty_sigma: float


class RealTimeTrackingErrorPredictor:
    """Serve intraday tracking error forecasts using an already trained model.

    The estimator itself is still the same offline-trained model. This class handles
    real-time mechanics around it: feature refresh cadence, rolling uncertainty
    estimation, and confidence interval calculation.
    """

    def __init__(
        self,
        model: TrackingErrorModel,
        rolling_window: int = 20,
        horizon: int = 1,
        min_history_rows: int = 80,
    ) -> None:
        self.model = model
        self.rolling_window = rolling_window
        self.horizon = horizon
        self.min_history_rows = min_history_rows

    @staticmethod
    def _required_intraday_columns() -> set[str]:
        """Columns expected in the intraday market panel."""
        return {
            "pair",
            "etf_close",
            "benchmark_close",
            "etf_high",
            "etf_low",
            "etf_volume",
            "etf_ticker",
            "benchmark_ticker",
        }

    def _validate_intraday_panel(self, intraday_panel: pd.DataFrame) -> None:
        """Validate input panel before feature generation and scoring."""
        missing = self._required_intraday_columns().difference(intraday_panel.columns)
        if missing:
            raise ValueError(f"Missing required columns for real-time prediction: {sorted(missing)}")

        if not isinstance(intraday_panel.index, pd.DatetimeIndex):
            raise ValueError("Intraday panel index must be a pandas DatetimeIndex.")

    def build_feature_panel(self, intraday_panel: pd.DataFrame) -> pd.DataFrame:
        """Generate model-ready intraday features for all pairs.

        Parameters
        ----------
        intraday_panel:
            Concatenated ETF-benchmark intraday panel returned by MarketDataLoader.
        """
        self._validate_intraday_panel(intraday_panel)

        engineer = FeatureEngineer(rolling_window=self.rolling_window, horizon=self.horizon)
        feature_panel = engineer.transform_universe(intraday_panel)
        feature_panel = feature_panel.sort_index()
        return feature_panel

    def _select_model_input_columns(self, feature_panel: pd.DataFrame) -> list[str]:
        """Return model input columns that are present in the incoming feature panel."""
        expected_columns = self.model.numeric_columns + self.model.categorical_columns
        available_columns = [column for column in expected_columns if column in feature_panel.columns]

        if len(available_columns) != len(expected_columns):
            missing = sorted(set(expected_columns).difference(available_columns))
            raise ValueError(
                "Feature panel does not contain all required model columns. "
                f"Missing columns: {missing}"
            )

        return available_columns

    def _estimate_pair_uncertainty_sigma(
        self,
        pair_history: pd.DataFrame,
        model_input_columns: Iterable[str],
        confidence_window: int,
    ) -> float:
        """Estimate predictive uncertainty from recent residual history.

        We estimate sigma from out-of-sample-like residuals on recent historical rows
        where target values are known. This is a practical and stable production proxy
        when the model is not natively probabilistic.
        """
        history_with_target = pair_history.dropna(subset=["target_te"]).copy()
        if len(history_with_target) < self.min_history_rows:
            # Fallback to recent realized TE volatility if there is not enough target history.
            fallback = float(history_with_target["realized_te"].tail(confidence_window).std(ddof=1))
            return max(fallback, 1e-6)

        recent = history_with_target.tail(confidence_window).copy()
        x_recent = recent[list(model_input_columns)]
        y_recent = recent["target_te"].to_numpy(dtype=float)

        predictions = self.model.predict(x_recent)
        residuals = y_recent - predictions
        sigma = float(np.std(residuals, ddof=1))

        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(recent["realized_te"].std(ddof=1))

        return max(sigma, 1e-6)

    @staticmethod
    def _prediction_interval(prediction: float, sigma: float, confidence_level: float) -> tuple[float, float]:
        """Compute symmetric normal-approximation prediction interval."""
        z_value = float(norm.ppf((1.0 + confidence_level) / 2.0))
        half_width = z_value * sigma

        lower_bound = max(0.0, prediction - half_width)
        upper_bound = prediction + half_width
        return float(lower_bound), float(upper_bound)

    def predict_latest(
        self,
        feature_panel: pd.DataFrame,
        confidence_level: float = 0.95,
        confidence_window: int = 120,
    ) -> pd.DataFrame:
        """Predict the latest intraday tracking error per pair with confidence interval."""
        if not (0.5 < confidence_level < 1.0):
            raise ValueError("confidence_level must be strictly between 0.5 and 1.0")

        model_input_columns = self._select_model_input_columns(feature_panel)

        rows: list[dict[str, float | str | pd.Timestamp]] = []
        for pair_name, pair_history in feature_panel.groupby("pair", observed=True):
            clean_history = pair_history.dropna(subset=model_input_columns)
            if clean_history.empty:
                continue

            latest_row = clean_history.tail(1)
            prediction = float(self.model.predict(latest_row[model_input_columns])[0])
            sigma = self._estimate_pair_uncertainty_sigma(
                pair_history=clean_history,
                model_input_columns=model_input_columns,
                confidence_window=confidence_window,
            )
            lower_bound, upper_bound = self._prediction_interval(prediction, sigma, confidence_level)

            rows.append(
                {
                    "timestamp": latest_row.index[-1],
                    "pair": pair_name,
                    "predicted_tracking_error": prediction,
                    "ci_lower": lower_bound,
                    "ci_upper": upper_bound,
                    "uncertainty_sigma": sigma,
                }
            )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "pair",
                    "predicted_tracking_error",
                    "ci_lower",
                    "ci_upper",
                    "uncertainty_sigma",
                ]
            )

        return pd.DataFrame(rows).sort_values("predicted_tracking_error", ascending=False).reset_index(drop=True)

    def build_prediction_series(
        self,
        feature_panel: pd.DataFrame,
        confidence_level: float = 0.95,
        confidence_window: int = 120,
    ) -> pd.DataFrame:
        """Build an intraday prediction time series for charting.

        The series is computed for every timestamp where model features are available,
        then enriched with a pair-level rolling uncertainty estimate.
        """
        model_input_columns = self._select_model_input_columns(feature_panel)
        output_rows: list[dict[str, float | str | pd.Timestamp]] = []

        for pair_name, pair_history in feature_panel.groupby("pair", observed=True):
            clean_history = pair_history.dropna(subset=model_input_columns).copy()
            if clean_history.empty:
                continue

            x_pair = clean_history[model_input_columns]
            clean_history["predicted_tracking_error"] = self.model.predict(x_pair)

            sigma = self._estimate_pair_uncertainty_sigma(
                pair_history=clean_history,
                model_input_columns=model_input_columns,
                confidence_window=confidence_window,
            )

            bounds = clean_history["predicted_tracking_error"].apply(
                lambda pred: self._prediction_interval(float(pred), sigma, confidence_level)
            )
            clean_history["ci_lower"] = bounds.apply(lambda interval: interval[0])
            clean_history["ci_upper"] = bounds.apply(lambda interval: interval[1])
            clean_history["uncertainty_sigma"] = sigma

            for timestamp, row in clean_history.iterrows():
                output_rows.append(
                    {
                        "timestamp": timestamp,
                        "pair": pair_name,
                        "predicted_tracking_error": float(row["predicted_tracking_error"]),
                        "ci_lower": float(row["ci_lower"]),
                        "ci_upper": float(row["ci_upper"]),
                        "uncertainty_sigma": float(row["uncertainty_sigma"]),
                    }
                )

        if not output_rows:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "pair",
                    "predicted_tracking_error",
                    "ci_lower",
                    "ci_upper",
                    "uncertainty_sigma",
                ]
            )

        return pd.DataFrame(output_rows).sort_values(["pair", "timestamp"]).reset_index(drop=True)
