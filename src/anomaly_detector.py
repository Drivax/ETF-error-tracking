"""Residual anomaly detection for ETF tracking error monitoring.

This module focuses on structural anomaly detection in tracking error residuals,
where residuals are defined as:

    residual = actual_tracking_error - predicted_tracking_error

Why this matters:
- Large residuals are not always anomalous (normal during volatile sessions).
- Small residuals can still be anomalous if their *pattern* changes (regime shift,
  autocorrelation break, volatility clustering, etc.).

The implementation intentionally combines three complementary lenses:
1) Neural autoencoder (MLP-based) to learn normal residual structure.
2) Isolation Forest as a fast, non-parametric baseline detector.
3) Regime diagnostics based on rolling mean/volatility/autocorrelation shifts.

The public API is designed for production pipelines and dashboard usage:
- fit(...): train detector on historical residuals.
- score(...): score residual timeline and return per-timestamp diagnostics.
- fit_score(...): convenience method for one-shot workflows.
- latest_result(...): get the most recent anomaly decision as a dictionary ready
  for UI display or alerting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DetectorThresholds:
    """Container for model and statistical thresholds learned during training."""

    autoencoder_error: float
    isolation_score: float
    mean_shift: float
    std_ratio: float
    autocorr_shift: float


class ResidualAnomalyDetector:
    """Detect structural anomalies in ETF tracking error residuals.

    The detector uses an ensemble approach where each component captures a different
    failure mode:
    - Autoencoder reconstruction error catches unusual joint feature geometry.
    - Isolation Forest catches sparse and hard-to-model outliers quickly.
    - Statistical shift tests catch regime transitions in time-series behavior.

    Parameters
    ----------
    short_window:
        Window used for "recent" behavior statistics.
    long_window:
        Window used for "baseline" behavior statistics.
    contamination:
        Expected anomaly proportion used by model thresholds.
    normal_quantile:
        Quantile used to pick an initial "mostly normal" subset for training.
    random_state:
        Reproducibility seed.
    """

    def __init__(
        self,
        short_window: int = 12,
        long_window: int = 36,
        contamination: float = 0.05,
        normal_quantile: float = 0.85,
        random_state: int = 42,
    ) -> None:
        if short_window < 5:
            raise ValueError("short_window must be >= 5 for stable local statistics")
        if long_window <= short_window:
            raise ValueError("long_window must be greater than short_window")
        if not (0.0 < contamination < 0.5):
            raise ValueError("contamination must be in (0, 0.5)")
        if not (0.5 < normal_quantile < 1.0):
            raise ValueError("normal_quantile must be in (0.5, 1.0)")

        self.short_window = short_window
        self.long_window = long_window
        self.contamination = contamination
        self.normal_quantile = normal_quantile
        self.random_state = random_state

        # Models are initialized lazily during fit.
        self._scaler: StandardScaler | None = None
        self._autoencoder: MLPRegressor | None = None
        self._isolation_forest: IsolationForest | None = None
        self._thresholds: DetectorThresholds | None = None
        self._is_fitted = False

    @staticmethod
    def _as_series(values: pd.Series | np.ndarray | list[float], name: str) -> pd.Series:
        """Convert values to float Series while preserving index when available."""
        if isinstance(values, pd.Series):
            return values.astype(float)
        return pd.Series(values, dtype=float, name=name)

    @staticmethod
    def _rolling_autocorr(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
        """Compute rolling autocorrelation for a given lag.

        Pandas provides a concise implementation through rolling corr against a lagged
        version of the same series.
        """
        shifted = series.shift(lag)
        return series.rolling(window=window, min_periods=max(5, window // 2)).corr(shifted)

    def _build_feature_frame(self, residuals: pd.Series) -> pd.DataFrame:
        """Build structural features from residual history.

        Feature design emphasizes both amplitude and temporal structure:
        - Levels and first differences
        - Rolling mean and volatility dynamics
        - Z-score and volatility ratio (short vs long regime)
        - Rolling lag-1 autocorrelation shifts
        - Sign persistence as a simple clustering/prolonged-bias proxy
        """
        eps = 1e-8
        frame = pd.DataFrame(index=residuals.index)
        frame["residual"] = residuals
        frame["abs_residual"] = residuals.abs()
        frame["residual_change"] = residuals.diff()

        frame["rolling_mean_short"] = residuals.rolling(
            self.short_window,
            min_periods=max(5, self.short_window // 2),
        ).mean()
        frame["rolling_mean_long"] = residuals.rolling(
            self.long_window,
            min_periods=max(8, self.long_window // 2),
        ).mean()

        frame["rolling_std_short"] = residuals.rolling(
            self.short_window,
            min_periods=max(5, self.short_window // 2),
        ).std(ddof=1)
        frame["rolling_std_long"] = residuals.rolling(
            self.long_window,
            min_periods=max(8, self.long_window // 2),
        ).std(ddof=1)

        frame["rolling_zscore"] = (
            (frame["residual"] - frame["rolling_mean_long"]) / (frame["rolling_std_long"] + eps)
        )
        frame["volatility_ratio"] = frame["rolling_std_short"] / (frame["rolling_std_long"] + eps)
        frame["mean_shift"] = (frame["rolling_mean_short"] - frame["rolling_mean_long"]).abs()

        frame["autocorr_short"] = self._rolling_autocorr(residuals, self.short_window, lag=1)
        frame["autocorr_long"] = self._rolling_autocorr(residuals, self.long_window, lag=1)
        frame["autocorr_shift"] = (frame["autocorr_short"] - frame["autocorr_long"]).abs()

        # Consecutive same-sign residuals can indicate persistent directional drift.
        sign = np.sign(frame["residual"]).replace(0.0, np.nan)
        same_sign = (sign == sign.shift(1)).astype(float)
        frame["sign_persistence"] = same_sign.rolling(
            self.short_window,
            min_periods=max(5, self.short_window // 2),
        ).mean()

        # Replace inf values before final cleaning.
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna()
        return frame

    def _fit_thresholds(self, train_frame: pd.DataFrame, recon_error: np.ndarray, iso_scores: np.ndarray) -> None:
        """Calibrate thresholds from training distributions.

        The thresholds are intentionally percentile-based to remain robust and avoid
        hard-coded assumptions about residual scale.
        """
        upper_pct = 100.0 * (1.0 - self.contamination)

        self._thresholds = DetectorThresholds(
            autoencoder_error=float(np.percentile(recon_error, upper_pct)),
            isolation_score=float(np.percentile(iso_scores, upper_pct)),
            mean_shift=float(np.percentile(train_frame["mean_shift"].to_numpy(), 95.0)),
            std_ratio=float(np.percentile(np.abs(train_frame["volatility_ratio"] - 1.0).to_numpy(), 95.0)),
            autocorr_shift=float(np.percentile(train_frame["autocorr_shift"].to_numpy(), 95.0)),
        )

    @staticmethod
    def _normalize_excess(value: float, threshold: float) -> float:
        """Map threshold exceedance into [0, 1] for confidence aggregation."""
        if threshold <= 0:
            return 0.0
        excess = max(0.0, value - threshold) / threshold
        # Saturating transform prevents one extreme metric from dominating confidence.
        return float(1.0 - np.exp(-2.0 * excess))

    def fit(
        self,
        actual_tracking_error: pd.Series | np.ndarray | list[float],
        predicted_tracking_error: pd.Series | np.ndarray | list[float],
    ) -> "ResidualAnomalyDetector":
        """Train the detector on historical residual behavior.

        Notes
        -----
        - We first derive residuals using actual - predicted.
        - We then build structural features and identify a mostly-normal subset
          based on absolute residual quantiles for stable model fitting.
        """
        actual = self._as_series(actual_tracking_error, name="actual_tracking_error")
        predicted = self._as_series(predicted_tracking_error, name="predicted_tracking_error")

        if len(actual) != len(predicted):
            raise ValueError("actual_tracking_error and predicted_tracking_error must have same length")
        if len(actual) < self.long_window * 2:
            raise ValueError(
                f"Need at least {self.long_window * 2} observations for robust anomaly training"
            )

        residuals = (actual - predicted).rename("residual")
        feature_frame = self._build_feature_frame(residuals)
        if len(feature_frame) < self.long_window:
            raise ValueError("Insufficient valid feature rows after rolling statistics")

        normal_cutoff = float(feature_frame["abs_residual"].quantile(self.normal_quantile))
        normal_mask = feature_frame["abs_residual"] <= normal_cutoff
        train_frame = feature_frame.loc[normal_mask].copy()

        # Fallback: if the filtered subset is too small, use all rows.
        if len(train_frame) < max(80, self.long_window):
            train_frame = feature_frame.copy()

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(train_frame)

        autoencoder = MLPRegressor(
            hidden_layer_sizes=(64, 24, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=500,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
        # Autoencoder objective: reconstruct the same feature vector.
        autoencoder.fit(x_train_scaled, x_train_scaled)

        x_train_recon = autoencoder.predict(x_train_scaled)
        train_recon_error = np.mean((x_train_scaled - x_train_recon) ** 2, axis=1)

        isolation_forest = IsolationForest(
            n_estimators=300,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        isolation_forest.fit(x_train_scaled)
        # decision_function: higher = more normal, so we negate for anomaly intensity.
        train_iso_score = -isolation_forest.decision_function(x_train_scaled)

        self._fit_thresholds(train_frame, train_recon_error, train_iso_score)
        self._scaler = scaler
        self._autoencoder = autoencoder
        self._isolation_forest = isolation_forest
        self._is_fitted = True
        return self

    def _build_result_row(self, row: pd.Series, thresholds: DetectorThresholds) -> dict[str, Any]:
        """Convert scored metrics into a human-readable anomaly decision."""
        ae_flag = bool(row["autoencoder_error"] > thresholds.autoencoder_error)
        iso_flag = bool(row["isolation_score"] > thresholds.isolation_score)
        mean_shift_flag = bool(row["mean_shift"] > thresholds.mean_shift)
        std_shift_flag = bool(abs(row["volatility_ratio"] - 1.0) > thresholds.std_ratio)
        autocorr_shift_flag = bool(row["autocorr_shift"] > thresholds.autocorr_shift)

        regime_shift_flag = bool(mean_shift_flag or std_shift_flag or autocorr_shift_flag)

        model_votes = int(ae_flag) + int(iso_flag)
        stat_votes = int(mean_shift_flag) + int(std_shift_flag) + int(autocorr_shift_flag)

        anomaly_detected = bool((model_votes >= 1 and stat_votes >= 1) or (model_votes + stat_votes >= 3))

        # Heuristic anomaly taxonomy prioritizes regime changes over pure magnitude spikes.
        if anomaly_detected and regime_shift_flag:
            anomaly_type = "regime_change"
        elif anomaly_detected and abs(row["rolling_zscore"]) >= 2.5:
            anomaly_type = "magnitude"
        elif anomaly_detected:
            anomaly_type = "structural"
        else:
            anomaly_type = "none"

        ae_component = self._normalize_excess(float(row["autoencoder_error"]), thresholds.autoencoder_error)
        iso_component = self._normalize_excess(float(row["isolation_score"]), thresholds.isolation_score)
        mean_component = self._normalize_excess(float(row["mean_shift"]), thresholds.mean_shift)
        std_component = self._normalize_excess(abs(float(row["volatility_ratio"]) - 1.0), thresholds.std_ratio)
        ac_component = self._normalize_excess(float(row["autocorr_shift"]), thresholds.autocorr_shift)

        confidence = (
            0.30 * ae_component
            + 0.25 * iso_component
            + 0.20 * mean_component
            + 0.15 * std_component
            + 0.10 * ac_component
        )
        if anomaly_detected:
            confidence = min(1.0, confidence + 0.15)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        explanation_parts: list[str] = []
        if ae_flag:
            explanation_parts.append("high autoencoder reconstruction error")
        if iso_flag:
            explanation_parts.append("Isolation Forest indicates low-likelihood point")
        if std_shift_flag:
            explanation_parts.append("sudden volatility regime shift")
        if mean_shift_flag:
            explanation_parts.append("rolling mean drift from long-run baseline")
        if autocorr_shift_flag:
            explanation_parts.append("autocorrelation structure changed")

        if not explanation_parts:
            explanation = "Residual behavior is consistent with the recent normal regime."
        else:
            explanation = "; ".join(explanation_parts).capitalize() + "."

        recommended_action = (
            "Monitor ETF closely - potential arbitrage or liquidity issue"
            if anomaly_detected
            else "No immediate action required - continue standard monitoring"
        )

        return {
            "anomaly_detected": anomaly_detected,
            "anomaly_type": anomaly_type,
            "confidence": confidence,
            "score": float(row["ensemble_score"]),
            "explanation": explanation,
            "recommended_action": recommended_action,
        }

    def score(
        self,
        actual_tracking_error: pd.Series | np.ndarray | list[float],
        predicted_tracking_error: pd.Series | np.ndarray | list[float],
    ) -> pd.DataFrame:
        """Score residual timeline and return anomaly diagnostics per timestamp.

        Returns
        -------
        pd.DataFrame
            Indexed by timestamp/row index, with model scores, regime metrics, and
            final decision fields that are immediately consumable by dashboards.
        """
        if not self._is_fitted or self._scaler is None or self._autoencoder is None or self._isolation_forest is None:
            raise RuntimeError("Detector is not fitted. Call fit(...) before score(...).")
        if self._thresholds is None:
            raise RuntimeError("Thresholds are missing. Refit the detector.")

        actual = self._as_series(actual_tracking_error, name="actual_tracking_error")
        predicted = self._as_series(predicted_tracking_error, name="predicted_tracking_error")

        if len(actual) != len(predicted):
            raise ValueError("actual_tracking_error and predicted_tracking_error must have same length")

        residuals = (actual - predicted).rename("residual")
        feature_frame = self._build_feature_frame(residuals)
        if feature_frame.empty:
            return pd.DataFrame()

        x_scaled = self._scaler.transform(feature_frame)

        reconstruction = self._autoencoder.predict(x_scaled)
        feature_frame["autoencoder_error"] = np.mean((x_scaled - reconstruction) ** 2, axis=1)
        feature_frame["isolation_score"] = -self._isolation_forest.decision_function(x_scaled)

        thresholds = self._thresholds
        feature_frame["ensemble_score"] = (
            0.45 * feature_frame["autoencoder_error"] / (thresholds.autoencoder_error + 1e-8)
            + 0.35 * feature_frame["isolation_score"] / (thresholds.isolation_score + 1e-8)
            + 0.10 * feature_frame["mean_shift"] / (thresholds.mean_shift + 1e-8)
            + 0.05 * np.abs(feature_frame["volatility_ratio"] - 1.0) / (thresholds.std_ratio + 1e-8)
            + 0.05 * feature_frame["autocorr_shift"] / (thresholds.autocorr_shift + 1e-8)
        )

        decision_rows = feature_frame.apply(lambda row: self._build_result_row(row, thresholds), axis=1)
        decisions = pd.DataFrame(decision_rows.tolist(), index=feature_frame.index)

        return pd.concat([feature_frame, decisions], axis=1)

    def fit_score(
        self,
        actual_tracking_error: pd.Series | np.ndarray | list[float],
        predicted_tracking_error: pd.Series | np.ndarray | list[float],
    ) -> pd.DataFrame:
        """Fit detector and score in one call."""
        self.fit(actual_tracking_error=actual_tracking_error, predicted_tracking_error=predicted_tracking_error)
        return self.score(actual_tracking_error=actual_tracking_error, predicted_tracking_error=predicted_tracking_error)

    def latest_result(
        self,
        actual_tracking_error: pd.Series | np.ndarray | list[float],
        predicted_tracking_error: pd.Series | np.ndarray | list[float],
    ) -> dict[str, Any]:
        """Return latest anomaly result as a clean dictionary for alerting/UI.

        Output schema mirrors the format requested by downstream dashboard logic.
        """
        scored = self.score(actual_tracking_error=actual_tracking_error, predicted_tracking_error=predicted_tracking_error)
        if scored.empty:
            return {
                "anomaly_detected": False,
                "anomaly_type": "none",
                "confidence": 0.0,
                "score": 0.0,
                "explanation": "Insufficient residual history for anomaly detection.",
                "recommended_action": "Collect more observations before triggering decisions",
            }

        latest = scored.iloc[-1]
        return {
            "anomaly_detected": bool(latest["anomaly_detected"]),
            "anomaly_type": str(latest["anomaly_type"]),
            "confidence": float(latest["confidence"]),
            "score": float(latest["score"]),
            "explanation": str(latest["explanation"]),
            "recommended_action": str(latest["recommended_action"]),
        }
