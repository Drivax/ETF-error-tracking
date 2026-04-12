"""Market regime detection and adaptive threshold calibration for ETF TE residuals.

This module detects latent residual regimes from tracking error prediction errors:

    residual_t = actual_tracking_error_t - predicted_tracking_error_t

Primary model
-------------
- Hidden Markov Model (Gaussian emissions) via hmmlearn.

Fallback model
--------------
- Gaussian Mixture Model clustering on the same feature space.

Detected regimes
----------------
- Calm
- Stress
- High_Vol

For each run, the detector returns:
- current regime label
- confidence score
- per-regime probability map
- adaptive thresholds for alerts, arbitrage confidence, and anomaly sensitivity
- human-readable explanation
- compact regime history for UI visualization

Design principles
-----------------
- Conservative and risk-aware defaults
- Stable transitions with probability smoothing and persistence rules
- Interpretable state ranking based on residual volatility structure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover - optional dependency in some environments
    GaussianHMM = None  # type: ignore[assignment]


REGIME_LABELS: tuple[str, str, str] = ("Calm", "Stress", "High_Vol")


@dataclass(frozen=True)
class AdaptiveThresholdConfig:
    """Base thresholds used as anchors for regime adjustments.

    Units
    -----
    - alert_tracking_error: percentage points (example: 0.80 means 0.80%)
    - arbitrage_confidence_min: unit interval [0, 1]
    - anomaly_reconstruction_threshold: residual-scale threshold proxy
    """

    alert_tracking_error: float = 0.80
    arbitrage_confidence_min: float = 0.70
    anomaly_reconstruction_threshold: float = 0.050


class RegimeDetector:
    """Detect residual regimes and adapt thresholds for risk controls.

    Parameters
    ----------
    rolling_window:
        Rolling window used for feature extraction.
    smoothing_alpha:
        Exponential smoothing factor applied to regime probabilities.
    min_regime_persistence:
        Minimum run length for a regime before switching is accepted.
    hmm_states:
        Number of latent states (must be 3 for Calm/Stress/High_Vol mapping).
    random_state:
        Seed for reproducibility.
    base_thresholds:
        Baseline thresholds used to derive regime-adaptive thresholds.
    """

    def __init__(
        self,
        rolling_window: int = 24,
        smoothing_alpha: float = 0.35,
        min_regime_persistence: int = 4,
        hmm_states: int = 3,
        random_state: int = 42,
        base_thresholds: AdaptiveThresholdConfig | None = None,
    ) -> None:
        if rolling_window < 10:
            raise ValueError("rolling_window must be >= 10")
        if not (0.05 <= smoothing_alpha <= 0.95):
            raise ValueError("smoothing_alpha must be in [0.05, 0.95]")
        if min_regime_persistence < 1:
            raise ValueError("min_regime_persistence must be >= 1")
        if hmm_states != 3:
            raise ValueError("hmm_states must be 3 for Calm/Stress/High_Vol")

        self.rolling_window = rolling_window
        self.smoothing_alpha = smoothing_alpha
        self.min_regime_persistence = min_regime_persistence
        self.hmm_states = hmm_states
        self.random_state = random_state
        self.base_thresholds = base_thresholds or AdaptiveThresholdConfig()

    @staticmethod
    def _to_series(residuals: pd.Series | np.ndarray | list[float]) -> pd.Series:
        if isinstance(residuals, pd.Series):
            return residuals.astype(float).dropna()
        return pd.Series(residuals, dtype=float).dropna()

    @staticmethod
    def _rolling_autocorr(series: pd.Series, window: int, lag: int = 1) -> pd.Series:
        shifted = series.shift(lag)
        return series.rolling(window=window, min_periods=max(8, window // 2)).corr(shifted)

    def _build_feature_frame(self, residuals: pd.Series) -> pd.DataFrame:
        """Construct regime-sensitive residual features.

        Required feature set:
        - rolling mean
        - rolling volatility
        - rolling autocorrelation
        - rolling skewness
        - rolling kurtosis
        """
        window = self.rolling_window

        frame = pd.DataFrame(index=residuals.index)
        frame["residual"] = residuals
        frame["rolling_mean"] = residuals.rolling(window, min_periods=max(8, window // 2)).mean()
        frame["rolling_volatility"] = residuals.rolling(window, min_periods=max(8, window // 2)).std(ddof=1)
        frame["rolling_autocorr"] = self._rolling_autocorr(residuals, window, lag=1)

        frame["rolling_skewness"] = residuals.rolling(window, min_periods=max(8, window // 2)).apply(
            lambda x: float(skew(x, bias=False)), raw=False
        )
        frame["rolling_kurtosis"] = residuals.rolling(window, min_periods=max(8, window // 2)).apply(
            lambda x: float(kurtosis(x, fisher=True, bias=False)), raw=False
        )

        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        return frame

    @staticmethod
    def _state_probability_frame(probabilities: np.ndarray, state_count: int, index: pd.Index) -> pd.DataFrame:
        columns = [f"state_{i}" for i in range(state_count)]
        return pd.DataFrame(probabilities, columns=columns, index=index)

    def _fit_hmm(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Fit HMM and return (state_path, state_probabilities)."""
        if GaussianHMM is None:
            return None

        hmm_model = GaussianHMM(
            n_components=self.hmm_states,
            covariance_type="full",
            n_iter=400,
            random_state=self.random_state,
        )

        try:
            hmm_model.fit(x)
            state_path = hmm_model.predict(x)
            state_prob = hmm_model.predict_proba(x)
            return state_path.astype(int), state_prob.astype(float)
        except Exception:
            return None

    def _fit_gmm_fallback(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit GMM fallback and return (cluster_path, soft_probabilities)."""
        gmm = GaussianMixture(
            n_components=self.hmm_states,
            covariance_type="full",
            random_state=self.random_state,
            n_init=10,
            reg_covar=1e-6,
        )
        gmm.fit(x)
        cluster_path = gmm.predict(x)
        probabilities = gmm.predict_proba(x)
        return cluster_path.astype(int), probabilities.astype(float)

    @staticmethod
    def _infer_regime_mapping(feature_frame: pd.DataFrame, state_path: np.ndarray) -> dict[int, str]:
        """Map latent states to domain regimes using volatility ranking.

        Lowest residual volatility -> Calm
        Median residual volatility -> Stress
        Highest residual volatility -> High_Vol
        """
        ranked: list[tuple[int, float]] = []
        for state_id in np.unique(state_path):
            state_mask = state_path == state_id
            state_vol = float(feature_frame.loc[state_mask, "rolling_volatility"].mean())
            ranked.append((int(state_id), state_vol))

        ranked.sort(key=lambda item: item[1])
        if len(ranked) != 3:
            raise ValueError("Unexpected number of latent states while mapping regimes")

        return {
            ranked[0][0]: "Calm",
            ranked[1][0]: "Stress",
            ranked[2][0]: "High_Vol",
        }

    def _smooth_probabilities(self, raw_prob_df: pd.DataFrame) -> pd.DataFrame:
        """Apply exponential smoothing to reduce noisy transitions."""
        smoothed = raw_prob_df.ewm(alpha=self.smoothing_alpha, adjust=False).mean()
        row_sums = smoothed.sum(axis=1).replace(0.0, np.nan)
        return smoothed.div(row_sums, axis=0).fillna(1.0 / smoothed.shape[1])

    def _enforce_min_persistence(self, regime_path: list[str]) -> list[str]:
        """Suppress short-lived regime flips to avoid whipsawing."""
        if len(regime_path) <= 1 or self.min_regime_persistence <= 1:
            return regime_path

        output = regime_path.copy()
        i = 1
        while i < len(output):
            if output[i] == output[i - 1]:
                i += 1
                continue

            # Find the extent of the new regime run.
            run_start = i
            run_label = output[i]
            run_end = i
            while run_end + 1 < len(output) and output[run_end + 1] == run_label:
                run_end += 1

            run_length = run_end - run_start + 1
            if run_length < self.min_regime_persistence:
                for j in range(run_start, run_end + 1):
                    output[j] = output[run_start - 1]

            i = run_end + 1

        return output

    def _adaptive_thresholds(self, regime: str) -> dict[str, float]:
        """Derive regime-dependent operational thresholds.

        Stress and high-volatility regimes tighten monitoring and increase execution
        conservatism for arbitrage signals.
        """
        multipliers = {
            "Calm": {
                "alert_tracking_error": 1.20,
                "arbitrage_confidence_min": 0.96,
                "anomaly_reconstruction_threshold": 1.12,
            },
            "Stress": {
                "alert_tracking_error": 0.82,
                "arbitrage_confidence_min": 1.08,
                "anomaly_reconstruction_threshold": 0.84,
            },
            "High_Vol": {
                "alert_tracking_error": 0.65,
                "arbitrage_confidence_min": 1.16,
                "anomaly_reconstruction_threshold": 0.70,
            },
        }

        regime_multiplier = multipliers[regime]
        calibrated = {
            "alert_tracking_error": self.base_thresholds.alert_tracking_error
            * regime_multiplier["alert_tracking_error"],
            "arbitrage_confidence_min": self.base_thresholds.arbitrage_confidence_min
            * regime_multiplier["arbitrage_confidence_min"],
            "anomaly_reconstruction_threshold": self.base_thresholds.anomaly_reconstruction_threshold
            * regime_multiplier["anomaly_reconstruction_threshold"],
        }

        # Bound confidence threshold for robust behavior.
        calibrated["arbitrage_confidence_min"] = float(
            np.clip(calibrated["arbitrage_confidence_min"], 0.55, 0.95)
        )

        return {key: float(round(value, 6)) for key, value in calibrated.items()}

    @staticmethod
    def _explanation(current_regime: str, latest_features: pd.Series) -> str:
        parts: list[str] = []

        vol = float(latest_features.get("rolling_volatility", 0.0))
        mean = float(latest_features.get("rolling_mean", 0.0))
        ac = float(latest_features.get("rolling_autocorr", 0.0))
        sk = float(latest_features.get("rolling_skewness", 0.0))
        kt = float(latest_features.get("rolling_kurtosis", 0.0))

        if vol > 0:
            parts.append(f"Residual volatility at {vol:.5f}")
        if abs(mean) > 0:
            direction = "positive" if mean > 0 else "negative"
            parts.append(f"residual mean drift is {direction} ({mean:.5f})")
        if np.isfinite(ac):
            parts.append(f"lag-1 autocorrelation is {ac:.2f}")
        if np.isfinite(sk):
            parts.append(f"skewness is {sk:.2f}")
        if np.isfinite(kt):
            parts.append(f"kurtosis is {kt:.2f}")

        descriptor = {
            "Calm": "stable residual behavior",
            "Stress": "elevated stress signatures",
            "High_Vol": "high-volatility residual dynamics",
        }[current_regime]

        joined = "; ".join(parts[:3]) if parts else "residual feature dynamics"
        return f"{descriptor} detected based on {joined}."

    def detect_regime(
        self,
        residuals: pd.Series | np.ndarray | list[float],
        history_points: int = 100,
    ) -> dict[str, Any]:
        """Detect current regime and produce adaptive thresholds.

        Parameters
        ----------
        residuals:
            Residual time series defined as actual - predicted tracking error.
        history_points:
            Number of latest points to include in returned regime history.
        """
        residual_series = self._to_series(residuals)
        if len(residual_series) < max(60, self.rolling_window * 2):
            default_regime = "Calm"
            default_prob = {"Calm": 1.0, "Stress": 0.0, "High_Vol": 0.0}
            return {
                "current_regime": default_regime,
                "confidence": 1.0,
                "regime_probability": default_prob,
                "adaptive_thresholds": self._adaptive_thresholds(default_regime),
                "explanation": (
                    "Insufficient residual history for robust regime estimation; "
                    "defaulting to Calm risk controls."
                ),
                "model_used": "insufficient_history_default",
                "regime_history": [],
            }

        feature_frame = self._build_feature_frame(residual_series)
        if len(feature_frame) < max(40, self.rolling_window):
            default_regime = "Calm"
            default_prob = {"Calm": 1.0, "Stress": 0.0, "High_Vol": 0.0}
            return {
                "current_regime": default_regime,
                "confidence": 1.0,
                "regime_probability": default_prob,
                "adaptive_thresholds": self._adaptive_thresholds(default_regime),
                "explanation": (
                    "Feature extraction yielded too few valid rows; using Calm "
                    "threshold profile as safe fallback."
                ),
                "model_used": "feature_fallback_default",
                "regime_history": [],
            }

        scaler = StandardScaler()
        x = scaler.fit_transform(
            feature_frame[
                [
                    "rolling_mean",
                    "rolling_volatility",
                    "rolling_autocorr",
                    "rolling_skewness",
                    "rolling_kurtosis",
                ]
            ]
        )

        hmm_result = self._fit_hmm(x)
        if hmm_result is not None:
            state_path, state_prob = hmm_result
            model_used = "HMM"
        else:
            state_path, state_prob = self._fit_gmm_fallback(x)
            model_used = "GMM_fallback"

        # HMM can occasionally converge to fewer than 3 effective states on short or
        # highly homogeneous samples. In that case we switch to the robust GMM path.
        try:
            state_to_regime = self._infer_regime_mapping(feature_frame, state_path)
        except ValueError:
            state_path, state_prob = self._fit_gmm_fallback(x)
            model_used = "GMM_fallback"
            state_to_regime = self._infer_regime_mapping(feature_frame, state_path)

        raw_prob_df = self._state_probability_frame(state_prob, self.hmm_states, index=feature_frame.index)
        smoothed_prob_df = self._smooth_probabilities(raw_prob_df)

        regime_probability_df = pd.DataFrame(index=feature_frame.index, columns=REGIME_LABELS, dtype=float)
        for state_column in raw_prob_df.columns:
            state_id = int(state_column.split("_")[1])
            regime_label = state_to_regime[state_id]
            regime_probability_df[regime_label] = smoothed_prob_df[state_column].to_numpy()

        regime_probability_df = regime_probability_df.fillna(0.0)
        row_sums = regime_probability_df.sum(axis=1).replace(0.0, np.nan)
        regime_probability_df = regime_probability_df.div(row_sums, axis=0).fillna(1.0 / 3.0)

        raw_regime_path = regime_probability_df.idxmax(axis=1).tolist()
        stable_regime_path = self._enforce_min_persistence(raw_regime_path)

        current_regime = stable_regime_path[-1]
        latest_probability_row = regime_probability_df.iloc[-1]
        confidence = float(np.clip(latest_probability_row[current_regime], 0.0, 1.0))

        regime_prob = {
            "Calm": float(round(latest_probability_row["Calm"], 6)),
            "Stress": float(round(latest_probability_row["Stress"], 6)),
            "High_Vol": float(round(latest_probability_row["High_Vol"], 6)),
        }

        adaptive_thresholds = self._adaptive_thresholds(current_regime)
        explanation = self._explanation(current_regime, feature_frame.iloc[-1])

        history_start = max(0, len(feature_frame) - int(history_points))
        history_index = feature_frame.index[history_start:]
        history_labels = stable_regime_path[history_start:]
        history_confidence = [
            float(regime_probability_df.loc[idx, label])
            for idx, label in zip(history_index, history_labels)
        ]

        regime_history = [
            {
                "timestamp": pd.Timestamp(idx).isoformat(),
                "regime": label,
                "confidence": float(round(conf, 6)),
            }
            for idx, label, conf in zip(history_index, history_labels, history_confidence)
        ]

        return {
            "current_regime": current_regime,
            "confidence": float(round(confidence, 6)),
            "regime_probability": regime_prob,
            "adaptive_thresholds": adaptive_thresholds,
            "explanation": explanation,
            "model_used": model_used,
            "regime_history": regime_history,
        }
