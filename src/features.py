"""Feature engineering for tracking error forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Create model-ready features and targets from aligned pair data."""

    def __init__(self, rolling_window: int = 20, horizon: int = 1) -> None:
        self.rolling_window = rolling_window
        self.horizon = horizon

    @staticmethod
    def _safe_ratio(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-8) -> pd.Series:
        """Compute a stable ratio with epsilon protection."""
        return numerator / (denominator.abs() + eps)

    def transform_pair(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features for one ETF-benchmark pair DataFrame."""
        df = data.copy().sort_index()

        df["etf_ret_1"] = df["etf_close"].pct_change()
        df["benchmark_ret_1"] = df["benchmark_close"].pct_change()
        df["tracking_diff_1"] = df["etf_ret_1"] - df["benchmark_ret_1"]

        # Lag features to avoid look-ahead bias.
        for lag in [1, 2, 3, 5, 10]:
            df[f"etf_ret_lag_{lag}"] = df["etf_ret_1"].shift(lag)
            df[f"benchmark_ret_lag_{lag}"] = df["benchmark_ret_1"].shift(lag)
            df[f"tracking_diff_lag_{lag}"] = df["tracking_diff_1"].shift(lag)

        for win in [5, 10, 20, 60]:
            # Mix short and medium windows to capture transient shocks and persistent drift.
            df[f"te_roll_std_{win}"] = df["tracking_diff_1"].rolling(win).std()
            df[f"etf_vol_{win}"] = df["etf_ret_1"].rolling(win).std()
            df[f"benchmark_vol_{win}"] = df["benchmark_ret_1"].rolling(win).std()
            df[f"spread_mean_{win}"] = np.log(df["etf_close"] / df["benchmark_close"]).rolling(win).mean()
            df[f"spread_std_{win}"] = np.log(df["etf_close"] / df["benchmark_close"]).rolling(win).std()

        covariance = df["etf_ret_1"].rolling(self.rolling_window).cov(df["benchmark_ret_1"])
        variance = df["benchmark_ret_1"].rolling(self.rolling_window).var()
        df["rolling_beta"] = self._safe_ratio(covariance, variance)

        df["price_ratio"] = self._safe_ratio(df["etf_close"], df["benchmark_close"])
        ratio_mean = df["price_ratio"].rolling(self.rolling_window).mean()
        ratio_std = df["price_ratio"].rolling(self.rolling_window).std()
        # Ratio z-score is a direct dislocation feature used by both model and signal diagnostics.
        df["ratio_zscore"] = self._safe_ratio(df["price_ratio"] - ratio_mean, ratio_std)

        df["volume_change_1"] = df["etf_volume"].pct_change().replace([np.inf, -np.inf], np.nan)
        df["hl_spread"] = self._safe_ratio(df["etf_high"] - df["etf_low"], df["etf_close"])

        df["realized_te"] = df["tracking_diff_1"].rolling(self.rolling_window).std()
        # Target is shifted forward to keep a strict predictive setup (no look-ahead leakage).
        df["target_te"] = df["realized_te"].shift(-self.horizon)

        return df

    def transform_universe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features and target for all pair groups."""
        all_frames: list[pd.DataFrame] = []

        for pair_name, pair_df in data.groupby("pair", observed=True):
            transformed = self.transform_pair(pair_df)
            transformed["pair"] = pair_name
            all_frames.append(transformed)

        result = pd.concat(all_frames, axis=0).sort_index()
        return result
