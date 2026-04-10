"""Statistical arbitrage signal detection for ETF-index dislocations."""

from __future__ import annotations

import numpy as np
import pandas as pd


class ArbitrageDetector:
    """Generate interpretable arbitrage and monitoring signals from spread dynamics."""

    def __init__(
        self,
        window: int = 60,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        volatility_filter_quantile: float = 0.80,
    ) -> None:
        self.window = window
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.volatility_filter_quantile = volatility_filter_quantile

    @staticmethod
    def estimate_half_life(series: pd.Series) -> float:
        """Estimate mean-reversion half-life from an AR(1)-like fit."""
        clean = series.dropna()
        if len(clean) < 10:
            return float("nan")

        lagged = clean.shift(1).dropna()
        aligned = clean.loc[lagged.index]

        if lagged.std() == 0:
            return float("nan")

        slope, intercept = np.polyfit(lagged.values, aligned.values, 1)
        phi = np.clip(abs(slope), 1e-6, 0.999999)
        half_life = -np.log(2.0) / np.log(phi)
        return float(half_life)

    def add_signal_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add spread statistics and discrete signal labels."""
        df = data.copy().sort_index()

        df["spread"] = np.log(df["etf_close"] / df["benchmark_close"])
        df["spread_ret"] = df["spread"].diff()

        rolling_mean = df["spread"].rolling(self.window).mean()
        rolling_std = df["spread"].rolling(self.window).std()

        df["spread_zscore"] = (df["spread"] - rolling_mean) / (rolling_std + 1e-8)
        df["spread_vol"] = df["spread_ret"].rolling(self.window).std()

        # Disable entry in high-noise regimes where z-score edges are less reliable.
        vol_threshold = df["spread_vol"].quantile(self.volatility_filter_quantile)
        df["vol_regime_ok"] = df["spread_vol"] <= vol_threshold

        df["signal"] = "NORMAL"

        long_condition = (df["spread_zscore"] <= -self.zscore_entry) & df["vol_regime_ok"]
        short_condition = (df["spread_zscore"] >= self.zscore_entry) & df["vol_regime_ok"]
        watch_condition = df["spread_zscore"].abs().between(self.zscore_exit, self.zscore_entry)

        df.loc[long_condition, "signal"] = "ARBITRAGE_LONG_ETF_SHORT_BENCH"
        df.loc[short_condition, "signal"] = "ARBITRAGE_SHORT_ETF_LONG_BENCH"
        # WATCH is assigned last so extreme entries keep priority over intermediate states.
        df.loc[watch_condition, "signal"] = "WATCH"

        return df

    def latest_signal(self, data: pd.DataFrame) -> dict[str, float | str]:
        """Compute latest signal snapshot for dashboard and CLI output."""
        signal_df = self.add_signal_columns(data)
        last_row = signal_df.iloc[-1]

        half_life = self.estimate_half_life(signal_df["spread"])

        return {
            "signal": str(last_row["signal"]),
            "spread_zscore": float(last_row["spread_zscore"]),
            "spread_vol": float(last_row["spread_vol"]),
            "half_life": float(half_life),
        }
