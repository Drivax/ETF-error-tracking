"""Market data loading utilities for ETF and benchmark pairs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import yfinance as yf


@dataclass
class MarketDataLoader:
    """Download and align ETF and benchmark price series."""

    auto_adjust: bool = True

    @staticmethod
    def _standardize_yf_columns(frame: pd.DataFrame) -> pd.DataFrame:
        """Flatten and normalize yfinance columns into lowercase snake_case names."""
        if isinstance(frame.columns, pd.MultiIndex):
            # yfinance can return MultiIndex columns when batch internals are triggered.
            frame.columns = [col[0] for col in frame.columns]

        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        frame = frame.rename(columns=rename_map)
        return frame

    def _download_ticker(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Download OHLCV data for a single ticker."""
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=self.auto_adjust,
            progress=False,
            threads=False,
        )

        if data.empty:
            raise ValueError(f"No data returned for ticker={ticker}, period={period}, interval={interval}")

        data = self._standardize_yf_columns(data)
        data.index = pd.to_datetime(data.index, utc=True)
        data = data.sort_index()
        return data

    def fetch_pair_data(
        self,
        etf_ticker: str,
        benchmark_ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch and align ETF and benchmark data for a pair."""
        etf = self._download_ticker(etf_ticker, period=period, interval=interval)
        benchmark = self._download_ticker(benchmark_ticker, period=period, interval=interval)

        etf = etf.add_prefix("etf_")
        benchmark = benchmark.add_prefix("benchmark_")

        merged = etf.join(benchmark, how="inner")
        merged["pair"] = f"{etf_ticker}_{benchmark_ticker}"
        merged["etf_ticker"] = etf_ticker
        merged["benchmark_ticker"] = benchmark_ticker

        # Keep rows only where the two close series are jointly observed.
        merged = merged.replace([pd.NA], pd.NA).dropna(subset=["etf_close", "benchmark_close"])
        return merged

    def fetch_universe(
        self,
        pair_mapping: dict[str, str],
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch a concatenated panel for all ETF-benchmark pairs."""
        frames: list[pd.DataFrame] = []
        errors: list[str] = []

        for etf_ticker, benchmark_ticker in pair_mapping.items():
            try:
                pair_df = self.fetch_pair_data(
                    etf_ticker=etf_ticker,
                    benchmark_ticker=benchmark_ticker,
                    period=period,
                    interval=interval,
                )
                frames.append(pair_df)
            except Exception as exc:  # pragma: no cover - defensive branch
                # Continue on partial failures so one broken symbol does not block the full universe.
                errors.append(f"{etf_ticker}->{benchmark_ticker}: {exc}")

        if not frames:
            message = "All pair downloads failed."
            if errors:
                message = f"{message} Errors: {' | '.join(errors)}"
            raise RuntimeError(message)

        panel = pd.concat(frames, axis=0).sort_index()
        return panel
