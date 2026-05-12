"""Automatic report generator for walk-forward backtest results.

Ranks per-pair performance and generates visualizations for drawdown,
spread capture distribution, and KPI comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class BacktestReportGenerator:
    """Generate comprehensive backtest reports from walk-forward results."""

    def __init__(self, output_dir: str | Path = "artifacts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")

    def load_backtest_results(
        self,
        predictions_csv: str | Path,
        paper_trades_csv: str | Path,
        alerts_csv: str | Path | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load walk-forward backtest result files."""
        predictions = pd.read_csv(predictions_csv)
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
        predictions = predictions.sort_values("timestamp")

        paper_trades = pd.read_csv(paper_trades_csv)
        paper_trades["timestamp"] = pd.to_datetime(paper_trades["timestamp"])
        paper_trades["entry_timestamp"] = pd.to_datetime(paper_trades["entry_timestamp"])
        paper_trades["exit_timestamp"] = pd.to_datetime(paper_trades["exit_timestamp"])
        paper_trades = paper_trades.sort_values("timestamp")

        alerts = None
        if alerts_csv and Path(alerts_csv).exists():
            alerts = pd.read_csv(alerts_csv)
            alerts["timestamp"] = pd.to_datetime(alerts["timestamp"])
            alerts = alerts.sort_values("timestamp")

        return {
            "predictions": predictions,
            "paper_trades": paper_trades,
            "alerts": alerts,
        }

    @staticmethod
    def compute_per_pair_metrics(paper_trades: pd.DataFrame) -> pd.DataFrame:
        """Compute trading KPIs aggregated by pair."""
        metrics: list[dict[str, Any]] = []

        for pair, group in paper_trades.groupby("pair"):
            actionable = group[(group["action"] != "HOLD") & (group["executed"])].copy()

            total_signals = len(group)
            executed_signals = len(actionable)
            precision = float((actionable["realized_net_profit_bps"] > 0).mean()) if not actionable.empty else 0.0
            hit_ratio = float(actionable["hit"].mean()) if not actionable.empty else 0.0
            avg_spread_capture = float(actionable["realized_spread_capture"].mean()) if not actionable.empty else 0.0
            total_pnl = float(actionable["realized_net_profit"].sum()) if not actionable.empty else 0.0
            avg_confidence = float(actionable["confidence"].mean()) if not actionable.empty else 0.0

            pnl_series = actionable.sort_values("exit_timestamp")["realized_net_profit"] if not actionable.empty else pd.Series(dtype=float)
            if not pnl_series.empty:
                equity = pnl_series.cumsum()
                running_max = equity.cummax()
                drawdown = equity - running_max
                max_dd = float(drawdown.min())
            else:
                max_dd = 0.0

            metrics.append(
                {
                    "pair": pair,
                    "total_signals": total_signals,
                    "executed_signals": executed_signals,
                    "precision": precision,
                    "hit_ratio": hit_ratio,
                    "avg_spread_capture": avg_spread_capture,
                    "total_pnl": total_pnl,
                    "avg_confidence": avg_confidence,
                    "max_drawdown": max_dd,
                }
            )

        metrics_df = pd.DataFrame(metrics).sort_values("total_pnl", ascending=False)
        return metrics_df

    def plot_per_pair_performance(self, metrics: pd.DataFrame, report_prefix: str) -> None:
        """Create bar charts ranking pairs by performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Per-Pair Performance Summary ({report_prefix})", fontsize=16, fontweight="bold")

        # Precision
        ax = axes[0, 0]
        bars = ax.barh(metrics["pair"], metrics["precision"], color="steelblue")
        ax.set_xlabel("Precision (Share of Profitable Trades)")
        ax.set_title("Trade Precision by Pair")
        ax.set_xlim([0, 1])
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2, f"{width:.2%}", va="center")

        # Hit Ratio
        ax = axes[0, 1]
        bars = ax.barh(metrics["pair"], metrics["hit_ratio"], color="darkgreen")
        ax.set_xlabel("Hit Ratio (Directionally Correct & Profitable)")
        ax.set_title("Hit Ratio by Pair")
        ax.set_xlim([0, 1])
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2, f"{width:.2%}", va="center")

        # Total PnL
        ax = axes[1, 0]
        colors = ["green" if x > 0 else "red" for x in metrics["total_pnl"]]
        bars = ax.barh(metrics["pair"], metrics["total_pnl"], color=colors)
        ax.set_xlabel("Total Realized PnL (USD)")
        ax.set_title("Total PnL by Pair")
        ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + (max(metrics["total_pnl"]) * 0.02 if width > 0 else -max(metrics["total_pnl"]) * 0.02), bar.get_y() + bar.get_height() / 2, f"${width:,.0f}", va="center", fontsize=9)

        # Executed Signals Count
        ax = axes[1, 1]
        bars = ax.barh(metrics["pair"], metrics["executed_signals"], color="coral")
        ax.set_xlabel("Number of Executed Trades")
        ax.set_title("Trade Volume by Pair")
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.2, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va="center")

        plt.tight_layout()
        output_path = self.output_dir / f"backtest_per_pair_metrics_{report_prefix}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def plot_spread_capture_distribution(self, paper_trades: pd.DataFrame, report_prefix: str) -> None:
        """Create distribution plot of realized spread capture by pair."""
        actionable = paper_trades[(paper_trades["action"] != "HOLD") & (paper_trades["executed"])].copy()
        if actionable.empty:
            print("No actionable trades for spread capture distribution plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        pairs = sorted(actionable["pair"].unique())
        data_to_plot = [actionable[actionable["pair"] == pair]["realized_spread_capture"].dropna().values for pair in pairs]

        bp = ax.boxplot(data_to_plot, labels=pairs, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")

        ax.set_ylabel("Realized Spread Capture (normalized)")
        ax.set_title(f"Spread Capture Distribution by Pair ({report_prefix})")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        output_path = self.output_dir / f"spread_capture_distribution_{report_prefix}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def plot_drawdown_time_series(self, paper_trades: pd.DataFrame, report_prefix: str) -> None:
        """Create drawdown time series plot per pair."""
        actionable = paper_trades[(paper_trades["action"] != "HOLD") & (paper_trades["executed"])].copy()
        if actionable.empty:
            print("No actionable trades for drawdown plot")
            return

        pairs = sorted(actionable["pair"].unique())
        fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 3 * len(pairs)))
        if len(pairs) == 1:
            axes = [axes]

        for idx, (ax, pair) in enumerate(zip(axes, pairs)):
            pair_trades = actionable[actionable["pair"] == pair].sort_values("exit_timestamp")
            if pair_trades.empty:
                continue

            pnl_series = pair_trades["realized_net_profit"]
            equity = pnl_series.cumsum()
            running_max = equity.cummax()
            drawdown = equity - running_max

            ax.fill_between(range(len(drawdown)), drawdown.values, 0, color="red", alpha=0.3, label="Drawdown")
            ax.plot(range(len(drawdown)), drawdown.values, color="darkred", linewidth=1.5)
            ax.set_ylabel("Drawdown (USD)")
            ax.set_title(f"Drawdown Time Series - {pair}")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.xlabel("Trade Sequence")
        plt.tight_layout()

        output_path = self.output_dir / f"drawdown_timeseries_{report_prefix}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    def generate_summary_table(
        self,
        metrics: pd.DataFrame,
        kpis_overall: dict[str, float],
        report_prefix: str,
    ) -> None:
        """Generate and save a summary table of results."""
        summary_lines = [
            f"# Walk-Forward Backtest Report: {report_prefix.upper()}",
            "",
            "## Overall KPIs",
            "",
        ]

        for metric, value in kpis_overall.items():
            if isinstance(value, float):
                summary_lines.append(f"- **{metric}**: {value:.6f}")
            else:
                summary_lines.append(f"- **{metric}**: {value}")

        summary_lines.extend(
            [
                "",
                "## Per-Pair Performance Ranking",
                "",
                "| Pair | Total Signals | Executed | Precision | Hit Ratio | Spread Capture | Total PnL | Max DD |",
                "|------|---------------|----------|-----------|-----------|----------------|-----------|--------|",
            ]
        )

        for _, row in metrics.iterrows():
            summary_lines.append(
                f"| {row['pair']} | {row['total_signals']} | {row['executed_signals']} | "
                f"{row['precision']:.2%} | {row['hit_ratio']:.2%} | {row['avg_spread_capture']:.4f} | "
                f"${row['total_pnl']:,.0f} | ${row['max_drawdown']:,.0f} |"
            )

        output_path = self.output_dir / f"backtest_summary_{report_prefix}.md"
        with open(output_path, "w") as f:
            f.write("\n".join(summary_lines))
        print(f"Saved: {output_path}")

    def generate_full_report(
        self,
        predictions_csv: str | Path,
        paper_trades_csv: str | Path,
        alerts_csv: str | Path | None = None,
        kpis_overall: dict[str, float] | None = None,
        report_prefix: str = "backtest",
    ) -> None:
        """Generate complete backtest report with all visualizations and tables."""
        print(f"Loading backtest results from {predictions_csv}")
        results = self.load_backtest_results(predictions_csv, paper_trades_csv, alerts_csv)

        print("Computing per-pair metrics...")
        metrics = self.compute_per_pair_metrics(results["paper_trades"])

        if kpis_overall is None:
            kpis_overall = {}

        print("Generating visualizations...")
        self.plot_per_pair_performance(metrics, report_prefix)
        self.plot_spread_capture_distribution(results["paper_trades"], report_prefix)
        self.plot_drawdown_time_series(results["paper_trades"], report_prefix)

        print("Generating summary tables...")
        self.generate_summary_table(metrics, kpis_overall, report_prefix)

        print(f"\n✓ Report generation complete! All artifacts saved to {self.output_dir}")


def main():
    """Example usage of the report generator."""
    import sys
    from config import ARTIFACTS_DIR

    if len(sys.argv) < 3:
        print("Usage: python report_generator.py <predictions_csv> <paper_trades_csv> [alerts_csv]")
        sys.exit(1)

    predictions_path = sys.argv[1]
    paper_trades_path = sys.argv[2]
    alerts_path = sys.argv[3] if len(sys.argv) > 3 else None

    generator = BacktestReportGenerator(output_dir=ARTIFACTS_DIR)
    generator.generate_full_report(
        predictions_csv=predictions_path,
        paper_trades_csv=paper_trades_path,
        alerts_csv=alerts_path,
        report_prefix="walk_forward_1d",
    )


if __name__ == "__main__":
    main()
