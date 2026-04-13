"""Production-grade Streamlit dashboard for ETF tracking error and arbitrage monitoring.

This dashboard is designed for risk managers and ETF trading desks. It provides:
1) Live tracking error monitoring across ETF-benchmark pairs.
2) ETF-level drilldown with explainability and anomaly diagnostics.
3) Arbitrage signal visibility with desk-ready sizing and PnL estimates.
4) Configurable alerting (email and Slack) with a persisted in-session alert log.
5) Historical analytics and model backtesting summaries.
"""

from __future__ import annotations

import json
import smtplib
import ssl
from datetime import timedelta
from email.message import EmailMessage
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (
    ARBITRAGE_WINDOW,
    DEFAULT_HORIZON,
    DEFAULT_INTERVAL,
    DEFAULT_INTRADAY_INTERVAL,
    DEFAULT_INTRADAY_PERIOD,
    DEFAULT_PERIOD,
    DEFAULT_WINDOW,
    ETF_SECTOR_MAP,
    MODEL_ARTIFACT_PATH,
    PAIR_CONFIGS,
)
from src.arbitrage_signal import ArbitrageSignalGenerator
from src.anomaly_detector import ResidualAnomalyDetector
from src.data_loader import MarketDataLoader
from src.explainability import TrackingErrorExplainer
from src.features import FeatureEngineer
from src.models import TrackingErrorModel
from src.portfolio_risk import PortfolioRiskAggregator
from src.real_time_predictor import RealTimeTrackingErrorPredictor
from src.regime_detector import AdaptiveThresholdConfig, RegimeDetector
from src.utils import time_split


st.set_page_config(
    page_title="ETF Tracking Error & Arbitrage Desk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Visual thresholds used consistently for table highlights and risk badges.
HIGH_DEVIATION_PCT = 0.80
MODERATE_DEVIATION_PCT = 0.40


def apply_global_styles() -> None:
    """Inject desk-style CSS to provide a clean and modern visual layout.

    Font strategy
    -------------
    The <link> tags below use the "media=print / onload" non-blocking pattern so the
    browser never stalls page paint waiting for the CDN.  If the Google Fonts CDN is
    unreachable (air-gapped networks, strict egress firewalls common in banks) the
    font-family stacks fall through immediately to high-quality system fonts — no
    layout shift, no blank text, no JS error.
    """
    # Non-blocking font loader: starts as media="print" (loads in background after
    # first paint), then onload flips it to media="all" so the font is adopted.
    # The <noscript> fallback serves environments with JS disabled.
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com" crossorigin>
        <link
            rel="stylesheet"
            href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap"
            media="print"
            onload="this.media='all'"
        >
        <noscript>
          <link
            rel="stylesheet"
            href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap"
          >
        </noscript>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
            /*
             * Font stack: Manrope is the preferred choice, loaded asynchronously above.
             * The stack falls through to Inter, then OS-native UI fonts so the page is
             * immediately readable even when the CDN is completely unreachable.
             */
            html, body, [class*="css"] {
                font-family: 'Manrope', 'Inter', -apple-system, BlinkMacSystemFont,
                             'Segoe UI', Helvetica, Arial, sans-serif;
            }

            :root {
                --panel-bg: #f7f9fc;
                --card-bg: #ffffff;
                --accent: #103a6f;
                --risk-high: #cf2e2e;
                --risk-mid: #d97a1f;
                --risk-low: #1d8a4c;
            }

            .stApp {
                background:
                    radial-gradient(circle at 15% 20%, rgba(16,58,111,0.08), transparent 32%),
                    radial-gradient(circle at 85% 12%, rgba(217,122,31,0.08), transparent 30%),
                    linear-gradient(180deg, #f5f7fb 0%, #eef3f9 100%);
            }

            .desk-panel {
                background: var(--card-bg);
                border: 1px solid rgba(16,58,111,0.08);
                border-radius: 14px;
                padding: 14px 16px;
                box-shadow: 0 6px 20px rgba(15, 35, 75, 0.06);
            }

            .desk-title {
                font-size: 1.7rem;
                font-weight: 800;
                color: #0f2646;
                letter-spacing: 0.2px;
            }

            .desk-subtitle {
                color: #4d5f7a;
                font-size: 0.95rem;
            }

            /*
             * Monospace stack: IBM Plex Mono preferred; Cascadia Code and Consolas
             * are pre-installed on most Windows bank workstations; Courier New is
             * the universal last-resort fallback.
             */
            .mono {
                font-family: 'IBM Plex Mono', 'Cascadia Code', 'Consolas',
                             'Courier New', monospace;
            }

            div[data-testid="stMetricValue"] {
                color: #102e54;
                font-weight: 800;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    """Initialize all dashboard session keys exactly once."""
    if "selected_pair" not in st.session_state:
        st.session_state.selected_pair = None

    if "alert_log" not in st.session_state:
        # List of dictionaries with timestamp, ETF, channel, and reason.
        st.session_state.alert_log = []

    if "pair_alert_enabled" not in st.session_state:
        st.session_state.pair_alert_enabled = {pair: True for pair in PAIR_CONFIGS}

    if "pair_breach_start" not in st.session_state:
        st.session_state.pair_breach_start = {}

    if "alert_channel_backoff" not in st.session_state:
        # Per-channel exponential back-off state, keyed by "{pair}::{channel}".
        # Schema: {"fail_count": int, "next_retry_ts": pd.Timestamp | None}
        st.session_state.alert_channel_backoff = {}


def _get_interval_minutes(interval: str) -> int:
    """Convert yfinance interval strings to minutes for UI labels."""
    mapping = {
        "1m": 1,
        "2m": 2,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "60m": 60,
        "90m": 90,
        "1h": 60,
        "1d": 390,
    }
    return mapping.get(interval, 5)


@st.cache_data(ttl=300)
def load_market_panel(period: str, interval: str) -> pd.DataFrame:
    """Fetch ETF-benchmark universe panel from Yahoo Finance."""
    loader = MarketDataLoader()
    return loader.fetch_universe(PAIR_CONFIGS, period=period, interval=interval)


@st.cache_data(ttl=300)
def build_feature_panel(data: pd.DataFrame, horizon: int, rolling_window: int) -> pd.DataFrame:
    """Create model-ready feature panel with caching for low-latency UX."""
    engineer = FeatureEngineer(rolling_window=rolling_window, horizon=horizon)
    return engineer.transform_universe(data)


def load_or_train_model(feature_panel: pd.DataFrame, artifact_path: Path) -> TrackingErrorModel:
    """Load model artifact, or train once if artifact does not exist."""
    if artifact_path.exists():
        return TrackingErrorModel.load(artifact_path)

    model = TrackingErrorModel(random_state=42)
    model.train(feature_panel, target_col="target_te", test_size=0.2)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(artifact_path)
    return model


def evaluate_backtest_metrics(
    feature_panel: pd.DataFrame,
    model: TrackingErrorModel,
    test_size: float = 0.2,
) -> dict[str, float]:
    """Compute holdout metrics from the latest transformed panel."""
    target_col = "target_te"
    if target_col not in feature_panel.columns:
        return {}

    model_df = feature_panel.dropna(subset=[target_col]).copy()
    feature_columns = [
        col for col in model_df.columns if col not in {target_col, "etf_ticker", "benchmark_ticker", "signal"}
    ]
    model_df = model_df.dropna(subset=feature_columns)

    if len(model_df) < 100:
        return {}

    _, test_df = time_split(model_df, test_size=test_size)
    if test_df.empty:
        return {}

    x_test = test_df[model.numeric_columns + model.categorical_columns]
    y_test = test_df[target_col].to_numpy()
    y_pred = model.predict(x_test)

    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    mape = float(np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-8))))
    r2 = float(1 - np.sum((y_test - y_pred) ** 2) / (np.sum((y_test - np.mean(y_test)) ** 2) + 1e-12))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "scored_rows": float(len(test_df)),
    }


def _build_shap_waterfall_figure(shap_df: pd.DataFrame, top_features: int = 10) -> go.Figure:
    """Build compact SHAP waterfall chart for desk users."""
    ranked = shap_df.head(top_features).copy()
    ranked = ranked.sort_values("shap_value", ascending=False)

    fig = go.Figure(
        go.Waterfall(
            name="SHAP Contribution",
            orientation="v",
            measure=["relative"] * len(ranked),
            x=ranked["feature"].tolist(),
            y=ranked["shap_value"].tolist(),
            connector={"line": {"color": "rgba(50, 50, 50, 0.35)"}},
        )
    )

    fig.update_layout(
        title="Latest Prediction - SHAP Waterfall",
        xaxis_title="Feature",
        yaxis_title="Contribution to Predicted Tracking Error",
        template="plotly_white",
        height=420,
        margin={"l": 30, "r": 20, "t": 50, "b": 10},
    )
    return fig


def _build_residual_panel(
    prediction_series: pd.DataFrame,
    feature_panel: pd.DataFrame,
    pair_name: str,
) -> pd.DataFrame:
    """Join predictions with realized/target TE to compute residual history."""
    pair_predictions = prediction_series[prediction_series["pair"] == pair_name].copy()
    if pair_predictions.empty:
        return pd.DataFrame()

    pair_actuals = feature_panel[feature_panel["pair"] == pair_name].copy()
    if pair_actuals.empty:
        return pd.DataFrame()

    pair_actuals = pair_actuals.reset_index().rename(columns={"index": "timestamp"})
    if "timestamp" not in pair_actuals.columns:
        # Defensive fallback for unexpected index naming.
        pair_actuals = pair_actuals.rename(columns={pair_actuals.columns[0]: "timestamp"})

    target_te = (
        pair_actuals["target_te"]
        if "target_te" in pair_actuals.columns
        else pd.Series(np.nan, index=pair_actuals.index)
    )
    realized_te = (
        pair_actuals["realized_te"]
        if "realized_te" in pair_actuals.columns
        else pd.Series(np.nan, index=pair_actuals.index)
    )
    pair_actuals["actual_tracking_error"] = target_te.where(target_te.notna(), realized_te)

    merge_columns = ["timestamp", "actual_tracking_error"]
    for optional_column in ["etf_ticker", "benchmark_ticker", "realized_te"]:
        if optional_column in pair_actuals.columns:
            merge_columns.append(optional_column)

    merged = pair_predictions.merge(
        pair_actuals[merge_columns],
        on="timestamp",
        how="left",
    )
    merged = merged.dropna(subset=["predicted_tracking_error", "actual_tracking_error"]).copy()
    if merged.empty:
        return merged

    merged["residual"] = merged["actual_tracking_error"] - merged["predicted_tracking_error"]
    return merged.sort_values("timestamp").reset_index(drop=True)


def compute_latest_anomaly_map(
    prediction_series: pd.DataFrame,
    feature_panel: pd.DataFrame,
    rolling_window: int,
    regime_latest_map: dict[str, dict[str, object]] | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, pd.DataFrame]]:
    """Compute latest anomaly status and full anomaly history per pair."""
    latest_by_pair: dict[str, dict[str, object]] = {}
    history_by_pair: dict[str, pd.DataFrame] = {}

    detector_short_window = max(10, int(rolling_window // 2))
    detector_long_window = max(24, int(rolling_window * 2))
    required_observations = max(80, detector_long_window * 2)

    for pair_name in prediction_series["pair"].unique().tolist():
        residual_panel = _build_residual_panel(
            prediction_series=prediction_series,
            feature_panel=feature_panel,
            pair_name=pair_name,
        )

        if len(residual_panel) < required_observations:
            latest_by_pair[pair_name] = {
                "anomaly_detected": False,
                "confidence": 0.0,
                "anomaly_type": "insufficient_history",
                "score": 0.0,
                "explanation": (
                    "Insufficient aligned residual history for robust anomaly detection "
                    f"(required {required_observations}, available {len(residual_panel)})."
                ),
                "recommended_action": "Monitor",
            }
            history_by_pair[pair_name] = pd.DataFrame()
            continue

        adaptive_anomaly_threshold = float(
            (regime_latest_map or {}).get(pair_name, {}).get("adaptive_thresholds", {}).get(
                "anomaly_reconstruction_threshold",
                0.05,
            )
        )
        adaptive_anomaly_threshold = max(adaptive_anomaly_threshold, 1e-6)

        # Lower anomaly threshold implies higher sensitivity.
        sensitivity_ratio = 0.05 / adaptive_anomaly_threshold
        adaptive_contamination = float(np.clip(0.05 * sensitivity_ratio, 0.02, 0.20))

        detector = ResidualAnomalyDetector(
            short_window=detector_short_window,
            long_window=detector_long_window,
            contamination=adaptive_contamination,
            normal_quantile=0.85,
            random_state=42,
        )

        scored = detector.fit_score(
            actual_tracking_error=residual_panel["actual_tracking_error"],
            predicted_tracking_error=residual_panel["predicted_tracking_error"],
        )
        latest = detector.latest_result(
            actual_tracking_error=residual_panel["actual_tracking_error"],
            predicted_tracking_error=residual_panel["predicted_tracking_error"],
        )

        latest_by_pair[pair_name] = latest
        history_by_pair[pair_name] = scored

    return latest_by_pair, history_by_pair


def compute_latest_regime_map(
    prediction_series: pd.DataFrame,
    feature_panel: pd.DataFrame,
    rolling_window: int,
    base_alert_tracking_error_pct: float,
    base_arbitrage_confidence_min: float,
    base_anomaly_threshold: float,
) -> tuple[dict[str, dict[str, object]], dict[str, pd.DataFrame]]:
    """Compute latest regime diagnostics and history for each ETF pair."""
    latest_by_pair: dict[str, dict[str, object]] = {}
    history_by_pair: dict[str, pd.DataFrame] = {}

    detector = RegimeDetector(
        rolling_window=max(12, int(rolling_window)),
        smoothing_alpha=0.35,
        min_regime_persistence=4,
        random_state=42,
        base_thresholds=AdaptiveThresholdConfig(
            alert_tracking_error=float(base_alert_tracking_error_pct),
            arbitrage_confidence_min=float(base_arbitrage_confidence_min),
            anomaly_reconstruction_threshold=float(base_anomaly_threshold),
        ),
    )

    for pair_name in prediction_series["pair"].unique().tolist():
        residual_panel = _build_residual_panel(
            prediction_series=prediction_series,
            feature_panel=feature_panel,
            pair_name=pair_name,
        )

        if residual_panel.empty:
            latest_by_pair[pair_name] = {
                "current_regime": "Calm",
                "confidence": 1.0,
                "regime_probability": {"Calm": 1.0, "Stress": 0.0, "High_Vol": 0.0},
                "adaptive_thresholds": {
                    "alert_tracking_error": float(base_alert_tracking_error_pct),
                    "arbitrage_confidence_min": float(base_arbitrage_confidence_min),
                    "anomaly_reconstruction_threshold": float(base_anomaly_threshold),
                },
                "explanation": "No aligned residual panel available for regime estimation.",
                "model_used": "empty_panel_default",
                "regime_history": [],
            }
            history_by_pair[pair_name] = pd.DataFrame()
            continue

        regime_result = detector.detect_regime(residual_panel["residual"], history_points=120)
        latest_by_pair[pair_name] = regime_result

        history_rows = regime_result.get("regime_history", [])
        if history_rows:
            history_frame = pd.DataFrame(history_rows)
            history_frame["timestamp"] = pd.to_datetime(history_frame["timestamp"])
            history_by_pair[pair_name] = history_frame.sort_values("timestamp").reset_index(drop=True)
        else:
            history_by_pair[pair_name] = pd.DataFrame(columns=["timestamp", "regime", "confidence"])

    return latest_by_pair, history_by_pair


def regime_badge_html(regime: str) -> str:
    """Return a compact color-coded regime badge."""
    color_map = {
        "Calm": ("#1d8a4c", "#e8f6ee"),
        "Stress": ("#d97a1f", "#fff4e8"),
        "High_Vol": ("#cf2e2e", "#fdeaea"),
    }
    fg, bg = color_map.get(regime, ("#103a6f", "#e8eef8"))
    return (
        "<span style='display:inline-block;padding:6px 10px;border-radius:10px;"
        f"font-weight:700;color:{fg};background:{bg};border:1px solid {fg}33;'>"
        f"{regime}"
        "</span>"
    )


def build_regime_history_chart(regime_history: pd.DataFrame, pair_name: str) -> go.Figure:
    """Build a compact regime history chart for a selected pair."""
    regime_to_level = {"Calm": 0, "Stress": 1, "High_Vol": 2}
    level_to_label = {0: "Calm", 1: "Stress", 2: "High Vol"}

    fig = go.Figure()
    if not regime_history.empty:
        plot_df = regime_history.copy()
        plot_df["regime_level"] = plot_df["regime"].map(regime_to_level).fillna(0)

        fig.add_trace(
            go.Scatter(
                x=plot_df["timestamp"],
                y=plot_df["regime_level"],
                mode="lines+markers",
                line={"color": "#103a6f", "width": 2},
                marker={
                    "size": 7,
                    "color": plot_df["confidence"],
                    "colorscale": "YlOrRd",
                    "showscale": True,
                    "colorbar": {"title": "Conf"},
                },
                customdata=np.stack([plot_df["regime"], plot_df["confidence"]], axis=1),
                hovertemplate=(
                    "Timestamp: %{x}<br>Regime: %{customdata[0]}"
                    "<br>Confidence: %{customdata[1]:.2f}<extra></extra>"
                ),
                name=pair_name,
            )
        )

    fig.update_layout(
        title=f"Regime History - {pair_name}",
        xaxis_title="Timestamp",
        yaxis_title="Regime",
        yaxis={
            "tickmode": "array",
            "tickvals": [0, 1, 2],
            "ticktext": [level_to_label[0], level_to_label[1], level_to_label[2]],
        },
        template="plotly_white",
        height=260,
        margin={"l": 20, "r": 20, "t": 40, "b": 10},
    )
    return fig


def build_latest_actual_snapshot(feature_panel: pd.DataFrame) -> pd.DataFrame:
    """Create one-row-per-pair snapshot with latest realized and target TE."""
    rows = []
    for pair_name, pair_df in feature_panel.groupby("pair", observed=True):
        latest = pair_df.sort_index().tail(1).copy()
        if latest.empty:
            continue

        actual_te = latest["target_te"].iloc[0]
        if pd.isna(actual_te):
            actual_te = latest["realized_te"].iloc[0]

        rows.append(
            {
                "pair": pair_name,
                "timestamp": latest.index[-1],
                "etf_ticker": str(latest["etf_ticker"].iloc[0]),
                "benchmark_ticker": str(latest["benchmark_ticker"].iloc[0]),
                "actual_tracking_error": float(actual_te) if pd.notna(actual_te) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def classify_risk_bucket(predicted_error_pct: float) -> str:
    """Convert predicted error percentage into qualitative risk bucket."""
    abs_val = abs(predicted_error_pct)
    if abs_val >= HIGH_DEVIATION_PCT:
        return "High"
    if abs_val >= MODERATE_DEVIATION_PCT:
        return "Moderate"
    return "Normal"


def build_live_overview_table(
    latest_predictions: pd.DataFrame,
    latest_actuals: pd.DataFrame,
    anomaly_latest_map: dict[str, dict[str, object]],
    signal_table: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the master live table consumed by overview and alert modules."""
    merged = latest_predictions.merge(
        latest_actuals[["pair", "etf_ticker", "benchmark_ticker", "actual_tracking_error"]],
        on="pair",
        how="left",
    )

    if signal_table.empty:
        signal_cols = pd.DataFrame(
            columns=[
                "pair",
                "action",
                "confidence",
                "recommended_shares",
                "estimated_profit",
                "regime",
                "regime_confidence",
                "applied_alert_tracking_error_pct",
                "applied_confidence_threshold",
            ]
        )
    else:
        signal_cols = signal_table[
            [
                "pair",
                "action",
                "confidence",
                "recommended_shares",
                "estimated_profit",
                "regime",
                "regime_confidence",
                "applied_alert_tracking_error_pct",
                "applied_confidence_threshold",
            ]
        ].copy()

    merged = merged.merge(signal_cols, on="pair", how="left")

    merged["predicted_error_pct"] = merged["predicted_tracking_error"] * 100.0
    merged["actual_error_pct"] = merged["actual_tracking_error"] * 100.0

    merged["anomaly_flag"] = merged["pair"].apply(
        lambda pair: bool(anomaly_latest_map.get(pair, {}).get("anomaly_detected", False))
    )
    merged["anomaly_type"] = merged["pair"].apply(
        lambda pair: str(anomaly_latest_map.get(pair, {}).get("anomaly_type", "none"))
    )
    merged["risk_bucket"] = merged["predicted_error_pct"].apply(classify_risk_bucket)

    # Use HOLD as default when no generated signal exists for a pair.
    merged["action"] = merged["action"].fillna("HOLD")
    merged["confidence"] = merged["confidence"].fillna(0.0)
    merged["recommended_shares"] = merged["recommended_shares"].fillna(0).astype(int)
    merged["estimated_profit"] = merged["estimated_profit"].fillna(0.0)
    merged["regime"] = merged["regime"].fillna("Calm")
    merged["regime_confidence"] = merged["regime_confidence"].fillna(1.0)
    merged["applied_alert_tracking_error_pct"] = merged["applied_alert_tracking_error_pct"].fillna(0.80)
    merged["applied_confidence_threshold"] = merged["applied_confidence_threshold"].fillna(0.70)

    merged = merged.sort_values("predicted_error_pct", ascending=False).reset_index(drop=True)
    return merged


def style_live_table(table: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Apply risk-based color encoding for desk readability."""

    def row_style(row: pd.Series) -> list[str]:
        color = ""
        if row["risk_bucket"] == "High":
            color = "background-color: rgba(207, 46, 46, 0.16);"
        elif row["risk_bucket"] == "Moderate":
            color = "background-color: rgba(217, 122, 31, 0.16);"
        else:
            color = "background-color: rgba(29, 138, 76, 0.12);"
        return [color] * len(row)

    formatters = {
        "predicted_error_pct": "{:.3f}%",
        "actual_error_pct": "{:.3f}%",
        "confidence": "{:.2f}",
        "estimated_profit": "${:,.0f}",
    }
    valid_formatters = {key: value for key, value in formatters.items() if key in table.columns}

    return table.style.apply(row_style, axis=1).format(valid_formatters)


def build_deviation_heatmap(table: pd.DataFrame) -> go.Figure:
    """Create heatmap that ranks ETFs by current predicted tracking error."""
    ranked = table[["pair", "predicted_error_pct"]].copy()
    ranked = ranked.sort_values("predicted_error_pct", ascending=False)

    fig = go.Figure(
        data=go.Heatmap(
            z=ranked[["predicted_error_pct"]].to_numpy(),
            x=["Predicted TE (%)"],
            y=ranked["pair"].tolist(),
            colorscale=[
                [0.0, "#1d8a4c"],
                [0.5, "#d97a1f"],
                [1.0, "#cf2e2e"],
            ],
            zmin=min(-HIGH_DEVIATION_PCT, float(ranked["predicted_error_pct"].min())),
            zmax=max(HIGH_DEVIATION_PCT, float(ranked["predicted_error_pct"].max())),
            colorbar={"title": "TE (%)"},
            hovertemplate="ETF Pair: %{y}<br>Predicted TE: %{z:.3f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Live ETF Tracking Error Heatmap (Highest Deviations First)",
        template="plotly_white",
        height=max(300, 70 * len(ranked)),
        margin={"l": 20, "r": 20, "t": 50, "b": 15},
    )
    return fig


def _default_portfolio_weight_text(etf_tickers: list[str]) -> str:
    """Build default equal-weight input text for the sidebar."""
    if not etf_tickers:
        return ""
    equal_weight = 1.0 / len(etf_tickers)
    return ", ".join(f"{ticker}:{equal_weight:.4f}" for ticker in etf_tickers)


def parse_portfolio_weights(raw_weights: str, universe_etfs: list[str]) -> tuple[dict[str, float], list[str]]:
    """Parse weights from 'ETF:weight' comma-separated text and validate entries."""
    parsed: dict[str, float] = {}
    errors: list[str] = []

    if not raw_weights.strip():
        return parsed, ["Portfolio weights text is empty."]

    universe = {ticker.upper() for ticker in universe_etfs}
    tokens = [token.strip() for token in raw_weights.split(",") if token.strip()]

    for token in tokens:
        if ":" not in token:
            errors.append(f"Invalid token '{token}', expected ETF:weight format.")
            continue

        etf, value = token.split(":", 1)
        etf_clean = etf.strip().upper()
        if etf_clean not in universe:
            errors.append(f"Unknown ETF '{etf_clean}' (not present in current universe).")
            continue

        try:
            numeric_weight = float(value.strip())
        except ValueError:
            errors.append(f"Invalid numeric weight for '{etf_clean}'.")
            continue

        if numeric_weight <= 0:
            errors.append(f"Weight for '{etf_clean}' must be strictly positive.")
            continue

        parsed[etf_clean] = numeric_weight

    if not parsed:
        errors.append("No valid positive weights were parsed.")
    return parsed, errors


def build_portfolio_risk_timeseries_figure(portfolio_series: pd.DataFrame) -> go.Figure:
    """Create time-series chart for aggregate portfolio predicted TE."""
    fig = go.Figure()

    if not portfolio_series.empty:
        fig.add_trace(
            go.Scatter(
                x=portfolio_series["timestamp"],
                y=portfolio_series["portfolio_predicted_te"] * 100.0,
                mode="lines",
                line={"color": "#103a6f", "width": 2},
                name="Portfolio Predicted TE",
            )
        )

    fig.update_layout(
        title="Portfolio Predicted Tracking Error Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Portfolio Predicted TE (%)",
        template="plotly_white",
        height=360,
        margin={"l": 20, "r": 20, "t": 45, "b": 10},
    )
    return fig


def build_sector_exposure_figure(sector_exposure: pd.DataFrame) -> go.Figure:
    """Create sector exposure bar chart from ETF weight map."""
    fig = go.Figure()

    if not sector_exposure.empty:
        fig.add_trace(
            go.Bar(
                x=sector_exposure["sector"],
                y=sector_exposure["exposure_pct"],
                marker={"color": "#d97a1f"},
                name="Sector Exposure",
            )
        )

    fig.update_layout(
        title="Portfolio Sector Exposure",
        xaxis_title="Sector",
        yaxis_title="Exposure (%)",
        template="plotly_white",
        height=360,
        margin={"l": 20, "r": 20, "t": 45, "b": 10},
    )
    return fig


def build_history_chart(
    pair_name: str,
    intraday_residual_panel: pd.DataFrame,
    daily_feature_panel: pd.DataFrame,
) -> go.Figure:
    """Build combined 30-day daily + intraday chart for selected ETF pair."""
    fig = go.Figure()

    pair_daily = daily_feature_panel[daily_feature_panel["pair"] == pair_name].copy().sort_index()
    pair_daily = pair_daily.tail(30)

    if not pair_daily.empty:
        fig.add_trace(
            go.Scatter(
                x=pair_daily.index,
                y=pair_daily["realized_te"] * 100.0,
                mode="lines+markers",
                name="Realized TE (Daily, last 30d)",
                line={"color": "#103a6f", "width": 2},
                marker={"size": 4},
            )
        )

    if not intraday_residual_panel.empty:
        fig.add_trace(
            go.Scatter(
                x=intraday_residual_panel["timestamp"],
                y=intraday_residual_panel["predicted_tracking_error"] * 100.0,
                mode="lines",
                name="Predicted TE (Intraday)",
                line={"color": "#d97a1f", "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=intraday_residual_panel["timestamp"],
                y=intraday_residual_panel["actual_tracking_error"] * 100.0,
                mode="lines",
                name="Actual TE (Intraday proxy)",
                line={"color": "#1d8a4c", "width": 2, "dash": "dot"},
            )
        )

    fig.update_layout(
        title=f"{pair_name}: Tracking Error (30-Day + Intraday)",
        xaxis_title="Timestamp",
        yaxis_title="Tracking Error (%)",
        template="plotly_white",
        height=460,
        margin={"l": 20, "r": 20, "t": 50, "b": 10},
    )
    return fig


def send_email_alert(config: dict[str, object], subject: str, body: str) -> tuple[bool, str]:
    """Send one email alert through SMTP; returns success status and message."""
    if not config.get("email_enabled"):
        return False, "Email channel disabled"

    required = ["smtp_host", "smtp_port", "username", "password", "from_email", "to_email"]
    if any(not config.get(field) for field in required):
        return False, "Missing SMTP/email configuration fields"

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = str(config["from_email"])
    message["To"] = str(config["to_email"])
    message.set_content(body)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(str(config["smtp_host"]), int(config["smtp_port"])) as smtp:
            smtp.starttls(context=context)
            smtp.login(str(config["username"]), str(config["password"]))
            smtp.send_message(message)
        return True, "Email sent"
    except Exception as exc:  # pragma: no cover - network dependent
        return False, f"Email send failed: {exc}"


def send_slack_alert(config: dict[str, object], body: str) -> tuple[bool, str]:
    """Send one Slack alert via incoming webhook URL."""
    if not config.get("slack_enabled"):
        return False, "Slack channel disabled"

    webhook_url = str(config.get("slack_webhook_url", "")).strip()
    if not webhook_url:
        return False, "Missing Slack webhook URL"

    payload = json.dumps({"text": body}).encode("utf-8")
    request = Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=8) as response:  # noqa: S310
            status_ok = 200 <= response.status < 300
        return status_ok, "Slack sent" if status_ok else f"Slack returned status {response.status}"
    except URLError as exc:  # pragma: no cover - network dependent
        return False, f"Slack send failed: {exc}"


# ── Alert rate-limiting / exponential back-off ───────────────────────────────
# Email and Slack channels are tracked independently and per ETF pair.
# After the first delivery failure the channel is silenced for BASE_BACKOFF_SECONDS.
# Each subsequent failure doubles the wait time, capped at MAX_BACKOFF_SECONDS.
# A successful delivery resets the counter immediately.
_ALERT_BASE_BACKOFF_SECONDS: int = 60      # 1 minute on first failure
_ALERT_MAX_BACKOFF_SECONDS: int = 1_800   # cap: 30 minutes


def _alert_channel_key(pair: str, channel: str) -> str:
    """Compose a session-state key scoped to one ETF pair and delivery channel."""
    return f"{pair}::{channel}"


def _channel_in_backoff(channel_key: str, now_ts: pd.Timestamp) -> bool:
    """Return True when the channel is inside its current back-off window."""
    state = st.session_state.alert_channel_backoff.get(channel_key)
    if state is None:
        return False
    next_retry: pd.Timestamp | None = state.get("next_retry_ts")
    return next_retry is not None and now_ts < next_retry


def _record_channel_failure(channel_key: str, now_ts: pd.Timestamp) -> None:
    """Increment failure counter and schedule next retry with exponential back-off."""
    state = st.session_state.alert_channel_backoff.setdefault(
        channel_key, {"fail_count": 0, "next_retry_ts": None}
    )
    state["fail_count"] += 1
    backoff_seconds = min(
        _ALERT_BASE_BACKOFF_SECONDS * (2 ** (state["fail_count"] - 1)),
        _ALERT_MAX_BACKOFF_SECONDS,
    )
    state["next_retry_ts"] = now_ts + timedelta(seconds=backoff_seconds)


def _record_channel_success(channel_key: str) -> None:
    """Reset back-off after a successful send so future alerts fire immediately."""
    st.session_state.alert_channel_backoff.pop(channel_key, None)


def process_threshold_alerts(
    live_table: pd.DataFrame,
    alert_config: dict[str, object],
    now_ts: pd.Timestamp,
) -> None:
    """Evaluate threshold/persistence alert conditions and dispatch notifications."""
    threshold_pct = float(alert_config["threshold_pct"])
    persistence_minutes = int(alert_config["persistence_minutes"])

    for _, row in live_table.iterrows():
        etf = str(row["etf_ticker"])
        pair = str(row["pair"])
        predicted_pct = float(row["predicted_error_pct"])
        pair_threshold_pct = float(row.get("applied_alert_tracking_error_pct", threshold_pct))

        if not st.session_state.pair_alert_enabled.get(etf, True):
            # Reset the breach timer if alerting is disabled for the ETF.
            st.session_state.pair_breach_start.pop(pair, None)
            continue

        above_threshold = abs(predicted_pct) >= pair_threshold_pct
        breach_start = st.session_state.pair_breach_start.get(pair)

        if above_threshold and breach_start is None:
            st.session_state.pair_breach_start[pair] = now_ts
            continue

        if not above_threshold:
            st.session_state.pair_breach_start.pop(pair, None)
            continue

        elapsed = now_ts - breach_start if breach_start is not None else timedelta(minutes=0)
        if elapsed < timedelta(minutes=persistence_minutes):
            continue

        reason = (
            f"Threshold breach: {pair} predicted tracking error {predicted_pct:.3f}% "
            f"exceeded {pair_threshold_pct:.3f}% for >= {persistence_minutes} minutes"
        )
        subject = f"ETF Desk Alert | {pair} tracking error breach"
        body = (
            f"{reason}\n"
            f"Action: {row['action']}\n"
            f"Signal Confidence: {float(row['confidence']):.2f}\n"
            f"Anomaly Flag: {bool(row['anomaly_flag'])}\n"
            f"Timestamp: {now_ts}"
        )

        email_key = _alert_channel_key(pair, "email")
        slack_key = _alert_channel_key(pair, "slack")

        # --- Email delivery with back-off gate ----------------------------------
        if _channel_in_backoff(email_key, now_ts):
            # Channel is in a back-off window; skip without logging a failure.
            email_ok, email_msg = False, "email rate-limited (back-off active)"
        else:
            email_ok, email_msg = send_email_alert(alert_config, subject=subject, body=body)
            if email_ok:
                _record_channel_success(email_key)
            elif "disabled" not in email_msg:
                # Back-off only on genuine delivery failures, not intentional disables.
                _record_channel_failure(email_key, now_ts)

        # --- Slack delivery with back-off gate ----------------------------------
        if _channel_in_backoff(slack_key, now_ts):
            slack_ok, slack_msg = False, "slack rate-limited (back-off active)"
        else:
            slack_ok, slack_msg = send_slack_alert(alert_config, body=body)
            if slack_ok:
                _record_channel_success(slack_key)
            elif "disabled" not in slack_msg:
                _record_channel_failure(slack_key, now_ts)

        for channel, ok, status in [
            ("email", email_ok, email_msg),
            ("slack", slack_ok, slack_msg),
        ]:
            if not ok:
                continue
            st.session_state.alert_log.append(
                {
                    "timestamp": now_ts,
                    "pair": pair,
                    "etf": etf,
                    "channel": channel,
                    "reason": reason,
                    "status": status,
                }
            )

        # Advance the breach timer only after a confirmed send so the persistence
        # window is re-evaluated from the actual last-delivered timestamp.
        if email_ok or slack_ok:
            st.session_state.pair_breach_start[pair] = now_ts


def main() -> None:
    """Render the complete dashboard."""
    apply_global_styles()
    initialize_session_state()

    st.markdown("<div class='desk-title'>ETF Tracking Error Prediction & Arbitrage Desk</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='desk-subtitle'>Live surveillance for risk managers and ETF trading desks.</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Configuration")

        st.subheader("Data & Model")
        mode = st.selectbox(
            "Data Mode",
            ["Intraday", "Daily"],
            index=0,
            help="Intraday is intended for desk monitoring every 5-15 minutes.",
        )

        if mode == "Intraday":
            period = st.selectbox("Lookback Period", ["5d", "30d", "60d"], index=2)
            interval = st.selectbox("Bar Interval", ["5m", "15m", "30m", "60m"], index=0)
        else:
            period = st.selectbox("Lookback Period", ["6mo", "1y", "2y", "5y"], index=2)
            interval = st.selectbox("Bar Interval", ["1d"], index=0)

        horizon = st.number_input("Forecast Horizon (bars)", min_value=1, max_value=12, value=DEFAULT_HORIZON)
        rolling_window = st.number_input("Feature Rolling Window", min_value=10, max_value=120, value=DEFAULT_WINDOW)
        confidence_level = st.slider("Prediction Confidence Level", min_value=0.80, max_value=0.99, value=0.95)
        portfolio_var_conf = st.slider(
            "Portfolio VaR Confidence",
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="Confidence level used for historical VaR on portfolio tracking error.",
        )
        portfolio_var_lookback = st.number_input(
            "Portfolio VaR Lookback (bars)",
            min_value=60,
            max_value=1000,
            value=240,
            step=20,
        )

        st.markdown("**Portfolio Weights**")
        st.caption("Format: ETF:weight, ETF:weight (for example: SPY:0.25, BNK.PA:0.20)")
        default_weight_text = _default_portfolio_weight_text(sorted(PAIR_CONFIGS.keys()))
        portfolio_weights_text = st.text_area(
            "ETF Weights",
            value=default_weight_text,
            height=90,
        )

        st.caption(
            f"Defaults: period={DEFAULT_INTRADAY_PERIOD if mode == 'Intraday' else DEFAULT_PERIOD}, "
            f"interval={DEFAULT_INTRADAY_INTERVAL if mode == 'Intraday' else DEFAULT_INTERVAL}"
        )

        st.divider()
        st.subheader("Alerting")
        auto_alerts_enabled = st.toggle("Enable Alert Engine", value=False)
        threshold_pct = st.number_input(
            "Tracking Error Threshold (%)",
            min_value=0.05,
            max_value=5.00,
            value=0.80,
            step=0.05,
            help="Trigger condition uses absolute predicted tracking error percentage.",
        )
        persistence_minutes = st.number_input(
            "Persistence (minutes)",
            min_value=1,
            max_value=240,
            value=15,
            step=1,
        )

        st.markdown("**Email Settings**")
        email_enabled = st.toggle("Enable Email Alerts", value=False)
        smtp_host = st.text_input("SMTP Host", value="")
        smtp_port = st.number_input("SMTP Port", min_value=1, max_value=65535, value=587)
        smtp_user = st.text_input("SMTP Username", value="")
        smtp_password = st.text_input("SMTP Password", type="password", value="")
        from_email = st.text_input("From Email", value="")
        to_email = st.text_input("To Email", value="")

        st.markdown("**Slack Settings**")
        slack_enabled = st.toggle("Enable Slack Alerts", value=False)
        slack_webhook_url = st.text_input("Slack Webhook URL", type="password", value="")

        run_button = st.button("Run / Refresh Dashboard", use_container_width=True)

    if not run_button:
        st.info("Configure settings in the sidebar, then click 'Run / Refresh Dashboard'.")
        return

    with st.spinner("Loading market data, refreshing model outputs, and computing diagnostics..."):
        market_panel = load_market_panel(period=period, interval=interval)
        feature_panel = build_feature_panel(data=market_panel, horizon=int(horizon), rolling_window=int(rolling_window))
        model = load_or_train_model(feature_panel=feature_panel, artifact_path=Path(MODEL_ARTIFACT_PATH))

        realtime_predictor = RealTimeTrackingErrorPredictor(
            model=model,
            rolling_window=int(rolling_window),
            horizon=int(horizon),
        )
        rt_feature_panel = realtime_predictor.build_feature_panel(market_panel)

        latest_predictions = realtime_predictor.predict_latest(
            feature_panel=rt_feature_panel,
            confidence_level=float(confidence_level),
            confidence_window=120,
        )
        prediction_series = realtime_predictor.build_prediction_series(
            feature_panel=rt_feature_panel,
            confidence_level=float(confidence_level),
            confidence_window=120,
        )

        latest_actuals = build_latest_actual_snapshot(rt_feature_panel)

        regime_latest_map, regime_history_map = compute_latest_regime_map(
            prediction_series=prediction_series,
            feature_panel=rt_feature_panel,
            rolling_window=int(rolling_window),
            base_alert_tracking_error_pct=float(threshold_pct),
            base_arbitrage_confidence_min=0.70,
            base_anomaly_threshold=0.05,
        )

        signal_generator = ArbitrageSignalGenerator(
            confidence_threshold=0.70,
            entry_tracking_error=0.0005,
            max_notional=1_500_000.0,
            min_notional=100_000.0,
            transaction_cost_bps=3.0,
            slippage_bps=2.0,
            persistence_window=max(6, ARBITRAGE_WINDOW // 10),
            liquidity_window=12,
        )
        universe_signals = signal_generator.generate_universe_signals(
            intraday_panel=market_panel,
            prediction_snapshot=latest_predictions,
            regime_results=regime_latest_map,
        )

        anomaly_latest_map, anomaly_history_map = compute_latest_anomaly_map(
            prediction_series=prediction_series,
            feature_panel=rt_feature_panel,
            rolling_window=int(rolling_window),
            regime_latest_map=regime_latest_map,
        )

    if latest_predictions.empty:
        st.warning("No valid rows available for real-time scoring. Check data availability and selected mode.")
        return

    live_overview = build_live_overview_table(
        latest_predictions=latest_predictions,
        latest_actuals=latest_actuals,
        anomaly_latest_map=anomaly_latest_map,
        signal_table=universe_signals,
    )

    portfolio_weight_map, portfolio_weight_errors = parse_portfolio_weights(
        raw_weights=portfolio_weights_text,
        universe_etfs=sorted(PAIR_CONFIGS.keys()),
    )
    portfolio_aggregator = PortfolioRiskAggregator(
        var_confidence=float(portfolio_var_conf),
        var_lookback=int(portfolio_var_lookback),
    )
    normalized_weight_map = portfolio_aggregator.normalize_weights(portfolio_weight_map)

    portfolio_series = portfolio_aggregator.build_portfolio_prediction_series(
        prediction_series=prediction_series,
        latest_predictions=latest_predictions,
        normalized_weights=normalized_weight_map,
    )
    portfolio_summary = portfolio_aggregator.summarize(
        live_overview=live_overview,
        portfolio_series=portfolio_series,
        universe_signals=universe_signals,
        normalized_weights=normalized_weight_map,
    )
    portfolio_contributions = portfolio_aggregator.compute_etf_contributions(
        live_overview=live_overview,
        normalized_weights=normalized_weight_map,
    )
    portfolio_sector_exposure = portfolio_aggregator.compute_sector_exposure(
        normalized_weights=normalized_weight_map,
        sector_map=ETF_SECTOR_MAP,
    )

    if st.session_state.selected_pair is None and not live_overview.empty:
        st.session_state.selected_pair = str(live_overview.iloc[0]["pair"])

    # ── Pair selector (persistent across all tabs, rendered before tabs) ───────────
    # Placing the selector ABOVE st.tabs() avoids the Streamlit ordering problem
    # where tab content executes before outside widgets on every rerun.  This
    # guarantees that `selected_pair` is settled before any tab body reads it.
    #
    # st.radio is used instead of a dataframe row-click because:
    # - st.radio works identically on every Streamlit version.
    # - It requires no version-conditional try/except guards.
    # - Its return value is always the user’s latest choice, resolving on the
    #   same rerun that triggered the interaction (no off-by-one lag).
    pair_options: list[str] = live_overview["pair"].tolist()
    current_pair: str = (
        st.session_state.selected_pair
        if st.session_state.selected_pair in pair_options
        else pair_options[0]
    )

    # Build human-readable radio labels: risk badge + tickers + latest predicted TE.
    risk_icon = {"High": "🔴", "Moderate": "🟡", "Normal": "🟢"}
    radio_labels: list[str] = [
        (
            f"{risk_icon.get(str(row['risk_bucket']), '⚪')}  "
            f"{row['etf_ticker']} / {row['benchmark_ticker']}  |"
            f"  TE {float(row['predicted_error_pct']):.3f}%  [{row['action']}]"
        )
        for _, row in live_overview.iterrows()
    ]
    # Reverse-mapping so we resolve the chosen label back to a canonical pair name.
    label_to_pair: dict[str, str] = {
        label: str(row["pair"])
        for label, (_, row) in zip(radio_labels, live_overview.iterrows())
    }
    default_radio_index: int = (
        list(label_to_pair.values()).index(current_pair)
        if current_pair in label_to_pair.values()
        else 0
    )
    # `index=default_radio_index` (not a key) ensures the radio always reflects
    # session_state on every rerun, whichever widget was last interacted with.
    chosen_label: str = st.radio(
        "ETF Pair",
        options=radio_labels,
        index=default_radio_index,
        horizontal=True,
        label_visibility="collapsed",
    )
    selected_pair: str = label_to_pair[chosen_label]
    st.session_state.selected_pair = selected_pair

    selected_regime_result = regime_latest_map.get(
        selected_pair,
        {
            "current_regime": "Calm",
            "confidence": 1.0,
            "adaptive_thresholds": {
                "alert_tracking_error": float(threshold_pct),
                "arbitrage_confidence_min": 0.70,
                "anomaly_reconstruction_threshold": 0.05,
            },
            "explanation": "No regime diagnostics available.",
        },
    )
    selected_adaptive_thresholds = selected_regime_result.get("adaptive_thresholds", {})

    top_tabs = st.tabs(
        [
            "Real-Time Overview",
            "Portfolio Risk",
            "Detailed ETF View",
            "Arbitrage Signal Panel",
            "Alerting System",
            "Historical & Analytics",
        ]
    )

    with top_tabs[0]:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Tracked ETFs", f"{live_overview['etf_ticker'].nunique()}")
        with m2:
            st.metric("High Deviation", f"{int((live_overview['risk_bucket'] == 'High').sum())}")
        with m3:
            st.metric("Anomaly Flags", f"{int(live_overview['anomaly_flag'].sum())}")
        with m4:
            st.metric("Actionable Signals", f"{int((live_overview['action'] != 'HOLD').sum())}")

        st.markdown("**Current Regime (Selected Pair)**")
        regime_col_1, regime_col_2, regime_col_3 = st.columns([0.9, 1.2, 1.6])
        with regime_col_1:
            st.markdown(
                regime_badge_html(str(selected_regime_result.get("current_regime", "Calm"))),
                unsafe_allow_html=True,
            )
        with regime_col_2:
            st.metric("Regime Confidence", f"{float(selected_regime_result.get('confidence', 0.0)):.2f}")
        with regime_col_3:
            st.caption(str(selected_regime_result.get("explanation", "")))

        st.markdown("**Adaptive Thresholds (Selected Pair)**")
        threshold_col_1, threshold_col_2, threshold_col_3 = st.columns(3)
        with threshold_col_1:
            st.metric(
                "Alert TE Threshold",
                f"{float(selected_adaptive_thresholds.get('alert_tracking_error', threshold_pct)):.3f}%",
            )
        with threshold_col_2:
            st.metric(
                "Arbitrage Confidence Min",
                f"{float(selected_adaptive_thresholds.get('arbitrage_confidence_min', 0.70)):.2f}",
            )
        with threshold_col_3:
            st.metric(
                "Anomaly Sensitivity",
                f"{float(selected_adaptive_thresholds.get('anomaly_reconstruction_threshold', 0.05)):.4f}",
            )

        c1, c2 = st.columns([1.0, 1.4])
        with c1:
            st.plotly_chart(build_deviation_heatmap(live_overview), use_container_width=True)

        with c2:
            st.markdown("**Live ETF Universe — use the selector above the tabs to switch pair**")
            display_cols = [
                "etf_ticker",
                "benchmark_ticker",
                "predicted_error_pct",
                "actual_error_pct",
                "risk_bucket",
                "anomaly_flag",
                "action",
                "confidence",
                "recommended_shares",
                "estimated_profit",
                "regime",
                "regime_confidence",
                "pair",
            ]
            # Static color-coded table — selection is handled by the radio above the
            # tabs, so no version-sensitive on_select API is needed here.
            st.dataframe(
                style_live_table(live_overview[display_cols].copy()),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("**Most Deviant ETFs (Top 10)**")
        top10 = live_overview.head(10).copy()
        top10_display = top10[
            [
                "etf_ticker",
                "benchmark_ticker",
                "predicted_error_pct",
                "actual_error_pct",
                "risk_bucket",
                "anomaly_flag",
                "action",
                "pair",
            ]
        ].rename(columns={"action": "arbitrage_signal"})
        st.dataframe(style_live_table(top10_display), use_container_width=True)

    with top_tabs[1]:
        st.subheader("Portfolio-Level Risk Aggregation")

        if portfolio_weight_errors:
            for message in portfolio_weight_errors:
                st.warning(message)

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("Portfolio Total Predicted TE", f"{portfolio_summary.total_predicted_te_pct:.3f}%")
        with p2:
            var_label = f"TE VaR ({int(float(portfolio_var_conf) * 100)}%)"
            var_value = (
                f"{portfolio_summary.portfolio_te_var_pct:.3f}%"
                if pd.notna(portfolio_summary.portfolio_te_var_pct)
                else "N/A"
            )
            st.metric(var_label, var_value)
        with p3:
            st.metric("Aggregate Arbitrage Risk", f"{portfolio_summary.aggregate_arbitrage_risk_score:.1f}")
        with p4:
            st.metric("Actionable Weight", f"{portfolio_summary.actionable_weight_pct:.1f}%")

        pc1, pc2 = st.columns([1.3, 1.0])
        with pc1:
            st.plotly_chart(build_portfolio_risk_timeseries_figure(portfolio_series), use_container_width=True)
        with pc2:
            st.plotly_chart(build_sector_exposure_figure(portfolio_sector_exposure), use_container_width=True)

        st.markdown("**ETF Contribution to Portfolio Risk**")
        if portfolio_contributions.empty:
            st.info("No portfolio contribution available. Verify ETF weights and data availability.")
        else:
            contribution_table = portfolio_contributions[
                [
                    "etf_ticker",
                    "pair",
                    "weight",
                    "predicted_tracking_error",
                    "risk_contribution_pct",
                    "risk_bucket",
                ]
            ].copy()
            contribution_table = contribution_table.rename(
                columns={
                    "weight": "portfolio_weight",
                    "predicted_tracking_error": "predicted_te",
                }
            )
            st.dataframe(
                contribution_table.style.format(
                    {
                        "portfolio_weight": "{:.2%}",
                        "predicted_te": "{:.4%}",
                        "risk_contribution_pct": "{:.2f}%",
                    }
                ),
                use_container_width=True,
            )

        st.markdown("**Normalized Portfolio Weights**")
        st.write(normalized_weight_map if normalized_weight_map else {"info": "No valid normalized weights."})

    with top_tabs[2]:
        st.subheader(f"Detailed View - {selected_pair}")

        pair_regime_result = regime_latest_map.get(selected_pair, {})
        pair_regime_history = regime_history_map.get(selected_pair, pd.DataFrame())

        rc1, rc2 = st.columns([0.8, 1.2])
        with rc1:
            st.markdown("**Detected Regime**")
            st.markdown(
                regime_badge_html(str(pair_regime_result.get("current_regime", "Calm"))),
                unsafe_allow_html=True,
            )
            st.metric("Confidence", f"{float(pair_regime_result.get('confidence', 0.0)):.2f}")
            st.caption(str(pair_regime_result.get("explanation", "")))

            regime_prob = pair_regime_result.get("regime_probability", {})
            st.write(
                {
                    "Calm": round(float(regime_prob.get("Calm", 0.0)), 3),
                    "Stress": round(float(regime_prob.get("Stress", 0.0)), 3),
                    "High_Vol": round(float(regime_prob.get("High_Vol", 0.0)), 3),
                }
            )
        with rc2:
            st.plotly_chart(
                build_regime_history_chart(pair_regime_history, pair_name=selected_pair),
                use_container_width=True,
            )

        residual_panel = _build_residual_panel(
            prediction_series=prediction_series,
            feature_panel=rt_feature_panel,
            pair_name=selected_pair,
        )

        # Build a separate daily panel for the 30-day history context.
        try:
            daily_market_panel = load_market_panel(period="60d", interval="1d")
            daily_feature_panel = build_feature_panel(
                data=daily_market_panel,
                horizon=int(horizon),
                rolling_window=int(rolling_window),
            )
        except Exception:
            daily_feature_panel = pd.DataFrame(columns=rt_feature_panel.columns)

        st.plotly_chart(
            build_history_chart(
                pair_name=selected_pair,
                intraday_residual_panel=residual_panel,
                daily_feature_panel=daily_feature_panel,
            ),
            use_container_width=True,
        )

        lower_left, lower_right = st.columns([1.1, 1.0])

        with lower_left:
            st.markdown("**SHAP Explanation (Latest Prediction)**")
            pair_features = rt_feature_panel[rt_feature_panel["pair"] == selected_pair].dropna()
            if pair_features.empty:
                st.info("Not enough clean feature rows for SHAP explainability.")
            else:
                input_columns = model.numeric_columns + model.categorical_columns
                latest_observation = pair_features.tail(1)[input_columns]

                explainer = TrackingErrorExplainer(model)
                shap_df = explainer.explain_observation(latest_observation)
                st.plotly_chart(_build_shap_waterfall_figure(shap_df, top_features=10), use_container_width=True)

                with st.expander("Counterfactual What-If Scenarios", expanded=False):
                    counterfactuals = explainer.generate_counterfactuals(
                        observation=latest_observation,
                        num_counterfactuals=3,
                        max_features_to_change=3,
                    )
                    for cf in counterfactuals:
                        st.write(
                            {
                                "scenario": cf.scenario_name,
                                "original_prediction": cf.original_prediction,
                                "counterfactual_prediction": cf.counterfactual_prediction,
                                "improvement_bps": cf.improvement_bps,
                                "changed_features": cf.changed_features,
                                "narrative": cf.narrative,
                            }
                        )

        with lower_right:
            st.markdown("**Anomaly Detection (Previous Module Output)**")
            latest_anomaly = anomaly_latest_map.get(selected_pair, {})

            a1, a2, a3 = st.columns(3)
            with a1:
                st.metric("Anomaly", "Yes" if bool(latest_anomaly.get("anomaly_detected", False)) else "No")
            with a2:
                st.metric("Type", str(latest_anomaly.get("anomaly_type", "none")))
            with a3:
                st.metric("Confidence", f"{float(latest_anomaly.get('confidence', 0.0)):.2f}")

            st.caption(str(latest_anomaly.get("explanation", "No anomaly diagnostics available.")))
            st.caption(f"Recommended action: {latest_anomaly.get('recommended_action', 'Monitor')}")

            pair_anomaly_history = anomaly_history_map.get(selected_pair, pd.DataFrame())
            if not pair_anomaly_history.empty:
                recent_anomalies = pair_anomaly_history[pair_anomaly_history["anomaly_detected"] == True].copy()  # noqa: E712
                if recent_anomalies.empty:
                    st.info("No structural anomalies detected for this pair in the current history window.")
                else:
                    st.dataframe(
                        recent_anomalies[
                            [
                                "residual",
                                "anomaly_type",
                                "confidence",
                                "score",
                                "recommended_action",
                            ]
                        ].tail(10),
                        use_container_width=True,
                    )
            else:
                st.info("Insufficient residual history for robust anomaly diagnostics.")

    with top_tabs[3]:
        st.subheader("Arbitrage Signal Panel")

        if universe_signals.empty:
            st.info("No actionable arbitrage signals generated at the current timestamp.")
        else:
            signal_display = universe_signals[
                [
                    "timestamp",
                    "pair",
                    "action",
                    "confidence",
                    "recommended_shares",
                    "notional",
                    "estimated_profit",
                    "estimated_profit_bps",
                    "regime",
                    "regime_confidence",
                    "applied_alert_tracking_error_pct",
                    "applied_confidence_threshold",
                    "reason",
                ]
            ].copy()

            signal_display = signal_display.rename(
                columns={
                    "action": "arbitrage_signal",
                    "confidence": "signal_confidence",
                    "notional": "recommended_size_usd",
                }
            )
            st.dataframe(
                signal_display.style.format(
                    {
                        "signal_confidence": "{:.2f}",
                        "recommended_size_usd": "${:,.0f}",
                        "estimated_profit": "${:,.0f}",
                        "estimated_profit_bps": "{:.1f}",
                        "regime_confidence": "{:.2f}",
                        "applied_alert_tracking_error_pct": "{:.3f}%",
                        "applied_confidence_threshold": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )

            pair_signal = universe_signals[universe_signals["pair"] == selected_pair]
            if not pair_signal.empty:
                signal = pair_signal.iloc[0]
                st.markdown(f"**Desk Recommendation - {selected_pair}**")
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.metric("Signal", str(signal["action"]))
                with s2:
                    st.metric("Confidence", f"{float(signal['confidence']):.2f}")
                with s3:
                    st.metric("Recommended Size", f"${float(signal['notional']):,.0f}")
                with s4:
                    st.metric("Estimated Profit", f"${float(signal['estimated_profit']):,.0f}")
                st.caption(str(signal["reason"]))

    with top_tabs[4]:
        st.subheader("Alerting System")

        st.markdown("**Per-ETF Alert Enablement**")
        pair_toggle_columns = st.columns(len(PAIR_CONFIGS))
        for idx, etf in enumerate(PAIR_CONFIGS.keys()):
            with pair_toggle_columns[idx]:
                st.session_state.pair_alert_enabled[etf] = st.checkbox(
                    etf,
                    value=st.session_state.pair_alert_enabled.get(etf, True),
                    key=f"alert_toggle_{etf}",
                )

        alert_config = {
            "enabled": bool(auto_alerts_enabled),
            "threshold_pct": float(threshold_pct),
            "persistence_minutes": int(persistence_minutes),
            "email_enabled": bool(email_enabled),
            "slack_enabled": bool(slack_enabled),
            "smtp_host": smtp_host,
            "smtp_port": int(smtp_port),
            "username": smtp_user,
            "password": smtp_password,
            "from_email": from_email,
            "to_email": to_email,
            "slack_webhook_url": slack_webhook_url,
        }

        # Dispatch alerts only when the engine is enabled.
        if alert_config["enabled"]:
            now_ts = pd.Timestamp.utcnow().tz_localize(None)
            process_threshold_alerts(
                live_table=live_overview,
                alert_config=alert_config,
                now_ts=now_ts,
            )

        st.markdown("**Alert Trigger Rule**")
        st.write(
            {
                "threshold": f"|tracking error| >= {threshold_pct:.2f}%",
                "persistence": f">= {int(persistence_minutes)} minutes",
                "engine_enabled": bool(auto_alerts_enabled),
                "email_enabled": bool(email_enabled),
                "slack_enabled": bool(slack_enabled),
            }
        )

        st.markdown("**Sent Alert Log**")
        if st.session_state.alert_log:
            alert_log_df = pd.DataFrame(st.session_state.alert_log).sort_values("timestamp", ascending=False)
            st.dataframe(alert_log_df, use_container_width=True)
        else:
            st.info("No alerts sent in this session.")

    with top_tabs[5]:
        st.subheader("Historical & Analytics")

        analytics_top = st.columns(2)
        with analytics_top[0]:
            # Trend chart: mean predicted TE over time for market-level drift monitoring.
            if not prediction_series.empty:
                global_trend = (
                    prediction_series.groupby("timestamp", observed=True)["predicted_tracking_error"]
                    .mean()
                    .reset_index()
                )
                trend_fig = go.Figure(
                    go.Scatter(
                        x=global_trend["timestamp"],
                        y=global_trend["predicted_tracking_error"] * 100.0,
                        mode="lines",
                        line={"color": "#103a6f", "width": 2},
                        name="Mean Predicted TE",
                    )
                )
                trend_fig.update_layout(
                    title="Global Tracking Error Trend",
                    xaxis_title="Timestamp",
                    yaxis_title="Predicted TE (%)",
                    template="plotly_white",
                    height=350,
                    margin={"l": 20, "r": 20, "t": 45, "b": 10},
                )
                st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.info("Insufficient prediction history to plot global trend.")

        with analytics_top[1]:
            # Distribution chart: shape of absolute TE to monitor tail risk concentration.
            if not prediction_series.empty:
                distribution_fig = go.Figure(
                    go.Histogram(
                        x=(prediction_series["predicted_tracking_error"].abs() * 100.0),
                        nbinsx=30,
                        marker={"color": "#d97a1f"},
                        name="|Predicted TE|",
                    )
                )
                distribution_fig.update_layout(
                    title="Distribution of Absolute Predicted Errors",
                    xaxis_title="Absolute Predicted TE (%)",
                    yaxis_title="Frequency",
                    template="plotly_white",
                    height=350,
                    margin={"l": 20, "r": 20, "t": 45, "b": 10},
                )
                st.plotly_chart(distribution_fig, use_container_width=True)
            else:
                st.info("Insufficient prediction history to plot distribution.")

        st.markdown("**Backtesting Summary**")
        backtest_metrics = evaluate_backtest_metrics(feature_panel=feature_panel, model=model)
        if backtest_metrics:
            b1, b2, b3, b4, b5 = st.columns(5)
            with b1:
                st.metric("MAE", f"{backtest_metrics['mae']:.6f}")
            with b2:
                st.metric("RMSE", f"{backtest_metrics['rmse']:.6f}")
            with b3:
                st.metric("MAPE", f"{backtest_metrics['mape']:.4f}")
            with b4:
                st.metric("R2", f"{backtest_metrics['r2']:.4f}")
            with b5:
                st.metric("Scored Rows", f"{int(backtest_metrics['scored_rows'])}")
        else:
            st.info("Backtest metrics are unavailable for the current sample size.")


if __name__ == "__main__":
    main()
