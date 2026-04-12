"""Streamlit dashboard for real-time ETF tracking error risk and arbitrage monitoring."""

from __future__ import annotations

from pathlib import Path

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
    MODEL_ARTIFACT_PATH,
    PAIR_CONFIGS,
)
from src.arbitrage_signal import ArbitrageSignalGenerator
from src.anomaly_detector import ResidualAnomalyDetector
from src.data_loader import MarketDataLoader
from src.explainability import TrackingErrorExplainer
from src.features import FeatureEngineer
from src.models import TrackingErrorModel
from src.real_time_predictor import RealTimeTrackingErrorPredictor
from src.utils import time_split

st.set_page_config(page_title="ETF Real-Time Tracking Error Desk", layout="wide")


def _get_interval_minutes(interval: str) -> int:
    """Convert yfinance interval strings to minutes for display labels."""
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


@st.cache_data(ttl=600)
def load_market_panel(period: str, interval: str) -> pd.DataFrame:
    """Fetch ETF-benchmark universe panel from Yahoo Finance."""
    loader = MarketDataLoader()
    return loader.fetch_universe(PAIR_CONFIGS, period=period, interval=interval)


@st.cache_data(ttl=600)
def build_feature_panel(data: pd.DataFrame, horizon: int, rolling_window: int) -> pd.DataFrame:
    """Create model-ready feature panel with caching for dashboard responsiveness."""
    engineer = FeatureEngineer(rolling_window=rolling_window, horizon=horizon)
    return engineer.transform_universe(data)


def load_or_train_model(feature_panel: pd.DataFrame, artifact_path: Path) -> TrackingErrorModel:
    """Load persisted model artifact, or train quickly if no artifact exists yet."""
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
    """Compute current holdout metrics from the latest feature panel.

    This mirrors the project's time-based validation protocol and lets the app show
    up-to-date model quality after retraining or data refresh.
    """
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
    mape = float(np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-8))) )
    r2 = float(1 - np.sum((y_test - y_pred) ** 2) / (np.sum((y_test - np.mean(y_test)) ** 2) + 1e-12))

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "scored_rows": float(len(test_df)),
    }


def _build_shap_waterfall_figure(shap_df: pd.DataFrame, top_features: int = 10) -> go.Figure:
    """Render a compact SHAP waterfall-style chart for one prediction."""
    ranked = shap_df.head(top_features).copy()
    ranked = ranked.sort_values("shap_value", ascending=False)

    fig = go.Figure(
        go.Waterfall(
            name="SHAP Contribution",
            orientation="v",
            measure=["relative"] * len(ranked),
            x=ranked["feature"].tolist(),
            y=ranked["shap_value"].tolist(),
            connector={"line": {"color": "rgba(60, 60, 60, 0.4)"}},
        )
    )

    fig.update_layout(
        title="SHAP Waterfall (Top Feature Contributions)",
        xaxis_title="Feature",
        yaxis_title="Contribution to Predicted Tracking Error",
        template="plotly_white",
        height=420,
    )
    return fig


def _build_residual_panel(
    prediction_series: pd.DataFrame,
    feature_panel: pd.DataFrame,
    pair_name: str,
) -> pd.DataFrame:
    """Join predictions with realized/target TE to compute residual history.

    The anomaly detector expects residuals defined as:
    residual = actual_tracking_error - predicted_tracking_error
    """
    pair_predictions = prediction_series[prediction_series["pair"] == pair_name].copy()
    if pair_predictions.empty:
        return pd.DataFrame()

    pair_actuals = feature_panel[feature_panel["pair"] == pair_name].copy()
    if pair_actuals.empty:
        return pd.DataFrame()

    # Prefer predictive target when available; otherwise fallback to realized TE.
    pair_actuals = pair_actuals.reset_index().rename(columns={"index": "timestamp"})
    pair_actuals["actual_tracking_error"] = pair_actuals["target_te"].where(
        pair_actuals["target_te"].notna(),
        pair_actuals["realized_te"],
    )

    merged = pair_predictions.merge(
        pair_actuals[["timestamp", "actual_tracking_error"]],
        on="timestamp",
        how="left",
    )
    merged = merged.dropna(subset=["predicted_tracking_error", "actual_tracking_error"]).copy()
    if merged.empty:
        return merged

    merged["residual"] = merged["actual_tracking_error"] - merged["predicted_tracking_error"]
    return merged.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    """Render dashboard with real-time prediction, explainability, and arbitrage panels."""
    st.title("ETF Intraday Tracking Error Intelligence")
    st.caption("Production-style monitoring for ETF risk and trading desks")

    with st.sidebar:
        st.subheader("Run Configuration")

        mode = st.selectbox(
            "Data Mode",
            ["Intraday", "Daily"],
            index=0,
            help="Intraday mode supports 5-15 minute monitoring cycles.",
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

        intraday_default_period = DEFAULT_INTRADAY_PERIOD if mode == "Intraday" else DEFAULT_PERIOD
        intraday_default_interval = DEFAULT_INTRADAY_INTERVAL if mode == "Intraday" else DEFAULT_INTERVAL
        st.caption(
            f"Config defaults: period={intraday_default_period}, interval={intraday_default_interval}"
        )

        run_button = st.button("Run Real-Time Desk")

    if not run_button:
        st.info("Select parameters and click 'Run Real-Time Desk' to refresh predictions.")
        return

    with st.spinner("Loading market panel and running forecasting pipeline..."):
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

    if latest_predictions.empty:
        st.warning("No valid rows available for real-time scoring.")
        return

    st.subheader("Live Tracking Error Snapshot")
    st.dataframe(latest_predictions, use_container_width=True)

    backtest_metrics = evaluate_backtest_metrics(feature_panel=feature_panel, model=model)
    if backtest_metrics:
        st.subheader("Latest Backtest Results")
        metric_a, metric_b, metric_c, metric_d, metric_e = st.columns(5)
        with metric_a:
            st.metric("MAE", f"{backtest_metrics['mae']:.6f}")
        with metric_b:
            st.metric("RMSE", f"{backtest_metrics['rmse']:.6f}")
        with metric_c:
            st.metric("MAPE", f"{backtest_metrics['mape']:.4f}")
        with metric_d:
            st.metric("R2", f"{backtest_metrics['r2']:.4f}")
        with metric_e:
            st.metric("Scored Rows", f"{int(backtest_metrics['scored_rows'])}")

    selected_pair = st.selectbox("Select Pair", latest_predictions["pair"].tolist())
    selected_series = prediction_series[prediction_series["pair"] == selected_pair].copy()
    selected_series = selected_series.sort_values("timestamp")

    if not selected_series.empty:
        chart = go.Figure()
        chart.add_trace(
            go.Scatter(
                x=selected_series["timestamp"],
                y=selected_series["predicted_tracking_error"],
                mode="lines",
                name="Predicted TE",
                line={"color": "#1f77b4", "width": 2},
            )
        )
        chart.add_trace(
            go.Scatter(
                x=selected_series["timestamp"],
                y=selected_series["ci_upper"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        chart.add_trace(
            go.Scatter(
                x=selected_series["timestamp"],
                y=selected_series["ci_lower"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(31, 119, 180, 0.15)",
                name=f"{int(confidence_level * 100)}% CI",
            )
        )

        chart.update_layout(
            title=f"{selected_pair}: Predicted Tracking Error every {_get_interval_minutes(interval)} minutes",
            xaxis_title="Timestamp",
            yaxis_title="Tracking Error",
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(chart, use_container_width=True)

    st.subheader("Anomaly Detection Panel")
    residual_panel = _build_residual_panel(
        prediction_series=prediction_series,
        feature_panel=rt_feature_panel,
        pair_name=selected_pair,
    )

    detector_short_window = max(10, int(rolling_window // 2))
    detector_long_window = max(24, int(rolling_window * 2))
    required_observations = max(80, detector_long_window * 2)

    if len(residual_panel) < required_observations:
        st.info(
            "Not enough residual history for robust structural anomaly detection. "
            f"Need at least {required_observations} aligned observations for the current "
            f"window settings (available: {len(residual_panel)})."
        )
    else:
        detector = ResidualAnomalyDetector(
            short_window=detector_short_window,
            long_window=detector_long_window,
            contamination=0.05,
            normal_quantile=0.85,
            random_state=42,
        )

        anomaly_scored = detector.fit_score(
            actual_tracking_error=residual_panel["actual_tracking_error"],
            predicted_tracking_error=residual_panel["predicted_tracking_error"],
        )
        latest_anomaly = detector.latest_result(
            actual_tracking_error=residual_panel["actual_tracking_error"],
            predicted_tracking_error=residual_panel["predicted_tracking_error"],
        )

        anomaly_plot_data = residual_panel.copy()
        aligned_scores = anomaly_scored.reset_index().rename(columns={"index": "timestamp"})
        anomaly_plot_data = anomaly_plot_data.merge(
            aligned_scores[["timestamp", "anomaly_detected", "confidence", "anomaly_type"]],
            on="timestamp",
            how="left",
        )

        anomaly_chart = go.Figure()
        anomaly_chart.add_trace(
            go.Scatter(
                x=anomaly_plot_data["timestamp"],
                y=anomaly_plot_data["residual"],
                mode="lines",
                name="Residual",
                line={"color": "#2f4b7c", "width": 2},
            )
        )

        flagged = anomaly_plot_data[anomaly_plot_data["anomaly_detected"] == True]
        if not flagged.empty:
            anomaly_chart.add_trace(
                go.Scatter(
                    x=flagged["timestamp"],
                    y=flagged["residual"],
                    mode="markers",
                    name="Detected Anomaly",
                    marker={"color": "#d62728", "size": 9, "symbol": "diamond"},
                    customdata=np.stack(
                        [
                            flagged["anomaly_type"].fillna("unknown").astype(str).to_numpy(),
                            flagged["confidence"].fillna(0.0).astype(float).to_numpy(),
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "Timestamp: %{x}<br>"
                        "Residual: %{y:.6f}<br>"
                        "Type: %{customdata[0]}<br>"
                        "Confidence: %{customdata[1]:.2f}<extra></extra>"
                    ),
                )
            )

        anomaly_chart.update_layout(
            title=f"{selected_pair}: Residual Structure and Detected Anomalies",
            xaxis_title="Timestamp",
            yaxis_title="Residual (Actual - Predicted Tracking Error)",
            template="plotly_white",
            height=420,
        )
        st.plotly_chart(anomaly_chart, use_container_width=True)

        status_col, type_col, confidence_col, score_col = st.columns(4)
        with status_col:
            st.metric("Latest Anomaly", "Yes" if latest_anomaly["anomaly_detected"] else "No")
        with type_col:
            st.metric("Anomaly Type", str(latest_anomaly["anomaly_type"]))
        with confidence_col:
            st.metric("Confidence", f"{float(latest_anomaly['confidence']):.2f}")
        with score_col:
            st.metric("Ensemble Score", f"{float(latest_anomaly['score']):.4f}")

        st.caption(str(latest_anomaly["explanation"]))
        st.caption(f"Recommended action: {latest_anomaly['recommended_action']}")

        recent_anomalies = anomaly_scored[anomaly_scored["anomaly_detected"] == True].copy()
        if recent_anomalies.empty:
            st.info("No structural anomalies were detected for the selected pair in the current window.")
        else:
            show_cols = [
                "residual",
                "autoencoder_error",
                "isolation_score",
                "mean_shift",
                "volatility_ratio",
                "autocorr_shift",
                "anomaly_type",
                "confidence",
                "explanation",
            ]
            st.markdown("**Recent Detected Anomalies**")
            st.dataframe(
                recent_anomalies[show_cols]
                .tail(12)
                .sort_index(ascending=False),
                use_container_width=True,
            )

    st.subheader("SHAP Explainability + Counterfactuals")
    pair_features = rt_feature_panel[rt_feature_panel["pair"] == selected_pair].dropna()
    if pair_features.empty:
        st.info("Not enough clean feature rows for explainability output.")
    else:
        input_columns = model.numeric_columns + model.categorical_columns
        latest_observation = pair_features.tail(1)[input_columns]

        explainer = TrackingErrorExplainer(model)
        shap_df = explainer.explain_observation(latest_observation)
        waterfall = _build_shap_waterfall_figure(shap_df, top_features=10)
        st.plotly_chart(waterfall, use_container_width=True)

        counterfactuals = explainer.generate_counterfactuals(
            observation=latest_observation,
            num_counterfactuals=3,
            max_features_to_change=3,
        )
        for counterfactual in counterfactuals:
            st.markdown(f"**{counterfactual.scenario_name}**")
            st.write(
                {
                    "original_prediction": counterfactual.original_prediction,
                    "counterfactual_prediction": counterfactual.counterfactual_prediction,
                    "improvement_bps": counterfactual.improvement_bps,
                    "changed_features": counterfactual.changed_features,
                    "narrative": counterfactual.narrative,
                }
            )

    st.subheader("Arbitrage Signal Panel")
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
    )

    if universe_signals.empty:
        st.info("No actionable arbitrage signals at the latest timestamp.")
    else:
        display_columns = [
            "pair",
            "action",
            "confidence",
            "recommended_shares",
            "notional",
            "estimated_profit",
            "reason",
        ]
        st.dataframe(universe_signals[display_columns], use_container_width=True)

        pair_signal = universe_signals[universe_signals["pair"] == selected_pair]
        if not pair_signal.empty:
            st.markdown("**Selected Pair Trade Recommendation**")
            signal = pair_signal.iloc[0]

            metric_left, metric_mid, metric_right, metric_far = st.columns(4)
            with metric_left:
                st.metric("Action", str(signal["action"]))
            with metric_mid:
                st.metric("Confidence", f"{float(signal['confidence']):.2f}")
            with metric_right:
                st.metric("Recommended Shares", f"{int(signal['recommended_shares']):,}")
            with metric_far:
                st.metric("Notional", f"${float(signal['notional']):,.0f}")

            profit_col, score_col = st.columns(2)
            with profit_col:
                st.metric("Estimated Net Profit", f"${float(signal['estimated_profit']):,.0f}")
            with score_col:
                st.metric("Expected Profit (bps)", f"{float(signal['estimated_profit_bps']):.1f}")

            st.caption(str(signal["reason"]))


if __name__ == "__main__":
    main()
