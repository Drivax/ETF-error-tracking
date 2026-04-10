"""Streamlit dashboard for real-time ETF tracking error risk and arbitrage monitoring."""

from __future__ import annotations

from pathlib import Path

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
from src.data_loader import MarketDataLoader
from src.explainability import TrackingErrorExplainer
from src.features import FeatureEngineer
from src.models import TrackingErrorModel
from src.real_time_predictor import RealTimeTrackingErrorPredictor

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
        rolling_window=ARBITRAGE_WINDOW,
        holding_bars=max(_get_interval_minutes(interval) // 5, 3),
        execution_cost_bps=3.0,
        slippage_bps=2.0,
        base_entry_zscore=2.0,
        exit_zscore=0.5,
        risk_budget_notional=1_000_000.0,
    )

    # Calibrate threshold globally from available history once per run.
    signal_generator.calibrate_threshold(market_panel, minimum_samples=60)

    universe_signals = signal_generator.generate_universe_signals(
        intraday_panel=market_panel,
        prediction_snapshot=latest_predictions,
    )

    if universe_signals.empty:
        st.info("No actionable arbitrage signals at the latest timestamp.")
    else:
        st.dataframe(universe_signals, use_container_width=True)

        pair_signal = universe_signals[universe_signals["pair"] == selected_pair]
        if not pair_signal.empty:
            st.markdown("**Selected Pair Trade Recommendation**")
            st.json(pair_signal.iloc[0].to_dict())


if __name__ == "__main__":
    main()
