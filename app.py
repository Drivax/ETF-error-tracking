"""Streamlit dashboard for ETF tracking error monitoring and arbitrage signals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    ARBITRAGE_WINDOW,
    DEFAULT_HORIZON,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    DEFAULT_WINDOW,
    MODEL_ARTIFACT_PATH,
    PAIR_CONFIGS,
    VOLATILITY_FILTER_QUANTILE,
    ZSCORE_ENTRY,
    ZSCORE_EXIT,
)
from src.arbitrage_detector import ArbitrageDetector
from src.data_loader import MarketDataLoader
from src.features import FeatureEngineer
from src.models import TrackingErrorModel

st.set_page_config(page_title="ETF Tracking Error Risk Dashboard", layout="wide")


@st.cache_data(ttl=1200)
def load_data(period: str, interval: str) -> pd.DataFrame:
    """Download pair universe from Yahoo Finance with caching."""
    loader = MarketDataLoader()
    return loader.fetch_universe(PAIR_CONFIGS, period=period, interval=interval)


@st.cache_data(ttl=1200)
def build_features(data: pd.DataFrame, horizon: int, window: int) -> pd.DataFrame:
    """Build feature panel with caching for dashboard speed."""
    engineer = FeatureEngineer(rolling_window=window, horizon=horizon)
    return engineer.transform_universe(data)


def load_or_train_model(feature_df: pd.DataFrame, model_path: str) -> TrackingErrorModel:
    """Load model from disk, or train quickly if no artifact exists."""
    artifact = Path(model_path)
    if artifact.exists():
        return TrackingErrorModel.load(artifact)

    model = TrackingErrorModel(random_state=42)
    model.train(feature_df, target_col="target_te", test_size=0.2)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    model.save(artifact)
    return model


def main() -> None:
    """Render dashboard layout and run latest scoring logic."""
    st.title("ETF Tracking Error Prediction and Arbitrage Monitoring")
    st.caption("Real-market ETF/index monitoring for risk and trading desks")

    with st.sidebar:
        st.subheader("Configuration")
        period = st.selectbox("Lookback Period", ["6mo", "1y", "2y", "5y"], index=2)
        interval = st.selectbox("Data Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)
        horizon = st.number_input("Forecast Horizon (steps)", min_value=1, max_value=20, value=DEFAULT_HORIZON)
        window = st.number_input("Rolling Window", min_value=5, max_value=120, value=DEFAULT_WINDOW)
        refresh = st.button("Run Monitoring")

    if not refresh:
        st.info("Select parameters and click 'Run Monitoring'.")
        return

    with st.spinner("Loading market data and scoring model..."):
        panel = load_data(period=period, interval=interval)
        feature_df = build_features(panel, horizon=horizon, window=window)
        model = load_or_train_model(feature_df, str(MODEL_ARTIFACT_PATH))

    detector = ArbitrageDetector(
        window=ARBITRAGE_WINDOW,
        zscore_entry=ZSCORE_ENTRY,
        zscore_exit=ZSCORE_EXIT,
        volatility_filter_quantile=VOLATILITY_FILTER_QUANTILE,
    )

    summary_rows: list[dict[str, str | float]] = []

    for pair_name, pair_features in feature_df.groupby("pair", observed=True):
        latest = pair_features.dropna().tail(1).copy()
        if latest.empty:
            continue

        x_latest = latest[model.numeric_columns + model.categorical_columns]
        prediction = float(model.predict(x_latest)[0])

        pair_panel = panel[panel["pair"] == pair_name]
        signal_snapshot = detector.latest_signal(pair_panel)

        summary_rows.append(
            {
                "pair": pair_name,
                "predicted_tracking_error": prediction,
                "signal": signal_snapshot["signal"],
                "spread_zscore": signal_snapshot["spread_zscore"],
                "spread_vol": signal_snapshot["spread_vol"],
                "half_life": signal_snapshot["half_life"],
            }
        )

    if not summary_rows:
        st.warning("No valid observations were available for scoring.")
        return

    summary_df = pd.DataFrame(summary_rows).sort_values("predicted_tracking_error", ascending=False)

    st.subheader("Latest Risk Snapshot")
    st.dataframe(summary_df, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Predicted Tracking Error by Pair")
        fig = px.bar(
            summary_df,
            x="pair",
            y="predicted_tracking_error",
            color="signal",
            title="Predicted Tracking Error",
        )
        st.plotly_chart(fig, use_container_width=True)

    selected_pair = st.selectbox("Select Pair for Diagnostics", summary_df["pair"].tolist())
    selected_panel = panel[panel["pair"] == selected_pair].copy()
    selected_signal_df = detector.add_signal_columns(selected_panel)

    with col_right:
        st.subheader("Spread Z-Score")
        fig_z = px.line(
            selected_signal_df.reset_index(),
            x=selected_signal_df.reset_index().columns[0],
            y="spread_zscore",
            title=f"Spread Z-Score: {selected_pair}",
        )
        fig_z.add_hline(y=ZSCORE_ENTRY, line_dash="dot")
        fig_z.add_hline(y=-ZSCORE_ENTRY, line_dash="dot")
        st.plotly_chart(fig_z, use_container_width=True)

    st.subheader("Feature Explainability (SHAP)")
    selected_features = feature_df[feature_df["pair"] == selected_pair].dropna().tail(200)
    if not selected_features.empty:
        x_shap = selected_features[model.numeric_columns + model.categorical_columns]
        shap_df = model.explain_shap(x_shap, max_samples=min(150, len(x_shap))).head(15)
        shap_fig = px.bar(
            shap_df,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Top SHAP Feature Importances",
        )
        st.plotly_chart(shap_fig, use_container_width=True)

        latest_row = x_shap.tail(1)
        counterfactual = model.counterfactual(latest_row)
        st.markdown("**Counterfactual Summary**")
        st.write(
            {
                "original_prediction": counterfactual["original_prediction"],
                "counterfactual_prediction": counterfactual["counterfactual_prediction"],
                "changed_features_count": len(counterfactual["changes"]),
            }
        )
    else:
        st.info("Insufficient non-null features for SHAP diagnostics on selected pair.")


if __name__ == "__main__":
    main()
