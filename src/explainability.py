"""Explainability toolkit for tracking error predictions.

This module provides two production-oriented capabilities:
1) SHAP attributions for individual or batch predictions.
2) Realistic counterfactual scenarios that suggest feature shifts likely to
   reduce predicted tracking error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from src.models import TrackingErrorModel


@dataclass(frozen=True)
class CounterfactualResult:
    """Structured output for one counterfactual recommendation."""

    scenario_name: str
    original_prediction: float
    counterfactual_prediction: float
    improvement_bps: float
    changed_features: dict[str, float]
    narrative: str


class TrackingErrorExplainer:
    """Compute SHAP diagnostics and generate constrained counterfactual scenarios."""

    def __init__(self, model: TrackingErrorModel) -> None:
        if model.pipeline is None:
            raise RuntimeError("Model pipeline is not initialized. Train or load the model first.")

        self.model = model
        self.preprocessor = model.pipeline.named_steps["preprocessor"]
        self.estimator = model.pipeline.named_steps["model"]
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        self.explainer = shap.TreeExplainer(self.estimator)

    def _transform_features(self, feature_frame: pd.DataFrame) -> np.ndarray:
        """Transform raw model inputs into the matrix expected by TreeExplainer."""
        transformed = self.preprocessor.transform(feature_frame)
        return transformed.toarray() if hasattr(transformed, "toarray") else transformed

    def explain_observation(self, observation: pd.DataFrame) -> pd.DataFrame:
        """Return SHAP decomposition for a single observation.

        Parameters
        ----------
        observation:
            One-row DataFrame containing model input columns.
        """
        if len(observation) != 1:
            raise ValueError("explain_observation expects exactly one row.")

        transformed = self._transform_features(observation)
        shap_values = self.explainer.shap_values(transformed)

        # For regression models this is a scalar expected value.
        expected_value = float(np.array(self.explainer.expected_value).reshape(-1)[0])
        row_values = np.asarray(shap_values).reshape(-1)

        contribution_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "shap_value": row_values,
                "abs_shap_value": np.abs(row_values),
                "expected_value": expected_value,
            }
        ).sort_values("abs_shap_value", ascending=False)

        prediction = float(self.model.predict(observation)[0])
        contribution_df["prediction"] = prediction
        contribution_df["direction"] = contribution_df["shap_value"].apply(
            lambda value: "increases_te" if value >= 0 else "reduces_te"
        )

        return contribution_df.reset_index(drop=True)

    def explain_batch(self, feature_frame: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame:
        """Compute SHAP values for a batch and return long-format output.

        This function is intentionally explicit (not just aggregated importance), so
        desks can audit each prediction and trace which factors were most influential.
        """
        if feature_frame.empty:
            return pd.DataFrame(columns=["row_id", "feature", "shap_value", "abs_shap_value"])

        sample = feature_frame.head(max_rows).copy()
        transformed = self._transform_features(sample)
        shap_values = self.explainer.shap_values(transformed)

        shap_matrix = np.asarray(shap_values)
        if shap_matrix.ndim == 1:
            shap_matrix = shap_matrix.reshape(1, -1)

        rows: list[dict[str, float | str | int]] = []
        for row_index in range(shap_matrix.shape[0]):
            for feature_index, feature_name in enumerate(self.feature_names):
                value = float(shap_matrix[row_index, feature_index])
                rows.append(
                    {
                        "row_id": int(row_index),
                        "feature": feature_name,
                        "shap_value": value,
                        "abs_shap_value": abs(value),
                    }
                )

        return pd.DataFrame(rows)

    @staticmethod
    def _clip_to_training_range(value: float, low: float, high: float) -> float:
        """Constrain scenario values to historically realistic feature ranges."""
        return float(np.clip(value, low, high))

    def _numeric_counterfactual_candidates(
        self,
        observation: pd.DataFrame,
        shap_df: pd.DataFrame,
        max_features_to_change: int,
    ) -> list[str]:
        """Select top numeric features pushing TE higher, based on SHAP ranking."""
        candidate_features: list[str] = []

        for _, row in shap_df.iterrows():
            transformed_name = str(row["feature"])
            if not transformed_name.startswith("num__"):
                continue

            original_name = transformed_name.replace("num__", "", 1)
            if original_name not in observation.columns:
                continue
            if original_name not in self.model.training_feature_stats:
                continue
            if float(row["shap_value"]) <= 0:
                # We focus on features increasing TE to find reduction opportunities.
                continue

            candidate_features.append(original_name)
            if len(candidate_features) >= max_features_to_change:
                break

        if not candidate_features:
            # Fallback: use top numeric features by absolute SHAP if no positive contributors found.
            for _, row in shap_df.iterrows():
                transformed_name = str(row["feature"])
                if not transformed_name.startswith("num__"):
                    continue
                original_name = transformed_name.replace("num__", "", 1)
                if original_name in observation.columns and original_name in self.model.training_feature_stats:
                    candidate_features.append(original_name)
                if len(candidate_features) >= max_features_to_change:
                    break

        return candidate_features

    def generate_counterfactuals(
        self,
        observation: pd.DataFrame,
        num_counterfactuals: int = 3,
        max_features_to_change: int = 3,
    ) -> list[CounterfactualResult]:
        """Generate realistic what-if scenarios that aim to reduce predicted TE.

        Strategy
        --------
        1) Identify top numeric features increasing TE through SHAP decomposition.
        2) Apply graduated shifts toward lower-risk values using training statistics.
        3) Keep changes within historical min/max bounds for realism.
        """
        if len(observation) != 1:
            raise ValueError("generate_counterfactuals expects exactly one observation row.")
        if num_counterfactuals < 1:
            raise ValueError("num_counterfactuals must be >= 1")

        shap_df = self.explain_observation(observation)
        candidate_features = self._numeric_counterfactual_candidates(
            observation=observation,
            shap_df=shap_df,
            max_features_to_change=max_features_to_change,
        )

        original_prediction = float(self.model.predict(observation)[0])
        scenario_scales = np.linspace(0.4, 1.0, num_counterfactuals)

        results: list[CounterfactualResult] = []
        for scenario_index, scale in enumerate(scenario_scales, start=1):
            scenario_row = observation.copy()
            changed_features: dict[str, float] = {}

            for feature_name in candidate_features:
                stats = self.model.training_feature_stats[feature_name]
                current_value = float(scenario_row.iloc[0][feature_name])
                std_step = max(float(stats["std"]), 1e-8)

                # Move one standard deviation step toward lower TE direction.
                # For numeric risk amplifiers, lowering the level usually reduces TE.
                target_value = current_value - scale * std_step
                target_value = self._clip_to_training_range(
                    target_value,
                    low=float(stats["min"]),
                    high=float(stats["max"]),
                )

                if abs(target_value - current_value) > 1e-12:
                    changed_features[feature_name] = float(target_value)
                    scenario_row.loc[scenario_row.index[0], feature_name] = target_value

            scenario_prediction = float(self.model.predict(scenario_row)[0])
            improvement_bps = float((original_prediction - scenario_prediction) * 10000.0)

            # Build a desk-friendly narrative with the largest feature move.
            if changed_features:
                feature_name, target_value = max(
                    changed_features.items(),
                    key=lambda item: abs(item[1] - float(observation.iloc[0][item[0]])),
                )
                baseline = float(observation.iloc[0][feature_name])
                if abs(baseline) > 1e-12:
                    pct_change = (target_value - baseline) / abs(baseline)
                    movement_text = f"{pct_change:+.1%}"
                else:
                    movement_text = f"{target_value - baseline:+.4f}"

                narrative = (
                    f"If {feature_name} moved {movement_text}, predicted tracking error "
                    f"would change by {improvement_bps:.1f} bp."
                )
            else:
                narrative = "No feasible numeric adjustment was found under current feature constraints."

            results.append(
                CounterfactualResult(
                    scenario_name=f"Scenario {scenario_index}",
                    original_prediction=original_prediction,
                    counterfactual_prediction=scenario_prediction,
                    improvement_bps=improvement_bps,
                    changed_features=changed_features,
                    narrative=narrative,
                )
            )

        # Present the best improvements first.
        results.sort(key=lambda item: item.improvement_bps, reverse=True)
        return results
