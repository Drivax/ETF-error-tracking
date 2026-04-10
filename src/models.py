"""Modeling module for ETF tracking error prediction and explainability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.utils import safe_mape, time_split


@dataclass
class TrainResult:
    """Container for model and evaluation outputs."""

    metrics: dict[str, float]
    train_rows: int
    test_rows: int


class TrackingErrorModel:
    """Train, evaluate, explain, and persist a tracking error model."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.training_feature_stats: dict[str, dict[str, float]] = {}

    @staticmethod
    def _select_feature_columns(data: pd.DataFrame, target_col: str) -> list[str]:
        excluded = {
            target_col,
            "etf_ticker",
            "benchmark_ticker",
            "signal",
        }
        return [col for col in data.columns if col not in excluded]

    def _build_pipeline(self, feature_frame: pd.DataFrame) -> Pipeline:
        self.categorical_columns = [col for col in ["pair"] if col in feature_frame.columns]
        self.numeric_columns = [col for col in feature_frame.columns if col not in self.categorical_columns]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", RobustScaler(), self.numeric_columns),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_columns),
            ],
            remainder="drop",
        )

        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            l2_regularization=0.1,
            random_state=self.random_state,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        return pipeline

    def _record_feature_stats(self, data: pd.DataFrame) -> None:
        self.training_feature_stats = {}
        for col in self.numeric_columns:
            self.training_feature_stats[col] = {
                "mean": float(data[col].mean(skipna=True)),
                "std": float(data[col].std(skipna=True) + 1e-8),
                "min": float(data[col].min(skipna=True)),
                "max": float(data[col].max(skipna=True)),
            }

    def train(
        self,
        dataset: pd.DataFrame,
        target_col: str = "target_te",
        test_size: float = 0.2,
    ) -> TrainResult:
        """Train the model with temporal split and compute regression metrics."""
        required = {target_col}
        if missing := required.difference(dataset.columns):
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        model_df = dataset.dropna(subset=[target_col]).copy()
        feature_columns = self._select_feature_columns(model_df, target_col)
        model_df = model_df.dropna(subset=feature_columns)

        train_df, test_df = time_split(model_df, test_size=test_size)
        # Minimum sample checks protect against unstable metrics on tiny segments.
        if len(train_df) < 100 or len(test_df) < 20:
            raise ValueError("Insufficient rows after preprocessing for reliable training/evaluation.")

        x_train = train_df[feature_columns]
        y_train = train_df[target_col]
        x_test = test_df[feature_columns]
        y_test = test_df[target_col]

        self.pipeline = self._build_pipeline(x_train)
        self.pipeline.fit(x_train, y_train)

        self._record_feature_stats(x_train)

        predictions = self.pipeline.predict(x_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "r2": float(r2_score(y_test, predictions)),
            "mape": float(safe_mape(y_test.values, predictions)),
        }

        return TrainResult(metrics=metrics, train_rows=len(train_df), test_rows=len(test_df))

    def predict(self, feature_frame: pd.DataFrame) -> np.ndarray:
        """Generate predictions from an input feature frame."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained or loaded.")
        return self.pipeline.predict(feature_frame)

    def explain_shap(self, feature_frame: pd.DataFrame, max_samples: int = 250) -> pd.DataFrame:
        """Return mean absolute SHAP values per feature for a sample frame."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained or loaded.")

        sample = feature_frame.head(max_samples).copy()
        preprocessor = self.pipeline.named_steps["preprocessor"]
        model = self.pipeline.named_steps["model"]

        transformed = preprocessor.transform(sample)
        # TreeExplainer expects a dense matrix for this estimator and sklearn pipeline combination.
        transformed_dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed

        feature_names = preprocessor.get_feature_names_out()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed_dense)

        importance = np.abs(shap_values).mean(axis=0)
        out = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": importance,
            }
        ).sort_values("mean_abs_shap", ascending=False)

        return out

    def counterfactual(
        self,
        observation: pd.DataFrame,
        distance_weight: float = 0.1,
    ) -> dict[str, Any]:
        """Find a close-by feature configuration that reduces predicted TE."""
        if self.pipeline is None:
            raise RuntimeError("Model has not been trained or loaded.")
        if len(observation) != 1:
            raise ValueError("Counterfactual requires a single-row DataFrame.")

        row = observation.copy()
        original_prediction = float(self.predict(row)[0])

        variable_columns = [col for col in self.numeric_columns if col in row.columns]
        if not variable_columns:
            return {
                "original_prediction": original_prediction,
                "counterfactual_prediction": original_prediction,
                "changes": {},
            }

        base_values = row[variable_columns].iloc[0].astype(float).values

        bounds = []
        for col, base in zip(variable_columns, base_values):
            stats = self.training_feature_stats.get(col)
            if not stats:
                bounds.append((base * 0.5, base * 1.5))
                continue

            spread = max(2.0 * stats["std"], 1e-4)
            low = max(stats["min"], base - spread)
            high = min(stats["max"], base + spread)
            if low >= high:
                low, high = base - spread, base + spread
            bounds.append((low, high))

        std_vector = np.array([self.training_feature_stats[col]["std"] for col in variable_columns])

        def objective(x: np.ndarray) -> float:
            trial = row.copy()
            trial.loc[trial.index[0], variable_columns] = x
            pred = float(self.predict(trial)[0])
            # Penalize large feature moves to keep counterfactuals close to feasible market states.
            distance = float(np.mean(np.abs((x - base_values) / std_vector)))
            return pred + distance_weight * distance

        result = minimize(
            objective,
            x0=base_values,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 200},
        )

        best_x = result.x if result.success else base_values
        counterfactual_row = row.copy()
        counterfactual_row.loc[counterfactual_row.index[0], variable_columns] = best_x
        counterfactual_prediction = float(self.predict(counterfactual_row)[0])

        deltas = {
            col: float(counterfactual_row[col].iloc[0] - row[col].iloc[0])
            for col in variable_columns
            if abs(counterfactual_row[col].iloc[0] - row[col].iloc[0]) > 1e-10
        }

        return {
            "original_prediction": original_prediction,
            "counterfactual_prediction": counterfactual_prediction,
            "changes": deltas,
        }

    def save(self, path: str | Path) -> None:
        """Persist model object and metadata to disk."""
        payload = {
            "pipeline": self.pipeline,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "training_feature_stats": self.training_feature_stats,
            "random_state": self.random_state,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "TrackingErrorModel":
        """Load model object and metadata from disk."""
        payload = joblib.load(path)
        model = cls(random_state=payload.get("random_state", 42))
        model.pipeline = payload["pipeline"]
        model.numeric_columns = payload["numeric_columns"]
        model.categorical_columns = payload["categorical_columns"]
        model.training_feature_stats = payload["training_feature_stats"]
        return model
