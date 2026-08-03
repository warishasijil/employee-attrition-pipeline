"""Utilities for selecting and saving the final machine-learning pipeline."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline


class ModelManager:
    """Select, package, save, and load trained models."""

    @staticmethod
    def select_best_model_name(
        comparison: pd.DataFrame,
        metric: str = "roc_auc",
    ) -> str:
        """
        Select the highest-ranked model using a configured metric.

        Args:
            comparison: Model-comparison table.
            metric: Metric used for ranking.

        Returns:
            Name of the best model.
        """
        if metric not in comparison.columns:
            raise ValueError(
                f"Selection metric is missing from comparison table: {metric}"
            )

        best_row = comparison.sort_values(
            by=metric,
            ascending=False,
        ).iloc[0]

        return str(best_row["model"])

    @staticmethod
    def build_pipeline(
        preprocessor: Any,
        model: ClassifierMixin,
    ) -> Pipeline:
        """
        Package the fitted preprocessor and fitted model together.

        Returns:
            A reusable scikit-learn inference pipeline.
        """
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    @staticmethod
    def save(
        model_pipeline: Pipeline,
        output_path: str | Path,
    ) -> None:
        """Save the final fitted pipeline."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model_pipeline, destination)

    @staticmethod
    def load(model_path: str | Path) -> Pipeline:
        """Load a previously saved fitted pipeline."""
        source = Path(model_path)

        if not source.exists():
            raise FileNotFoundError(
                f"Saved model was not found: {source.resolve()}"
            )

        return joblib.load(source)