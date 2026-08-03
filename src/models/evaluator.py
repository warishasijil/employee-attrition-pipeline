"""Model evaluation utilities."""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelEvaluator:
    """Evaluate trained classification models."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def evaluate_model(
        model: ClassifierMixin,
        X_test: Any,
        y_test: Any,
    ) -> dict[str, float]:
        """Calculate classification metrics for one model."""
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        return {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "recall": recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "pr_auc": average_precision_score(
                y_test,
                probabilities,
            ),
        }

    def compare(
        self,
        models: dict[str, ClassifierMixin],
        X_test: Any,
        y_test: Any,
    ) -> pd.DataFrame:
        """Evaluate all models and save the comparison table."""
        results: list[dict[str, Any]] = []

        for model_name, model in models.items():
            metrics = self.evaluate_model(
                model,
                X_test,
                y_test,
            )

            results.append(
                {
                    "model": model_name,
                    **metrics,
                }
            )

        comparison = pd.DataFrame(results).sort_values(
            by="roc_auc",
            ascending=False,
        )

        comparison.to_csv(self.output_path, index=False)

        return comparison