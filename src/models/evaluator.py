"""Model evaluation and visualization utilities."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelEvaluator:
    """Evaluate trained classification models and save reports."""

    def __init__(
        self,
        output_path: str | Path,
        figures_directory: str | Path = "reports/figures",
        metrics_directory: str | Path = "reports/metrics",
    ) -> None:
        """Initialize output locations."""
        self.output_path = Path(output_path)
        self.figures_directory = Path(figures_directory)
        self.metrics_directory = Path(metrics_directory)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.figures_directory.mkdir(parents=True, exist_ok=True)
        self.metrics_directory.mkdir(parents=True, exist_ok=True)

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

    def save_final_evaluation(
        self,
        model: Any,
        X_test: Any,
        y_test: Any,
    ) -> None:
        """
        Generate reports and plots for the selected final model.

        The supplied model may be a complete scikit-learn pipeline.
        """
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        report = classification_report(
            y_test,
            predictions,
            target_names=["No Attrition", "Attrition"],
            output_dict=True,
            zero_division=0,
        )

        report_dataframe = pd.DataFrame(report).transpose()

        report_dataframe.to_csv(
            self.metrics_directory / "classification_report.csv"
        )

        self._plot_confusion_matrix(
            model=model,
            X_test=X_test,
            y_test=y_test,
        )

        self._plot_roc_curve(
            y_test=y_test,
            probabilities=probabilities,
        )

        self._plot_precision_recall_curve(
            y_test=y_test,
            probabilities=probabilities,
        )

    def _plot_confusion_matrix(
        self,
        model: Any,
        X_test: Any,
        y_test: Any,
    ) -> None:
        """Save the final model confusion matrix."""
        figure, axis = plt.subplots(figsize=(7, 6))

        ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            display_labels=["No Attrition", "Attrition"],
            cmap="Blues",
            values_format="d",
            ax=axis,
        )

        axis.set_title("Final Model Confusion Matrix")
        figure.tight_layout()

        figure.savefig(
            self.figures_directory / "confusion_matrix.png",
            dpi=300,
        )

        plt.close(figure)

    def _plot_roc_curve(
        self,
        y_test: Any,
        probabilities: Any,
    ) -> None:
        """Save the receiver operating characteristic curve."""
        figure, axis = plt.subplots(figsize=(7, 6))

        RocCurveDisplay.from_predictions(
            y_test,
            probabilities,
            name="Final Model",
            ax=axis,
        )

        axis.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random Classifier",
        )

        axis.set_title("Final Model ROC Curve")
        axis.legend()
        figure.tight_layout()

        figure.savefig(
            self.figures_directory / "roc_curve.png",
            dpi=300,
        )

        plt.close(figure)

    def _plot_precision_recall_curve(
        self,
        y_test: Any,
        probabilities: Any,
    ) -> None:
        """Save the precision–recall curve."""
        figure, axis = plt.subplots(figsize=(7, 6))

        PrecisionRecallDisplay.from_predictions(
            y_test,
            probabilities,
            name="Final Model",
            ax=axis,
        )

        axis.set_title("Final Model Precision–Recall Curve")
        figure.tight_layout()

        figure.savefig(
            self.figures_directory / "precision_recall_curve.png",
            dpi=300,
        )

        plt.close(figure)