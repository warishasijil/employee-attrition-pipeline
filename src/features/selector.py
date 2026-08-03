"""Feature importance and selection utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline


class FeatureSelector:
    """Analyze model coefficients and identify important features."""

    def __init__(
        self,
        output_csv: str | Path,
        output_figure: str | Path,
        top_n: int = 20,
    ) -> None:
        """Initialize feature-importance output settings."""
        self.output_csv = Path(output_csv)
        self.output_figure = Path(output_figure)
        self.top_n = top_n

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.output_figure.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_importance(
        model_pipeline: Pipeline,
    ) -> pd.DataFrame:
        """
        Extract Logistic Regression coefficients and feature names.

        Returns:
            DataFrame ranked by absolute coefficient magnitude.
        """
        preprocessor = model_pipeline.named_steps["preprocessor"]
        model = model_pipeline.named_steps["model"]

        if not hasattr(model, "coef_"):
            raise TypeError(
                "The selected model does not expose coefficient-based "
                "feature importance."
            )

        feature_names = preprocessor.get_feature_names_out()
        coefficients = model.coef_[0]

        if len(feature_names) != len(coefficients):
            raise ValueError(
                "Feature-name and coefficient counts do not match."
            )

        importance = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "absolute_importance": abs(coefficients),
                "direction": [
                    "increases_attrition_risk"
                    if coefficient > 0
                    else "decreases_attrition_risk"
                    for coefficient in coefficients
                ],
            }
        )

        return importance.sort_values(
            by="absolute_importance",
            ascending=False,
        ).reset_index(drop=True)

    def save_outputs(
        self,
        importance: pd.DataFrame,
    ) -> None:
        """Save feature importance as CSV and a top-feature chart."""
        importance.to_csv(self.output_csv, index=False)

        top_features = importance.head(self.top_n).sort_values(
            by="absolute_importance",
            ascending=True,
        )

        plt.figure(figsize=(10, 8))
        plt.barh(
            top_features["feature"],
            top_features["absolute_importance"],
        )
        plt.xlabel("Absolute Logistic Regression Coefficient")
        plt.ylabel("Feature")
        plt.title(f"Top {self.top_n} Attrition Prediction Features")
        plt.tight_layout()
        plt.savefig(self.output_figure, dpi=300)
        plt.close()

    def run(
        self,
        model_pipeline: Pipeline,
    ) -> pd.DataFrame:
        """Extract and save feature-importance results."""
        importance = self.extract_importance(model_pipeline)
        self.save_outputs(importance)
        return importance