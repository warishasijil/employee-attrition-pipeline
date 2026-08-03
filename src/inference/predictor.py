"""Prediction utilities for employee attrition inference."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.features import FeatureEngineer
from src.models import ModelManager
from src.preprocessing import DataCleaner
from src.utils import Config


class AttritionPredictor:
    """Load the saved pipeline and generate attrition predictions."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        """
        Initialize the predictor.

        Args:
            model_path: Optional path to the saved model pipeline.
        """
        self.model_path = Path(
            model_path or Config.get("paths", "model_output")
        )

        self.model_pipeline = ModelManager.load(self.model_path)

        self.data_cleaner = DataCleaner(
            target_column=Config.get("data", "target_column"),
            constant_columns=Config.get("data", "constant_columns"),
            identifier_columns=Config.get("data", "identifier_columns"),
        )

        self.feature_engineer = FeatureEngineer(
            target_column=Config.get("data", "target_column"),
        )

    def prepare_input(
        self,
        input_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare raw employee records for prediction.

        The target column is optional during inference.
        """
        prepared_data = input_data.copy()

        target_column = Config.get("data", "target_column")

        if target_column in prepared_data.columns:
            prepared_data = prepared_data.drop(
                columns=[target_column]
            )

        prepared_data = self.data_cleaner.remove_constant_columns(
            prepared_data
        )

        prepared_data = self.data_cleaner.remove_identifier_columns(
            prepared_data
        )

        prepared_data = self.feature_engineer.transform(
            prepared_data
        )

        return prepared_data

    def predict(
        self,
        input_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Predict attrition class and probability.

        Returns:
            DataFrame with predicted class, label, and probability.
        """
        prepared_data = self.prepare_input(input_data)

        predictions = self.model_pipeline.predict(prepared_data)
        probabilities = self.model_pipeline.predict_proba(
            prepared_data
        )[:, 1]

        results = pd.DataFrame(
            {
                "predicted_attrition": predictions.astype(int),
                "predicted_label": [
                    "Yes" if prediction == 1 else "No"
                    for prediction in predictions
                ],
                "attrition_probability": probabilities,
            }
        )

        return results

    def predict_from_csv(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Load employee records from CSV and optionally save predictions."""
        source = Path(input_path)

        if not source.exists():
            raise FileNotFoundError(
                f"Inference input file not found: {source.resolve()}"
            )

        input_data = pd.read_csv(source)
        predictions = self.predict(input_data)

        combined_results = pd.concat(
            [
                input_data.reset_index(drop=True),
                predictions,
            ],
            axis=1,
        )

        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            combined_results.to_csv(destination, index=False)

        return combined_results