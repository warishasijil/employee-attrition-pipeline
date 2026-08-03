"""Pipeline orchestration for the Employee Attrition project."""

from typing import Any

import pandas as pd

from src.data import DataLoader, DataSplitter, DataValidator
from src.features import FeatureEngineer
from src.models import ModelEvaluator, ModelTrainer
from src.preprocessing import DataCleaner, ModelPreprocessor
from src.utils import Config
from src.visualization import EDAAnalyzer


class EmployeeAttritionPipeline:
    """Coordinate data preparation and model-training stages."""

    def __init__(self) -> None:
        """Initialize pipeline components using project configuration."""
        self.target_column = Config.get("data", "target_column")
        self.random_state = Config.get("project", "random_state")

        self.data_loader = DataLoader(
            Config.get("paths", "raw_data")
        )

        self.data_validator = DataValidator(
            target_column=self.target_column,
            required_columns=Config.get(
                "data",
                "required_columns",
            ),
            expected_target_values=Config.get(
                "data",
                "expected_target_values",
            ),
            constant_columns=Config.get(
                "data",
                "constant_columns",
            ),
        )

        self.data_cleaner = DataCleaner(
            target_column=self.target_column,
            constant_columns=Config.get(
                "data",
                "constant_columns",
            ),
            identifier_columns=Config.get(
                "data",
                "identifier_columns",
            ),
        )

        self.feature_engineer = FeatureEngineer(
            target_column=self.target_column,
        )

        self.eda_analyzer = EDAAnalyzer(
            target_column=self.target_column,
            figures_directory=Config.get(
                "paths",
                "figures_directory",
            ),
            metrics_directory=Config.get(
                "paths",
                "metrics_directory",
            ),
        )

        self.data_splitter = DataSplitter(
            target_column=self.target_column,
            test_size=Config.get("data", "test_size"),
            random_state=self.random_state,
        )

        self.preprocessor = ModelPreprocessor()

        self.model_trainer = ModelTrainer(
            random_state=self.random_state
        )

        self.model_evaluator = ModelEvaluator(
            Config.get("paths", "metrics_output")
        )

    def run_data_pipeline(self) -> pd.DataFrame:
        """
        Run loading, validation, cleaning, feature engineering, saving,
        and exploratory analysis.

        Returns:
            Processed employee attrition dataset.
        """
        raw_dataframe = self.data_loader.load()

        validation_report = self.data_validator.validate(
            raw_dataframe
        )

        cleaned_dataframe = self.data_cleaner.clean(
            raw_dataframe
        )

        processed_dataframe = self.feature_engineer.transform(
            cleaned_dataframe
        )

        self.data_cleaner.save(
            processed_dataframe,
            Config.get("paths", "processed_data"),
        )

        eda_summary = self.eda_analyzer.run(
            processed_dataframe
        )

        self._display_data_pipeline_summary(
            raw_dataframe=raw_dataframe,
            processed_dataframe=processed_dataframe,
            validation_report=validation_report,
            eda_summary=eda_summary,
        )

        return processed_dataframe

    def run_training_pipeline(self) -> pd.DataFrame:
        """
        Run the complete data and baseline model-training pipeline.

        Returns:
            Model comparison table.
        """
        processed_dataframe = self.run_data_pipeline()

        data_split = self.data_splitter.split(
            processed_dataframe
        )

        X_train_transformed = self.preprocessor.fit_transform(
            data_split.X_train
        )

        X_test_transformed = self.preprocessor.transform(
            data_split.X_test
        )

        trained_models = self.model_trainer.train(
            X_train_transformed,
            data_split.y_train,
        )

        comparison = self.model_evaluator.compare(
            trained_models,
            X_test_transformed,
            data_split.y_test,
        )

        self._display_training_pipeline_summary(
            comparison=comparison,
            training_rows=data_split.X_train.shape[0],
            testing_rows=data_split.X_test.shape[0],
            transformed_feature_count=(
                X_train_transformed.shape[1]
            ),
        )

        return comparison

    @staticmethod
    def _display_data_pipeline_summary(
        raw_dataframe: pd.DataFrame,
        processed_dataframe: pd.DataFrame,
        validation_report: dict[str, Any],
        eda_summary: dict[str, Any],
    ) -> None:
        """Display a concise summary of the data pipeline."""
        removed_columns = sorted(
            set(raw_dataframe.columns)
            - set(processed_dataframe.columns)
        )

        added_columns = sorted(
            set(processed_dataframe.columns)
            - set(raw_dataframe.columns)
        )

        print("=" * 60)
        print("DATA PIPELINE COMPLETED")
        print("=" * 60)
        print(
            f"Validation passed: "
            f"{validation_report['validation_passed']}"
        )
        print(f"Raw shape: {raw_dataframe.shape}")
        print(f"Processed shape: {processed_dataframe.shape}")
        print(f"Removed columns: {removed_columns}")
        print(f"Engineered columns: {added_columns}")
        print(
            f"Duplicate rows: "
            f"{validation_report['duplicate_rows']}"
        )
        print(
            f"Missing values: "
            f"{validation_report['missing_values_total']}"
        )
        print(
            f"Numerical columns: "
            f"{len(eda_summary['numerical_columns'])}"
        )
        print(
            f"Categorical columns: "
            f"{len(eda_summary['categorical_columns'])}"
        )
        print(
            "Processed data saved to:",
            Config.get("paths", "processed_data"),
        )

    @staticmethod
    def _display_training_pipeline_summary(
        comparison: pd.DataFrame,
        training_rows: int,
        testing_rows: int,
        transformed_feature_count: int,
    ) -> None:
        """Display baseline model-training results."""
        print()
        print("=" * 60)
        print("TRAINING PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Training rows: {training_rows}")
        print(f"Testing rows: {testing_rows}")
        print(
            f"Transformed feature count: "
            f"{transformed_feature_count}"
        )
        print()
        print("BASELINE MODEL COMPARISON")
        print(comparison.round(4).to_string(index=False))