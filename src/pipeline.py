"""Pipeline orchestration for the Employee Attrition project."""

from typing import Any

import pandas as pd

from src.data import DataLoader, DataValidator
from src.features import FeatureEngineer
from src.preprocessing import DataCleaner
from src.utils import Config
from src.visualization import EDAAnalyzer


class EmployeeAttritionPipeline:
    """Coordinate the data stages of the employee attrition pipeline."""

    def __init__(self) -> None:
        """Initialize pipeline components using project configuration."""
        self.target_column = Config.get("data", "target_column")

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

    def run_data_pipeline(self) -> pd.DataFrame:
        """
        Run data loading, validation, cleaning, feature engineering,
        saving, and exploratory data analysis.

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

        engineered_dataframe = self.feature_engineer.transform(
            cleaned_dataframe
        )

        self.data_cleaner.save(
            engineered_dataframe,
            Config.get("paths", "processed_data"),
        )

        eda_summary = self.eda_analyzer.run(
            engineered_dataframe
        )

        self._display_data_pipeline_summary(
            raw_dataframe=raw_dataframe,
            processed_dataframe=engineered_dataframe,
            validation_report=validation_report,
            eda_summary=eda_summary,
        )

        return engineered_dataframe

    @staticmethod
    def _display_data_pipeline_summary(
        raw_dataframe: pd.DataFrame,
        processed_dataframe: pd.DataFrame,
        validation_report: dict[str, Any],
        eda_summary: dict[str, Any],
    ) -> None:
        """Display a concise summary of the completed data pipeline."""
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
        print(
            "Figures saved to:",
            Config.get("paths", "figures_directory"),
        )
        print(
            "Summary tables saved to:",
            Config.get("paths", "metrics_directory"),
        )