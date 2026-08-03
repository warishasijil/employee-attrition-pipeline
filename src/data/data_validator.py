"""Validation utilities for the employee attrition dataset."""

from typing import Any

import pandas as pd


class DataValidator:
    """Validate dataset structure and business expectations."""

    def __init__(
        self,
        target_column: str,
        required_columns: list[str],
        expected_target_values: list[str],
        constant_columns: list[str],
    ) -> None:
        """
        Initialize the validator.

        Args:
            target_column: Name of the prediction target.
            required_columns: Columns expected to exist in the dataset.
            expected_target_values: Allowed values in the target column.
            constant_columns: Columns expected to contain one unique value.
        """
        self.target_column = target_column
        self.required_columns = required_columns
        self.expected_target_values = set(expected_target_values)
        self.constant_columns = constant_columns

    def validate_required_columns(self, dataframe: pd.DataFrame) -> None:
        """
        Confirm that all required columns exist.

        Raises:
            ValueError: If required columns are missing.
        """
        missing_columns = [
            column
            for column in self.required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Required columns are missing: {missing_columns}"
            )

    def validate_target(self, dataframe: pd.DataFrame) -> None:
        """
        Validate the target column and its values.

        Raises:
            ValueError: If the target is missing, empty, or contains unexpected values.
        """
        if self.target_column not in dataframe.columns:
            raise ValueError(
                f"Target column is missing: {self.target_column}"
            )

        if dataframe[self.target_column].isna().any():
            raise ValueError(
                f"Target column contains missing values: {self.target_column}"
            )

        observed_values = set(
            dataframe[self.target_column].astype(str).unique()
        )

        unexpected_values = observed_values - self.expected_target_values

        if unexpected_values:
            raise ValueError(
                f"Unexpected target values found: {sorted(unexpected_values)}"
            )

    def validate_constant_columns(self, dataframe: pd.DataFrame) -> dict[str, bool]:
        """
        Check whether configured constant columns are actually constant.

        Returns:
            Mapping of column names to validation results.
        """
        results: dict[str, bool] = {}

        for column in self.constant_columns:
            if column not in dataframe.columns:
                results[column] = False
                continue

            results[column] = dataframe[column].nunique(dropna=False) == 1

        return results

    def validate(self, dataframe: pd.DataFrame) -> dict[str, Any]:
        """
        Run all validation checks and return a validation report.

        Returns:
            Dictionary containing validation results.
        """
        if dataframe.empty:
            raise ValueError("Dataset is empty.")

        self.validate_required_columns(dataframe)
        self.validate_target(dataframe)

        missing_by_column = dataframe.isna().sum()
        missing_by_column = missing_by_column[
            missing_by_column > 0
        ].to_dict()

        validation_report = {
            "is_empty": dataframe.empty,
            "row_count": int(dataframe.shape[0]),
            "column_count": int(dataframe.shape[1]),
            "duplicate_rows": int(dataframe.duplicated().sum()),
            "missing_values_total": int(dataframe.isna().sum().sum()),
            "missing_values_by_column": missing_by_column,
            "target_distribution": (
                dataframe[self.target_column]
                .value_counts()
                .to_dict()
            ),
            "constant_column_checks": self.validate_constant_columns(
                dataframe
            ),
            "validation_passed": True,
        }

        return validation_report