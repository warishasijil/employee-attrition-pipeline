"""Data cleaning utilities for the employee attrition dataset."""

from pathlib import Path

import pandas as pd


class DataCleaner:
    """Clean the employee attrition dataset without applying model preprocessing."""

    def __init__(
        self,
        target_column: str,
        constant_columns: list[str],
        identifier_columns: list[str],
    ) -> None:
        """
        Initialize the data cleaner.

        Args:
            target_column: Name of the prediction target.
            constant_columns: Columns that contain no predictive information.
            identifier_columns: Identifier columns excluded from modelling.
        """
        self.target_column = target_column
        self.constant_columns = constant_columns
        self.identifier_columns = identifier_columns

    @staticmethod
    def remove_duplicates(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the dataset with duplicate rows removed."""
        return dataframe.drop_duplicates().reset_index(drop=True)

    def remove_constant_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Remove configured constant columns when present."""
        columns_to_drop = [
            column
            for column in self.constant_columns
            if column in dataframe.columns
        ]

        return dataframe.drop(columns=columns_to_drop)

    def remove_identifier_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Remove configured identifier columns when present."""
        columns_to_drop = [
            column
            for column in self.identifier_columns
            if column in dataframe.columns
        ]

        return dataframe.drop(columns=columns_to_drop)

    def encode_target(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the target values as binary integers.

        Yes becomes 1 and No becomes 0.
        """
        if self.target_column not in dataframe.columns:
            raise ValueError(
                f"Target column is missing: {self.target_column}"
            )

        cleaned_dataframe = dataframe.copy()

        target_mapping = {
            "No": 0,
            "Yes": 1,
        }

        unexpected_values = set(
            cleaned_dataframe[self.target_column].dropna().unique()
        ) - set(target_mapping)

        if unexpected_values:
            raise ValueError(
                f"Cannot encode unexpected target values: "
                f"{sorted(unexpected_values)}"
            )

        cleaned_dataframe[self.target_column] = (
            cleaned_dataframe[self.target_column]
            .map(target_mapping)
            .astype("int8")
        )

        return cleaned_dataframe

    def clean(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Run all data-cleaning stages in sequence."""
        cleaned_dataframe = dataframe.copy()

        cleaned_dataframe = self.remove_duplicates(cleaned_dataframe)
        cleaned_dataframe = self.remove_constant_columns(cleaned_dataframe)
        cleaned_dataframe = self.remove_identifier_columns(cleaned_dataframe)
        cleaned_dataframe = self.encode_target(cleaned_dataframe)

        return cleaned_dataframe

    @staticmethod
    def save(
        dataframe: pd.DataFrame,
        output_path: str | Path,
    ) -> None:
        """Save the cleaned dataset as a CSV file."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        dataframe.to_csv(destination, index=False)