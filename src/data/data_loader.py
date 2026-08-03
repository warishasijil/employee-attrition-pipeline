"""Data loading utilities for the employee attrition dataset."""

from pathlib import Path
from typing import Any

import pandas as pd


class DataLoader:
    """Load a CSV dataset and return basic metadata."""

    def __init__(self, file_path: str | Path) -> None:
        """
        Initialize the data loader.

        Args:
            file_path: Path to the CSV dataset.
        """
        self.file_path = Path(file_path)

    def _validate_file(self) -> None:
        """
        Validate that the dataset file exists and is a CSV file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not `.csv`.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.file_path.resolve()}"
            )

        if self.file_path.suffix.lower() != ".csv":
            raise ValueError(
                f"Expected a CSV file, but received: {self.file_path.suffix}"
            )

    def load(self) -> pd.DataFrame:
        """
        Load the CSV dataset.

        Returns:
            A pandas DataFrame containing the dataset.

        Raises:
            ValueError: If the loaded dataset is empty.
        """
        self._validate_file()

        dataframe = pd.read_csv(self.file_path)

        if dataframe.empty:
            raise ValueError(
                f"The dataset is empty: {self.file_path.resolve()}"
            )

        return dataframe

    @staticmethod
    def get_metadata(dataframe: pd.DataFrame) -> dict[str, Any]:
        """
        Generate basic dataset metadata.

        Args:
            dataframe: Loaded dataset.

        Returns:
            Dictionary containing dataset metadata.
        """
        return {
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
            "duplicate_rows": int(dataframe.duplicated().sum()),
            "missing_values": int(dataframe.isna().sum().sum()),
            "memory_usage_mb": round(
                dataframe.memory_usage(deep=True).sum() / (1024**2),
                4,
            ),
            "column_names": dataframe.columns.tolist(),
        }