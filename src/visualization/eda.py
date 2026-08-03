"""Exploratory data analysis utilities."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class EDAAnalyzer:
    """Generate and save exploratory data analysis outputs."""

    def __init__(
        self,
        target_column: str,
        figures_directory: str | Path,
        metrics_directory: str | Path,
    ) -> None:
        """
        Initialize the EDA analyzer.

        Args:
            target_column: Name of the prediction target.
            figures_directory: Directory used to save plots.
            metrics_directory: Directory used to save summary tables.
        """
        self.target_column = target_column
        self.figures_directory = Path(figures_directory)
        self.metrics_directory = Path(metrics_directory)

        self.figures_directory.mkdir(parents=True, exist_ok=True)
        self.metrics_directory.mkdir(parents=True, exist_ok=True)

    def generate_summary(self, dataframe: pd.DataFrame) -> dict[str, Any]:
        """Generate and save numerical and categorical summaries."""
        numerical_summary = dataframe.describe().transpose()
        categorical_summary = dataframe.describe(
            include=["object", "category"]
        ).transpose()

        numerical_summary.to_csv(
            self.metrics_directory / "numerical_summary.csv"
        )

        categorical_summary.to_csv(
            self.metrics_directory / "categorical_summary.csv"
        )

        data_types = dataframe.dtypes.astype(str).rename("data_type")
        data_types.to_csv(
            self.metrics_directory / "column_data_types.csv"
        )

        return {
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
            "numerical_columns": dataframe.select_dtypes(
                include="number"
            ).columns.tolist(),
            "categorical_columns": dataframe.select_dtypes(
                exclude="number"
            ).columns.tolist(),
        }

    def plot_target_distribution(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Plot and save the attrition target distribution."""
        plt.figure(figsize=(7, 5))

        sns.countplot(
            data=dataframe,
            x=self.target_column,
        )

        plt.title("Employee Attrition Distribution")
        plt.xlabel("Attrition")
        plt.ylabel("Employee Count")
        plt.tight_layout()

        plt.savefig(
            self.figures_directory / "target_distribution.png",
            dpi=300,
        )

        plt.close()

    def plot_numeric_distributions(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Plot selected numerical feature distributions."""
        selected_columns = [
            "Age",
            "MonthlyIncome",
            "TotalWorkingYears",
            "YearsAtCompany",
            "DistanceFromHome",
        ]

        available_columns = [
            column
            for column in selected_columns
            if column in dataframe.columns
        ]

        for column in available_columns:
            plt.figure(figsize=(7, 5))

            sns.histplot(
                data=dataframe,
                x=column,
                hue=self.target_column,
                kde=True,
                multiple="layer",
            )

            plt.title(f"{column} Distribution by Attrition")
            plt.tight_layout()

            plt.savefig(
                self.figures_directory
                / f"{column.lower()}_distribution.png",
                dpi=300,
            )

            plt.close()

    def plot_categorical_relationships(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Plot selected categorical features against attrition."""
        selected_columns = [
            "OverTime",
            "JobRole",
            "Department",
            "BusinessTravel",
            "MaritalStatus",
        ]

        available_columns = [
            column
            for column in selected_columns
            if column in dataframe.columns
        ]

        for column in available_columns:
            plt.figure(figsize=(10, 6))

            sns.countplot(
                data=dataframe,
                x=column,
                hue=self.target_column,
            )

            plt.title(f"{column} by Attrition")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            plt.savefig(
                self.figures_directory
                / f"{column.lower()}_vs_attrition.png",
                dpi=300,
            )

            plt.close()

    def plot_correlation_heatmap(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Plot a correlation heatmap for numerical variables."""
        numerical_dataframe = dataframe.select_dtypes(
            include="number"
        )

        correlation_matrix = numerical_dataframe.corr()

        plt.figure(figsize=(16, 12))

        sns.heatmap(
            correlation_matrix,
            cmap="coolwarm",
            center=0,
            square=False,
            linewidths=0.2,
        )

        plt.title("Numerical Feature Correlation Heatmap")
        plt.tight_layout()

        plt.savefig(
            self.figures_directory / "correlation_heatmap.png",
            dpi=300,
        )

        plt.close()

    def run(self, dataframe: pd.DataFrame) -> dict[str, Any]:
        """Run the complete exploratory analysis stage."""
        summary = self.generate_summary(dataframe)
        self.plot_target_distribution(dataframe)
        self.plot_numeric_distributions(dataframe)
        self.plot_categorical_relationships(dataframe)
        self.plot_correlation_heatmap(dataframe)

        return summary