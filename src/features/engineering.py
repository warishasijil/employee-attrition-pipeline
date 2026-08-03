"""Feature engineering for the employee attrition dataset."""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Create reusable employee-related features."""

    def __init__(self, target_column: str = "Attrition") -> None:
        """
        Initialize the feature engineer.

        Args:
            target_column: Name of the target variable.
        """
        self.target_column = target_column

    @staticmethod
    def _safe_divide(
        numerator: pd.Series,
        denominator: pd.Series,
    ) -> pd.Series:
        """
        Safely divide two pandas Series.

        Zero denominators are replaced with NaN before division.
        Infinite and missing results are replaced with zero.
        """
        result = numerator / denominator.replace(0, np.nan)

        return (
            result.replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    def add_experience_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create career and organizational experience ratios."""
        engineered_dataframe = dataframe.copy()

        engineered_dataframe["CompanyTenureRatio"] = self._safe_divide(
            engineered_dataframe["YearsAtCompany"],
            engineered_dataframe["TotalWorkingYears"],
        )

        engineered_dataframe["CurrentRoleTenureRatio"] = self._safe_divide(
            engineered_dataframe["YearsInCurrentRole"],
            engineered_dataframe["YearsAtCompany"],
        )

        engineered_dataframe["ManagerTenureRatio"] = self._safe_divide(
            engineered_dataframe["YearsWithCurrManager"],
            engineered_dataframe["YearsAtCompany"],
        )

        engineered_dataframe["PromotionDelayRatio"] = self._safe_divide(
            engineered_dataframe["YearsSinceLastPromotion"],
            engineered_dataframe["YearsAtCompany"],
        )

        return engineered_dataframe

    @staticmethod
    def add_compensation_features(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create compensation-related features."""
        engineered_dataframe = dataframe.copy()

        engineered_dataframe["IncomePerWorkingYear"] = (
            engineered_dataframe["MonthlyIncome"]
            / engineered_dataframe["TotalWorkingYears"].replace(0, 1)
        )

        engineered_dataframe["IncomePerJobLevel"] = (
            engineered_dataframe["MonthlyIncome"]
            / engineered_dataframe["JobLevel"].replace(0, 1)
        )

        return engineered_dataframe

    @staticmethod
    def add_satisfaction_features(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create aggregate employee satisfaction features."""
        engineered_dataframe = dataframe.copy()

        satisfaction_columns = [
            "EnvironmentSatisfaction",
            "JobSatisfaction",
            "RelationshipSatisfaction",
            "WorkLifeBalance",
        ]

        engineered_dataframe["OverallSatisfaction"] = (
            engineered_dataframe[satisfaction_columns].mean(axis=1)
        )

        engineered_dataframe["LowSatisfactionCount"] = (
            engineered_dataframe[satisfaction_columns]
            .le(2)
            .sum(axis=1)
        )

        return engineered_dataframe

    @staticmethod
    def add_career_stage_feature(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create an ordinal career-stage feature from total experience."""
        engineered_dataframe = dataframe.copy()

        engineered_dataframe["CareerStage"] = pd.cut(
            engineered_dataframe["TotalWorkingYears"],
            bins=[-1, 2, 7, 15, float("inf")],
            labels=[
                "Early Career",
                "Developing",
                "Experienced",
                "Senior",
            ],
        ).astype(str)

        return engineered_dataframe

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply all feature-engineering transformations.

        Returns:
            DataFrame containing original and engineered features.
        """
        engineered_dataframe = dataframe.copy()

        engineered_dataframe = self.add_experience_features(
            engineered_dataframe
        )

        engineered_dataframe = self.add_compensation_features(
            engineered_dataframe
        )

        engineered_dataframe = self.add_satisfaction_features(
            engineered_dataframe
        )

        engineered_dataframe = self.add_career_stage_feature(
            engineered_dataframe
        )

        return engineered_dataframe