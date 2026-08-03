"""Pipeline orchestration for the Employee Attrition project."""

from typing import Any

import pandas as pd

from src.data import DataLoader, DataSplitter, DataValidator
from src.features import FeatureEngineer, FeatureSelector
from src.models import (
    ModelEvaluator,
    ModelManager,
    ModelTrainer,
    ModelTuner,
)
from src.preprocessing import DataCleaner, ModelPreprocessor
from src.utils import Config
from src.visualization import EDAAnalyzer


class EmployeeAttritionPipeline:
    """Coordinate data preparation, training, tuning, and evaluation."""

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

        self.feature_selector = FeatureSelector(
            output_csv=Config.get(
                "paths",
                "feature_importance_output",
            ),
            output_figure=Config.get(
                "paths",
                "feature_importance_figure",
            ),
            top_n=20,
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
            test_size=Config.get(
                "data",
                "test_size",
            ),
            validation_size=Config.get(
                "data",
                "validation_size",
            ),
            random_state=self.random_state,
        )

        self.preprocessor = ModelPreprocessor()

        self.model_trainer = ModelTrainer(
            random_state=self.random_state
        )

        self.model_tuner = ModelTuner(
            random_state=self.random_state,
            cv_folds=Config.get(
                "training",
                "cross_validation_folds",
            ),
            scoring=Config.get(
                "training",
                "scoring_metric",
            ),
            n_jobs=Config.get(
                "training",
                "n_jobs",
            ),
        )

        self.model_evaluator = ModelEvaluator(
            output_path=Config.get(
                "paths",
                "metrics_output",
            ),
            figures_directory=Config.get(
                "paths",
                "figures_directory",
            ),
            metrics_directory=Config.get(
                "paths",
                "metrics_directory",
            ),
        )

        self.model_manager = ModelManager()

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
        Run the complete model-training pipeline.

        Workflow:
        - process the raw dataset;
        - create training, validation, and test sets;
        - fit preprocessing using training data only;
        - train baseline models;
        - tune selected models using training cross-validation;
        - compare all models on the validation set;
        - select the best validation model;
        - evaluate the selected model once on the test set;
        - save the complete inference pipeline;
        - generate feature-importance and evaluation artifacts.

        Returns:
            Validation-set model-comparison table.
        """
        processed_dataframe = self.run_data_pipeline()

        data_split = self.data_splitter.split(
            processed_dataframe
        )

        X_train_transformed = self.preprocessor.fit_transform(
            data_split.X_train
        )

        X_validation_transformed = self.preprocessor.transform(
            data_split.X_validation
        )

        baseline_models = self.model_trainer.train(
            X_train_transformed,
            data_split.y_train,
        )

        baseline_comparison = self.model_evaluator.compare(
            baseline_models,
            X_validation_transformed,
            data_split.y_validation,
        )

        tuned_models, tuning_results = self.model_tuner.tune(
            X_train_transformed,
            data_split.y_train,
        )

        tuned_comparison = self.model_evaluator.compare(
            tuned_models,
            X_validation_transformed,
            data_split.y_validation,
        )

        tuned_comparison["model"] = (
            "tuned_" + tuned_comparison["model"]
        )

        combined_comparison = pd.concat(
            [
                baseline_comparison,
                tuned_comparison,
            ],
            ignore_index=True,
        ).sort_values(
            by=Config.get(
                "training",
                "scoring_metric",
            ),
            ascending=False,
        ).reset_index(drop=True)

        combined_comparison.to_csv(
            Config.get("paths", "metrics_output"),
            index=False,
        )

        all_models = {
            **baseline_models,
            **{
                f"tuned_{model_name}": model
                for model_name, model in tuned_models.items()
            },
        }

        best_model_name = (
            self.model_manager.select_best_model_name(
                comparison=combined_comparison,
                metric=Config.get(
                    "training",
                    "scoring_metric",
                ),
            )
        )

        best_model = all_models[best_model_name]

        if self.preprocessor.transformer is None:
            raise RuntimeError(
                "The preprocessing transformer was not fitted."
            )

        final_model_pipeline = self.model_manager.build_pipeline(
            preprocessor=self.preprocessor.transformer,
            model=best_model,
        )

        model_output_path = Config.get(
            "paths",
            "model_output",
        )

        self.model_manager.save(
            final_model_pipeline,
            model_output_path,
        )

        final_test_metrics = (
            self.model_evaluator.evaluate_model(
                model=final_model_pipeline,
                X_test=data_split.X_test,
                y_test=data_split.y_test,
            )
        )

        self.model_evaluator.save_final_evaluation(
            model=final_model_pipeline,
            X_test=data_split.X_test,
            y_test=data_split.y_test,
        )

        feature_importance = self.feature_selector.run(
            final_model_pipeline
        )

        self._display_training_pipeline_summary(
            comparison=combined_comparison,
            training_rows=data_split.X_train.shape[0],
            validation_rows=(
                data_split.X_validation.shape[0]
            ),
            testing_rows=data_split.X_test.shape[0],
            transformed_feature_count=(
                X_train_transformed.shape[1]
            ),
            tuning_results=tuning_results,
            best_model_name=best_model_name,
            model_output_path=model_output_path,
            final_test_metrics=final_test_metrics,
            feature_importance=feature_importance,
        )

        return combined_comparison

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
        validation_rows: int,
        testing_rows: int,
        transformed_feature_count: int,
        tuning_results: dict[str, dict[str, Any]],
        best_model_name: str,
        model_output_path: str,
        final_test_metrics: dict[str, float],
        feature_importance: pd.DataFrame,
    ) -> None:
        """Display training, validation, test, and artifact results."""
        print()
        print("=" * 60)
        print("TRAINING PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Training rows: {training_rows}")
        print(f"Validation rows: {validation_rows}")
        print(f"Testing rows: {testing_rows}")
        print(
            f"Transformed feature count: "
            f"{transformed_feature_count}"
        )

        print()
        print("VALIDATION MODEL COMPARISON")
        print(comparison.round(4).to_string(index=False))

        print()
        print("BEST HYPERPARAMETERS")

        for model_name, result in tuning_results.items():
            print(f"{model_name}:")
            print(
                f"  CV ROC-AUC: "
                f"{result['best_score']:.4f}"
            )
            print(
                f"  Parameters: "
                f"{result['best_params']}"
            )

        print()
        print("FINAL MODEL SELECTION")
        print(f"Selected model: {best_model_name}")
        print(
            "Selection basis: highest validation "
            f"{Config.get('training', 'scoring_metric')}"
        )
        print(f"Model saved to: {model_output_path}")

        print()
        print("FINAL TEST-SET PERFORMANCE")
        print(
            f"Accuracy: "
            f"{final_test_metrics['accuracy']:.4f}"
        )
        print(
            f"Precision: "
            f"{final_test_metrics['precision']:.4f}"
        )
        print(
            f"Recall: "
            f"{final_test_metrics['recall']:.4f}"
        )
        print(
            f"F1-score: "
            f"{final_test_metrics['f1_score']:.4f}"
        )
        print(
            f"ROC-AUC: "
            f"{final_test_metrics['roc_auc']:.4f}"
        )
        print(
            f"PR-AUC: "
            f"{final_test_metrics['pr_auc']:.4f}"
        )

        print()
        print("TOP 10 IMPORTANT FEATURES")
        print(
            feature_importance[
                [
                    "feature",
                    "coefficient",
                    "direction",
                ]
            ]
            .head(10)
            .round(4)
            .to_string(index=False)
        )

        print()
        print("FEATURE IMPORTANCE ARTIFACTS")
        print(
            "Feature importance CSV:",
            Config.get(
                "paths",
                "feature_importance_output",
            ),
        )
        print(
            "Feature importance figure:",
            Config.get(
                "paths",
                "feature_importance_figure",
            ),
        )

        print()
        print("FINAL EVALUATION ARTIFACTS")
        print(
            "Confusion matrix:",
            "reports/figures/confusion_matrix.png",
        )
        print(
            "ROC curve:",
            "reports/figures/roc_curve.png",
        )
        print(
            "Precision–Recall curve:",
            "reports/figures/precision_recall_curve.png",
        )
        print(
            "Classification report:",
            "reports/metrics/classification_report.csv",
        )