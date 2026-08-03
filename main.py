"""Entry point for the Employee Attrition Pipeline."""

from src.pipeline import EmployeeAttritionPipeline


def main() -> None:
    """Run the complete baseline training pipeline."""
    pipeline = EmployeeAttritionPipeline()
    pipeline.run_training_pipeline()


if __name__ == "__main__":
    main()