"""Entry point for the Employee Attrition Pipeline."""

from src.pipeline import EmployeeAttritionPipeline


def main() -> None:
    """Run the employee attrition data pipeline."""
    pipeline = EmployeeAttritionPipeline()
    pipeline.run_data_pipeline()


if __name__ == "__main__":
    main()