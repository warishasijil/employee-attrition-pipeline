"""Command-line entry point for the Employee Attrition Pipeline."""

import argparse
from pathlib import Path

from src.inference import AttritionPredictor
from src.pipeline import EmployeeAttritionPipeline


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the employee attrition model or generate "
            "predictions from employee data."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "train",
        help="Run the complete training pipeline.",
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Generate attrition predictions from a CSV file.",
    )

    predict_parser.add_argument(
        "--input",
        required=True,
        help="Path to the input CSV containing employee records.",
    )

    predict_parser.add_argument(
        "--output",
        default="artifacts/predictions.csv",
        help=(
            "Path where predictions will be saved. "
            "Default: artifacts/predictions.csv"
        ),
    )

    return parser


def run_training() -> None:
    """Run the complete model-training pipeline."""
    pipeline = EmployeeAttritionPipeline()
    pipeline.run_training_pipeline()


def run_prediction(
    input_path: str,
    output_path: str,
) -> None:
    """Generate and save predictions from a CSV file."""
    predictor = AttritionPredictor()

    results = predictor.predict_from_csv(
        input_path=input_path,
        output_path=output_path,
    )

    print("=" * 60)
    print("INFERENCE COMPLETED")
    print("=" * 60)
    print(f"Input file: {Path(input_path)}")
    print(f"Records processed: {len(results)}")
    print(f"Predictions saved to: {Path(output_path)}")
    print()
    print(
        results[
            [
                "predicted_label",
                "attrition_probability",
            ]
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )


def main() -> None:
    """Parse CLI arguments and run the selected operation."""
    parser = build_parser()
    arguments = parser.parse_args()

    if arguments.command == "train":
        run_training()
    elif arguments.command == "predict":
        run_prediction(
            input_path=arguments.input,
            output_path=arguments.output,
        )


if __name__ == "__main__":
    main()