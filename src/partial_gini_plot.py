import argparse
import pandas as pd
from analysis.evaluator import ModelEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a partial Gini plot")
    parser.add_argument("csv", help="CSV file with prediction results")
    parser.add_argument("--actual", required=True, help="Column containing actual values")
    parser.add_argument("--predicted", required=True, help="Column containing predicted values")
    parser.add_argument("--output", default="partial_gini.png", help="Output image path")
    parser.add_argument("--top_percent", type=int, default=20, help="Top percent for partial Gini")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    evaluator = ModelEvaluator(df, actual_col=args.actual, predicted_col=args.predicted)
    fig = evaluator.plot_partial_gini(top_percent=args.top_percent)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Partial Gini plot saved to {args.output}")


if __name__ == "__main__":
    main()
