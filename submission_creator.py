# This script evaluate the format of your submission and add flops

from argparse import ArgumentParser

import pandas as pd


def main(predictions_file: str, flops: int):
    # Load predictions
    predictions = pd.read_csv(predictions_file)
    predictions["affected"] = predictions["affected"].astype(int)
    # Check format
    assert all(
        col in predictions.columns for col in ["key", "magnitude", "affected"]
    ), "Missing columns in predictions file"
    # Check values
    assert predictions["magnitude"].dtype == float, "Magnitude should be a float"
    assert predictions["affected"].dtype == int, "Affected should be an int"
    # Add flops
    predictions["flops"] = flops
    # Save
    predictions.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--flops", type=int, required=True)
    args = parser.parse_args()
    main(args.predictions, args.flops)
