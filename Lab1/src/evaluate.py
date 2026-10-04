"""
Reads a CSV of model predictions and prints a metrics report.

Usage:
    python -m src.evaluate                       # uses data/predictions.csv
    python -m src.evaluate path/to/file.csv
"""

import csv
import os
import sys

from src import metrics

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "predictions.csv")


def load_predictions(path):
    """
    Loads a CSV with columns y_true and y_pred.
    Returns:
        tuple: (list of true labels, list of predicted labels)
    """
    y_true, y_pred = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            y_true.append(int(row["y_true"]))
            y_pred.append(int(row["y_pred"]))
    return y_true, y_pred


def evaluate(y_true, y_pred):
    """
    Computes every classification metric.
    Returns:
        dict: metric name -> value
    """
    return {
        "accuracy": metrics.accuracy(y_true, y_pred),
        "precision": metrics.precision(y_true, y_pred),
        "recall": metrics.recall(y_true, y_pred),
        "f1_score": metrics.f1_score(y_true, y_pred),
        **metrics.confusion_matrix(y_true, y_pred),
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    y_true, y_pred = load_predictions(path)
    print(f"Evaluated {len(y_true)} predictions from {os.path.basename(path)}")
    for name, value in evaluate(y_true, y_pred).items():
        print(f"  {name:<10} {value:.3f}" if isinstance(value, float) else f"  {name:<10} {value}")


if __name__ == "__main__":
    main()
