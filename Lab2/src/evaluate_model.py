"""
Scores a trained model on the held-out test split and applies a quality gate.

Exits with code 1 if the F1 score is below --min-f1, which fails the
GitHub Actions job and stops the model from being promoted.

Usage:
    python src/evaluate_model.py --version 20261018120000-abc1234 --min-f1 0.90
"""

import argparse
import json
import os
import sys

import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import load_split


def evaluate(model):
    """
    Returns:
        dict: accuracy, precision, recall, f1 and roc_auc, rounded to 4 places
    """
    _, X_test, _, y_test = load_split()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return {name: round(float(value), 4) for name, value in metrics.items()}


def passes_gate(metrics, min_f1):
    return metrics["f1"] >= min_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--min-f1", type=float, default=0.90)
    parser.add_argument("--build-dir", default="build")
    args = parser.parse_args()

    model_path = os.path.join(args.build_dir, f"model_{args.version}.joblib")
    if not os.path.exists(model_path):
        sys.exit(f"Model not found: {model_path}")
    model = joblib.load(model_path)

    metrics = evaluate(model)
    with open(os.path.join(args.build_dir, f"metrics_{args.version}.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    for name, value in metrics.items():
        print(f"  {name:<10} {value:.4f}")

    if not passes_gate(metrics, args.min_f1):
        sys.exit(f"QUALITY GATE FAILED: F1 {metrics['f1']:.4f} is below the minimum {args.min_f1:.4f}")
    print(f"Quality gate passed: F1 {metrics['f1']:.4f} >= {args.min_f1:.4f}")


if __name__ == "__main__":
    main()
