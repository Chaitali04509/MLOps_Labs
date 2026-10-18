"""
Trains a classifier and saves it as a versioned .joblib file.

Usage:
    python src/train_model.py --version 20261018120000-abc1234 \
        --model-type random_forest --n-estimators 200 --max-depth 0
"""

import argparse
import json
import os
import sys

from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import load_split

MODEL_TYPES = ("random_forest", "gradient_boosting", "logistic_regression")


def build_model(model_type, n_estimators=100, max_depth=0, seed=0):
    """
    Creates an untrained model. max_depth=0 means "no limit" for the forest
    and "use the default (3)" for gradient boosting.
    """
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth or None, random_state=seed
        )
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth or 3, random_state=seed
        )
    if model_type == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    raise ValueError(f"Unknown model type '{model_type}'. Choose from {MODEL_TYPES}.")


def train(model_type, n_estimators=100, max_depth=0, seed=0):
    """
    Trains on the training split only. The test split is kept for evaluation.
    """
    X_train, _, y_train, _ = load_split()
    model = build_model(model_type, n_estimators, max_depth, seed)
    model.fit(X_train, y_train)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Version tag, e.g. timestamp + commit SHA")
    parser.add_argument("--model-type", default="random_forest", choices=MODEL_TYPES)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--build-dir", default="build")
    args = parser.parse_args()

    print(f"Training {args.model_type} (n_estimators={args.n_estimators}, max_depth={args.max_depth})")
    model = train(args.model_type, args.n_estimators, args.max_depth, args.seed)

    os.makedirs(args.build_dir, exist_ok=True)
    model_path = os.path.join(args.build_dir, f"model_{args.version}.joblib")
    dump(model, model_path, compress=3)

    params = {
        "version": args.version,
        "model_type": args.model_type,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "seed": args.seed,
    }
    with open(os.path.join(args.build_dir, f"params_{args.version}.json"), "w") as f:
        json.dump(params, f, indent=4)

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
