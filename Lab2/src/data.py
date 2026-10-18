"""
Loads the dataset and splits it the same way every time.

A fixed split seed means every model is scored on the exact same test rows,
so metrics from different runs can be compared fairly.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

SPLIT_SEED = 42
TEST_SIZE = 0.2


def load_split():
    """
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y)
