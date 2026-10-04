import os

import pytest
from src import metrics
from src.evaluate import evaluate, load_predictions


@pytest.fixture
def sample():
    # tp=3, fp=1, tn=2, fn=1
    y_true = [1, 1, 1, 0, 0, 0, 1]
    y_pred = [1, 1, 1, 1, 0, 0, 0]
    return y_true, y_pred


def test_confusion_matrix(sample):
    assert metrics.confusion_matrix(*sample) == {"tp": 3, "fp": 1, "tn": 2, "fn": 1}


def test_accuracy(sample):
    assert metrics.accuracy(*sample) == pytest.approx(5 / 7)


def test_precision(sample):
    assert metrics.precision(*sample) == pytest.approx(0.75)


def test_recall(sample):
    assert metrics.recall(*sample) == pytest.approx(0.75)


def test_f1_score(sample):
    assert metrics.f1_score(*sample) == pytest.approx(0.75)


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], 1.0),  # perfect model
        ([1, 0, 1, 0], [0, 1, 0, 1], 0.0),  # always wrong
        ([1, 1, 0, 0], [1, 1, 1, 1], 0.5),  # always predicts positive
    ],
)
def test_accuracy_cases(y_true, y_pred, expected):
    assert metrics.accuracy(y_true, y_pred) == pytest.approx(expected)


def test_zero_division_returns_zero():
    # No predicted positives and no actual positives
    assert metrics.precision([0, 0], [0, 0]) == 0.0
    assert metrics.recall([0, 0], [0, 0]) == 0.0
    assert metrics.f1_score([0, 0], [0, 0]) == 0.0


@pytest.mark.parametrize(
    "y_true, y_pred, mae, mse",
    [
        ([3, -0.5, 2, 7], [2.5, 0.0, 2, 8], 0.5, 0.375),
        ([1, 2, 3], [1, 2, 3], 0.0, 0.0),
        ([0, 0], [2, -2], 2.0, 4.0),
    ],
)
def test_regression_metrics(y_true, y_pred, mae, mse):
    assert metrics.mean_absolute_error(y_true, y_pred) == pytest.approx(mae)
    assert metrics.mean_squared_error(y_true, y_pred) == pytest.approx(mse)


@pytest.mark.parametrize(
    "func, y_true, y_pred",
    [
        (metrics.accuracy, [], []),                    # empty
        (metrics.accuracy, [1, 0], [1]),               # length mismatch
        (metrics.accuracy, [1, 2], [1, 0]),            # non-binary label
        (metrics.mean_absolute_error, [1, "a"], [1, 2]),  # non-numeric
        (metrics.mean_squared_error, [True], [1]),     # bool is not a number here
    ],
)
def test_invalid_inputs_raise(func, y_true, y_pred):
    with pytest.raises(ValueError):
        func(y_true, y_pred)


def test_evaluate_on_dataset():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "predictions.csv")
    y_true, y_pred = load_predictions(path)
    report = evaluate(y_true, y_pred)
    assert len(y_true) == 20
    assert report["tp"] + report["fp"] + report["tn"] + report["fn"] == 20
    # Quality gate: CI fails if the model's accuracy drops below 75%
    assert report["accuracy"] >= 0.75
