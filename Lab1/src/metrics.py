"""
Evaluation metrics for machine learning models, written in plain Python.

Classification metrics assume binary labels (0 = negative, 1 = positive).
Regression metrics accept any int/float values.
"""


def _validate(y_true, y_pred):
    """
    Checks that the two inputs are non-empty lists of numbers with equal length.
    Raises:
        ValueError: If the inputs are empty, have different lengths,
                    or contain non-numeric values.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Inputs must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    for value in list(y_true) + list(y_pred):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("All values must be numbers.")


def _validate_binary(y_true, y_pred):
    """
    Same as _validate, and also checks that every label is 0 or 1.
    """
    _validate(y_true, y_pred)
    for value in list(y_true) + list(y_pred):
        if value not in (0, 1):
            raise ValueError("Classification labels must be 0 or 1.")


def confusion_matrix(y_true, y_pred):
    """
    Counts true/false positives and negatives.
    Returns:
        dict: {"tp": int, "fp": int, "tn": int, "fn": int}
    """
    _validate_binary(y_true, y_pred)
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            counts["tp"] += 1
        elif actual == 0 and predicted == 1:
            counts["fp"] += 1
        elif actual == 0 and predicted == 0:
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts


def accuracy(y_true, y_pred):
    """
    Fraction of predictions that match the true labels.
    """
    _validate_binary(y_true, y_pred)
    correct = sum(1 for a, p in zip(y_true, y_pred) if a == p)
    return correct / len(y_true)


def precision(y_true, y_pred):
    """
    Of everything predicted positive, the fraction that was actually positive.
    Returns 0.0 when nothing was predicted positive.
    """
    cm = confusion_matrix(y_true, y_pred)
    predicted_positive = cm["tp"] + cm["fp"]
    if predicted_positive == 0:
        return 0.0
    return cm["tp"] / predicted_positive


def recall(y_true, y_pred):
    """
    Of everything actually positive, the fraction the model found.
    Returns 0.0 when there are no actual positives.
    """
    cm = confusion_matrix(y_true, y_pred)
    actual_positive = cm["tp"] + cm["fn"]
    if actual_positive == 0:
        return 0.0
    return cm["tp"] / actual_positive


def f1_score(y_true, y_pred):
    """
    Harmonic mean of precision and recall.
    Returns 0.0 when both precision and recall are 0.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mean_absolute_error(y_true, y_pred):
    """
    Average absolute difference between predictions and true values.
    """
    _validate(y_true, y_pred)
    return sum(abs(a - p) for a, p in zip(y_true, y_pred)) / len(y_true)


def mean_squared_error(y_true, y_pred):
    """
    Average squared difference between predictions and true values.
    """
    _validate(y_true, y_pred)
    return sum((a - p) ** 2 for a, p in zip(y_true, y_pred)) / len(y_true)
