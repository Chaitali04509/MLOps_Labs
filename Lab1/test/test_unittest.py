import os
import sys
import unittest

# Get the path to the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src import metrics


class TestClassificationMetrics(unittest.TestCase):

    def setUp(self):
        # tp=3, fp=1, tn=2, fn=1
        self.y_true = [1, 1, 1, 0, 0, 0, 1]
        self.y_pred = [1, 1, 1, 1, 0, 0, 0]

    def test_confusion_matrix(self):
        self.assertEqual(
            metrics.confusion_matrix(self.y_true, self.y_pred),
            {"tp": 3, "fp": 1, "tn": 2, "fn": 1},
        )

    def test_accuracy(self):
        self.assertAlmostEqual(metrics.accuracy(self.y_true, self.y_pred), 5 / 7)

    def test_precision_recall_f1(self):
        self.assertAlmostEqual(metrics.precision(self.y_true, self.y_pred), 0.75)
        self.assertAlmostEqual(metrics.recall(self.y_true, self.y_pred), 0.75)
        self.assertAlmostEqual(metrics.f1_score(self.y_true, self.y_pred), 0.75)

    def test_zero_division_returns_zero(self):
        self.assertEqual(metrics.precision([0, 0], [0, 0]), 0.0)
        self.assertEqual(metrics.recall([0, 0], [0, 0]), 0.0)
        self.assertEqual(metrics.f1_score([0, 0], [0, 0]), 0.0)

    def test_non_binary_label_raises(self):
        with self.assertRaises(ValueError):
            metrics.accuracy([1, 2], [1, 0])


class TestRegressionMetrics(unittest.TestCase):

    def test_mean_absolute_error(self):
        self.assertAlmostEqual(metrics.mean_absolute_error([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]), 0.5)

    def test_mean_squared_error(self):
        self.assertAlmostEqual(metrics.mean_squared_error([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]), 0.375)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            metrics.mean_absolute_error([], [])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            metrics.mean_squared_error([1, 2], [1])


if __name__ == '__main__':
    unittest.main()
