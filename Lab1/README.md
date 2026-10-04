# MLOps Lab 1: Testing ML Evaluation Metrics with Pytest, Unittest and GitHub Actions

This is my version of [Lab 1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab1) from the MLOps course (IE-7374). The original lab tests a basic calculator. This version tests an **ML model evaluation metrics module** instead, runs it against a predictions dataset, and checks it automatically with GitHub Actions.

## What I changed from the original lab

| Original lab | This version |
|---|---|
| `calculator.py` with add / subtract / multiply | `metrics.py` with accuracy, precision, recall, F1, confusion matrix, MAE and MSE |
| No data | `data/predictions.csv` with 20 model predictions, plus `src/evaluate.py` that prints a metrics report for it |
| Plain assert tests | Pytest **fixtures**, **parametrized** tests, edge cases (divide by zero, empty input, bad labels) and a **quality gate** that fails if model accuracy drops below 75% |
| Runs on Python 3.8 only | Runs on a **matrix of Python 3.9, 3.10, 3.11 and 3.12** |
| No coverage | **Code coverage** with `pytest-cov`. CI fails if coverage drops below 90% |
| `actions/*@v2` (upload-artifact v2 no longer works) | Current versions: `checkout@v4`, `setup-python@v5`, `upload-artifact@v4`, with pip caching |
| Runs only on push | Runs on push, pull requests and manually (`workflow_dispatch`) |

## Project structure

```
MLOps_Labs/
├── .github/workflows/          # GitHub only runs workflows from the repo root
│   ├── lab1_pytest.yml         # pytest + coverage on Python 3.9 to 3.12
│   └── lab1_unittest.yml       # unittest suite
└── Lab1/
    ├── data/
    │   └── predictions.csv     # y_true, y_pred for 20 samples
    ├── src/
    │   ├── metrics.py          # evaluation metrics
    │   └── evaluate.py         # loads the CSV and prints a report
    ├── test/
    │   ├── test_pytest.py
    │   └── test_unittest.py
    └── requirements.txt
```

## Running locally

```bash
cd Lab1
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Pytest with a coverage report
pytest test/test_pytest.py --cov=src --cov-report=term-missing

# Unittest
python -m unittest test.test_unittest -v

# Metrics report for the dataset
python -m src.evaluate
```

Example output:

```
Evaluated 20 predictions from predictions.csv
  accuracy   0.800
  precision  0.800
  recall     0.800
  f1_score   0.800
  tp         8
  fp         2
  tn         8
  fn         2
```

## GitHub Actions

Every push or pull request to `main` that changes files in `Lab1/` triggers both workflows. They run with `Lab1/` as the working directory:

- **Lab1 - Testing with Pytest** runs the pytest suite with coverage on four Python versions, prints the metrics report, and uploads the JUnit XML and coverage XML reports as artifacts.
- **Lab1 - Python Unittests** runs the unittest suite.

You can see the results in the repository's **Actions** tab.
