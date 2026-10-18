# Lab 2: Automated Model Training, Evaluation and Versioning with GitHub Actions

This lab is based on [Lab 2 of the MLOps course](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab2). It uses GitHub Actions to train a machine learning model, check its quality, and save a new version of it in the repository automatically.

I rebuilt the **GitHub Actions pipeline**, which is the core of the lab. The original workflow ran every step in one job and committed every model it trained. The new pipeline:

- runs in **4 separate jobs**: test → train → evaluate → promote
- **blocks bad models** with a quality gate
- commits a new model only when it **beats the current best model** (champion/challenger)
- can be **run by hand with your choice of model and settings**
- shows a **results table** on every run

All changes are listed in [What I changed and why](#what-i-changed-and-why).

---

## Contents

1. [How the pipeline works](#how-the-pipeline-works)
2. [Project structure](#project-structure)
3. [What you need to install](#what-you-need-to-install)
4. [Set up the project](#set-up-the-project)
5. [Run it on your computer](#run-it-on-your-computer)
6. [Run it on GitHub Actions](#run-it-on-github-actions)
7. [What a successful run looks like](#what-a-successful-run-looks-like)
8. [What I changed and why](#what-i-changed-and-why)
9. [Troubleshooting](#troubleshooting)

---

## How the pipeline works

```
 push to main / pull request / manual run
                 │
        ┌────────▼────────┐
        │ 1. Unit tests   │  pytest: 13 tests on data, training, evaluation and promotion code
        └────────┬────────┘
        ┌────────▼────────┐
        │ 2. Train model  │  trains the chosen model, saves model_<version>.joblib
        └────────┬────────┘  and passes it to the next job as an artifact
        ┌────────▼────────┐
        │ 3. Evaluate     │  scores the model on a held-out test set
        │    + gate       │  FAILS the run if F1 < minimum (default 0.90)
        │    + compare    │  compares F1 with the current champion in models/registry.json
        └────────┬────────┘
                 │  only if the new model beats the champion, and not on a pull request
        ┌────────▼────────┐
        │ 4. Promote      │  copies the model to models/, metrics to metrics/,
        └─────────────────┘  updates registry.json, commits and pushes back to the repo
```

- **Dataset:** the [Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html). It has 569 samples and 30 features, and the task is to classify tumors as malignant or benign. It comes with scikit-learn, so there is nothing to download.
- **Test set:** 20% of the data (114 rows) is held out with a fixed random seed. Every model is scored on the same rows, so results from different runs can be compared fairly.
- **Version tag:** each model gets a version made of the UTC time and the commit it was trained from, for example `20261018213045-9a359a7`.
- **Models:** `random_forest`, `gradient_boosting` or `logistic_regression`.

---

## Project structure

```
MLOps_Labs/
├── .github/workflows/
│   └── lab2_model_pipeline.yml   # the pipeline (GitHub only reads workflows from the repo root)
└── Lab2/
    ├── README.md
    ├── requirements.txt          # pinned package versions
    ├── src/
    │   ├── data.py               # loads the dataset and makes the fixed train/test split
    │   ├── train_model.py        # trains a model and saves build/model_<version>.joblib
    │   ├── evaluate_model.py     # scores the model and applies the quality gate
    │   └── promote_model.py      # champion/challenger check and model registry
    ├── test/
    │   └── test_pipeline.py      # unit tests run by job 1
    ├── models/                   # promoted models + registry.json (written by the pipeline)
    └── metrics/                  # metrics for each promoted model (written by the pipeline)
```

`build/` is a temporary folder for models that haven't been promoted yet. Git ignores it.

---

## What you need to install

| Tool | Version | Why |
|---|---|---|
| **Git** | any recent version | to download the code and push changes |
| **Python** | **3.11** recommended (3.9 to 3.12 work) | to run the code. GitHub Actions uses 3.11 |
| **GitHub account** | free | to run the pipeline on GitHub Actions |

These Python packages are installed by `pip` in a later step. The versions are pinned in `requirements.txt`, so everyone gets the same results:

| Package | Version |
|---|---|
| scikit-learn | 1.5.2 |
| numpy | 1.26.4 |
| joblib | 1.4.2 |
| pytest | 8.3.3 |

### Install Git

- **macOS:** open Terminal and run `git --version`. If Git isn't installed, macOS offers to install the Command Line Tools. Click **Install**.
- **Windows:** download and run the installer from <https://git-scm.com/download/win>, keeping the default options. Use **Git Bash** or **PowerShell** for the commands below.
- **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install git`

Check that it worked: `git --version` should print something like `git version 2.x.x`.

### Install Python 3.11

- **macOS:** download the macOS installer for Python 3.11 from <https://www.python.org/downloads/> and run it.
- **Windows:** download the Windows installer for Python 3.11 from <https://www.python.org/downloads/>. On the first screen, **tick "Add python.exe to PATH"**, then click Install Now.
- **Linux (Ubuntu/Debian):** `sudo apt install python3.11 python3.11-venv`

Check that it worked:

```bash
python3 --version      # macOS / Linux, should print Python 3.11.x
python --version       # Windows, should print Python 3.11.x
```

> In the rest of this guide, use `python3` on macOS/Linux and `python` on Windows.

---

## Set up the project

**1. Download the code**

```bash
git clone https://github.com/Chaitali04509/MLOps_Labs.git
cd MLOps_Labs/Lab2
```

**2. Create a virtual environment.** This keeps this lab's packages separate from the rest of your computer.

```bash
python3 -m venv venv
```

**3. Activate it**

```bash
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows (PowerShell or Command Prompt)
```

When it's active, `(venv)` appears at the start of your terminal prompt. Activate it again every time you open a new terminal.

**4. Install the dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run it on your computer

Run every command from inside the `Lab2` folder with the virtual environment active. These are the same commands the GitHub Actions jobs run.

**Step 1: Run the unit tests**

```bash
pytest -v
```

Expected: `13 passed`.

**Step 2: Train a model.** `--version` can be any label.

```bash
python3 src/train_model.py --version local-1 --model-type random_forest --n-estimators 100
```

Expected:

```
Training random_forest (n_estimators=100, max_depth=0)
Saved model to build/model_local-1.joblib
```

**Step 3: Evaluate it and apply the quality gate**

```bash
python3 src/evaluate_model.py --version local-1 --min-f1 0.90
```

Expected:

```
  accuracy   0.9561
  precision  0.9589
  recall     0.9722
  f1         0.9655
  roc_auc    0.9922
Quality gate passed: F1 0.9655 >= 0.9000
```

**Step 4: Compare it with the champion, then promote it**

```bash
python3 src/promote_model.py check --version local-1
python3 src/promote_model.py promote --version local-1
```

Expected, ending with:

```
Decision: PROMOTE. No champion yet, so the first model that passes the gate is promoted.
Promoted local-1 to champion (models/model_local-1.joblib)
```

You now have `models/model_local-1.joblib`, `metrics/metrics_local-1.json` and `models/registry.json`.

**Optional: see the gate and the comparison in action**

```bash
# Train a different model. Logistic regression scores higher (F1 0.9861), so it replaces the champion
python3 src/train_model.py --version local-2 --model-type logistic_regression
python3 src/evaluate_model.py --version local-2
python3 src/promote_model.py check --version local-2
# -> Decision: PROMOTE. Challenger F1 0.9861 beats champion F1 0.9655.

# A deliberately weak model with a strict gate fails
python3 src/train_model.py --version local-3 --n-estimators 1 --max-depth 1
python3 src/evaluate_model.py --version local-3 --min-f1 0.99
# -> QUALITY GATE FAILED: F1 0.8889 is below the minimum 0.9900   (exit code 1)
```

**Clean up before pushing.** Files you promoted locally would become the pipeline's starting champion, so delete them:

```bash
rm -rf build models/model_* models/registry.json metrics/metrics_*     # macOS / Linux / Git Bash
```

---

## Run it on GitHub Actions

To run the pipeline in your own account, put the code in a GitHub repository you own:

1. On GitHub, click **+ → New repository**. Give it a name, leave it empty (no README), and click **Create repository**.
2. Push this code to it. Replace `<your-username>` and `<your-repo>` with your own:
   ```bash
   cd ..        # go to the repository root (MLOps_Labs)
   git remote set-url origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```

The `.github/workflows/` folder must stay at the **root** of the repository, not inside `Lab2/`. GitHub ignores workflows anywhere else.

There are three ways to start the pipeline:

| Trigger | What happens |
|---|---|
| **Push to `main`** that changes a file in `Lab2/` | Runs all 4 jobs with the default settings (random forest, 100 trees, F1 gate 0.90) |
| **Pull request to `main`** | Runs test, train and evaluate, and shows the comparison. It **never promotes or commits** anything, so a PR can't change the champion before it's reviewed |
| **Manual run** | Go to **Actions** → **Lab2 - Model Training Pipeline** → **Run workflow**. Choose the model type, number of trees, max depth, minimum F1 and whether to force promotion, then click **Run workflow** |

Suggested first test:

1. **Push the code.** No champion exists yet, so the random forest is promoted and the pipeline commits it.
2. **Start a manual run with `logistic_regression`.** It scores higher, so it becomes the new champion.
3. **Start a manual run with `random_forest` again.** It doesn't beat the champion, so job 4 is **skipped** and nothing is committed.
4. **Start a manual run with `min_f1` set to `0.99`.** Job 3 **fails** at the quality gate.

---

## What a successful run looks like

In the **Actions** tab, open a run of **Lab2 - Model Training Pipeline**:

- **When a new champion is found**, all 4 jobs have green checks: `1. Unit tests` → `2. Train model` → `3. Evaluate and quality gate` → `4. Promote and version model`.
- **When the champion is kept**, jobs 1 to 3 are green and job 4 is grey (skipped). This is the expected result, not an error.
- **Run summary:** the bottom of the run page shows a table like this:

  | Metric | Challenger `20261018213045-9a359a7` | Champion (none) |
  |---|---|---|
  | model | random_forest | - |
  | accuracy | 0.9561 | - |
  | precision | 0.9589 | - |
  | recall | 0.9722 | - |
  | f1 | 0.9655 | - |
  | roc_auc | 0.9922 | - |

  **Decision: PROMOTE.** No champion yet, so the first model that passes the gate is promoted.

- **Artifacts:** the run page has `trained-model` and `evaluated-model` downloads, which contain the model and its metrics.
- **New commit:** when a model is promoted, a commit by **github-actions[bot]** named `Lab2: promote model <version> (<model_type>) [skip ci]` appears on `main`. It adds:
  - `Lab2/models/model_<version>.joblib` (the versioned model)
  - `Lab2/metrics/metrics_<version>.json` (its scores and settings)
  - an updated `Lab2/models/registry.json`

  Run `git pull` to get these files on your computer.

A **red X on job 3** with `QUALITY GATE FAILED` in the log means the gate worked: the model wasn't good enough, so nothing was saved.

---

## What I changed and why

### Pipeline changes (`.github/workflows/lab2_model_pipeline.yml`)

| Original lab | My version | What it does |
|---|---|---|
| One job runs every step | **4 jobs chained with `needs:`** (test → train → evaluate → promote) | Each stage shows its own pass/fail on GitHub. A failure stops the later jobs |
| No tests in the pipeline | **Job 1 runs pytest** before training | Broken code is caught before any model is trained |
| Files are moved between steps in one job | **Jobs pass the model as artifacts** (`upload-artifact` / `download-artifact`) | Jobs run on separate machines, as in real pipelines. You can download the model from the run page |
| Every trained model is committed | **Quality gate:** job 3 fails if F1 < `min_f1` | A bad model can never reach the repo |
| Every model is committed, even worse ones | **Champion/challenger:** the model is promoted only if its F1 is strictly higher than the champion's in `models/registry.json` | The repo only gets better models. The registry records which model is current and keeps a history of all promoted models |
| Runs only on push | Also runs on **pull requests** (evaluate only, never commits) and **manual runs with inputs** (model type, trees, depth, gate threshold, force promote) | You can try models from the browser without changing code. PRs show how a change would affect the model before it's merged |
| No results shown | **Run summary table** written to `$GITHUB_STEP_SUMMARY` | The challenger-vs-champion table is shown on the run page |
| Default token permissions. Fails on new repos, where the token is read-only | `permissions: contents: read` for the whole workflow, `contents: write` **only for the promote job** | Least privilege. Only the job that needs to push can push |
| Two runs could push at once and conflict | **`concurrency:` group** | Runs queue instead of overlapping, so registry commits never clash |
| Any push, including the pipeline's own commits, could retrigger it | **`paths:` filters** (ignore `models/`, `metrics/` and `.md` changes) plus `[skip ci]` in the bot's commit message | No endless loop. The pipeline runs only when Lab 2 code changes |
| Version is a timestamp only | **Timestamp + commit SHA**, e.g. `20261018213045-9a359a7` | You can trace each model back to the exact code that produced it |
| Commits under the instructor's name | Commits as **github-actions[bot]** | Shows that the commit was made by automation |
| `setup-python@v4`, Python 3.9, unpinned packages | `setup-python@v5`, Python 3.11, **pip cache**, **pinned versions** | Faster runs and the same results every time |

### Code changes (`src/`)

| Original lab | My version | Why |
|---|---|---|
| Random synthetic data with `randint(0, 2000)` rows. It can generate 0 rows and crash | A real dataset (breast cancer), always 569 rows | Reliable, meaningful results |
| Evaluated on newly generated data drawn from the same generator as the training data | **Fixed held-out test split** (`data.py`) | An honest score, and every run is tested on the same rows, so comparisons are fair |
| Random forest only, hard-coded | 3 model types with settings passed in from the workflow | Makes the manual run inputs possible |
| F1 only | Accuracy, precision, recall, F1, ROC AUC | A fuller picture of model quality |
| No tests | 13 pytest tests (`test/test_pipeline.py`) | Run by job 1 |
| MLflow logging to a local folder that was thrown away after each CI run | Removed. Metrics are saved as JSON in `metrics/` and `registry.json` | Results are now kept in the repo, and installs are faster |
| `promote_model.py` didn't exist | New. Decides champion vs challenger and updates the registry, using only the Python standard library | The promote job doesn't need to install anything |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python3: command not found` (Windows) | Use `python` instead of `python3`. If that also fails, reinstall Python and tick **"Add python.exe to PATH"** |
| `ModuleNotFoundError: No module named 'sklearn'` | The virtual environment isn't active. Run the activate command from [Set up the project](#set-up-the-project), then `pip install -r requirements.txt` |
| `Model not found: build/model_<version>.joblib` | Use the same `--version` for train, evaluate and promote, and run them from inside `Lab2/` |
| The workflow doesn't appear in the Actions tab | `.github/workflows/` must be at the repository root. Also check that **Settings → Actions → General** allows actions |
| Job 4 fails with `Permission denied` / `403` when pushing | Go to **Settings → Actions → General → Workflow permissions** and choose **Read and write permissions**. This is only needed if your organization blocks the `permissions:` setting in the workflow |
| Job 4 is skipped | The new model didn't beat the champion, or the run came from a pull request. This is expected. To promote anyway, start a manual run with **force_promote** ticked |
| A push didn't start the pipeline | Pushes start it only when files in `Lab2/` change, not counting `models/`, `metrics/` or `.md` files. Use a manual run instead |
