# MLOps Labs

My lab submissions for the MLOps course (IE-7374). The labs are based on [raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps), with my own changes.

| Lab | Topic | What I changed |
|---|---|---|
| [Lab1](Lab1/) | Pytest, Unittest and GitHub Actions | Tests an ML evaluation metrics module and a predictions dataset instead of a calculator. Adds a Python version matrix, coverage and an accuracy quality gate |
| [Lab2](Lab2/) | Automated model training and versioning with GitHub Actions | Rebuilt the pipeline as 4 jobs (test → train → evaluate → promote) with a quality gate, champion/challenger promotion, a model registry, manual runs with inputs and run summaries |
