# Analysis Tools

A curated set of analysis scripts, utilities, and notebooks for modeling, data exploration, and evaluation.

## Setup

Clone the repo and install dependencies with [`uv`](https://github.com/astral-sh/uv):

```bash
  1. source ~/.bashrc
  2. python -m pip install poetry
  3. poetry install --no-root (cd analysis-tools)
  
  # For using and testing tools - Will create a kernel called "Python (analysis-tools)"
  4. poetry run pip install ipykernel
  5. poetry run python -m ipykernel install --user --name=analysis-tools --display-name "Python (analysis-tools)"

```

## Model Evaluation


## Hyperparameter Tuning

`GBMModelTrainer` supports Bayesian hyperparameter search via
[`scikit-optimize`](https://scikit-optimize.github.io/).
Add a `tuning` section to your YAML configuration:

```yaml
actual_col: target
predicted_col: pred
tuning:
  search_space:
    n_estimators: [50, 100]
    max_depth: [3, 4, 5]
  n_iter: 10
  scoring: accuracy
  n_jobs: -1
```

Call `trainer.tune()` before `trainer.train()` and the best parameters will be
used for training and stored in the configuration.

