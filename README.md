# Analysis Tools

A curated set of analysis scripts, utilities, and notebooks for modeling, data exploration, and evaluation.

## Setup 

### For use in your own project


### For Development
```bash
  1. source ~/.bashrc
  2. python -m pip install poetry
  3. poetry install --no-root (cd analysis-tools)
  
  # For using and testing tools - Will create a kernel called "Python (analysis-tools)"
  4. poetry run pip install ipykernel
  5. poetry run python -m ipykernel install --user --name=analysis-tools --display-name "Python (analysis-tools)"

```

## Model Evaluation


## Model Training and Hyperparameter Tuning

`GBMModelTrainer` is a class that supports GBM model training, logging, and evaluation using XGBoost, CATBoost, or LightGBM. 

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


`GBMModelTrainer` now supports XGBoost only and utilises
[`xgboost.train`](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.train).
Hyperparameter tuning is performed via `skopt.gp_minimize` and can accept a
custom objective function. Add a `tuning` section to your YAML configuration:

```yaml
actual_col: target
predicted_col: pred
base_margin_col: offset
tuning:
  search_space:
    n_estimators: [50, 100]
    max_depth: [3, 4, 5]
  n_iter: 10
  # optionally a path to write tuning results
  tuning_file: tuning.csv
```

Call `trainer.tune()` before `trainer.train()` and the best parameters will be
used for training and stored in the configuration. You can also override the
search space directly:

```python
trainer.tune({"max_depth": (2, 5), "eta": (0.1, 0.3)})
```

If your training data includes an offset column, specify it in the configuration
using `base_margin_col` and the values will be supplied to XGBoost as
`base_margin`.
