# Analysis Tools

A curated set of analysis scripts, utilities, and notebooks for modeling, data exploration, and evaluation.

## 🔧 Setup

Clone the repo and install dependencies with [`uv`](https://github.com/astral-sh/uv):

```bash
# If uv is not installed
pip install uv

# Create virtual environment
uv venv
source .venv/bin/activate  # or `.venv\\Scripts\\activate` on Windows
uv pip install -r uv.lock.new

# For use within a jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name=analysis-tools --display-name "Python (analysis-tools)"

```

## ⚙️ Hyperparameter Tuning

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
