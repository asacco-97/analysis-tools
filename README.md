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
