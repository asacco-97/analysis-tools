import pytest

pytest.importorskip("xgboost")

import pandas as pd
import yaml
from sklearn.datasets import make_classification

from training.gbm_model_trainer import GBMModelTrainer


def test_tune_updates_params(tmp_path):
    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(4)])
    df["target"] = y
    train_df = df.iloc[:20].reset_index(drop=True)
    valid_df = df.iloc[20:30].reset_index(drop=True)

    cfg = {
        "actual_col": "target",
        "predicted_col": "pred",
        "tuning": {"n_iter": 2},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    trainer = GBMModelTrainer(object, str(cfg_path), train_df, valid_df)
    trainer.tune({"max_depth": (2, 3)})

    assert "max_depth" in trainer.config.hyperparameters
    assert any("Trial" in line for line in trainer.log_lines)


def test_base_margin_column(tmp_path):
    X, y = make_classification(n_samples=20, n_features=4, random_state=1)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(4)])
    df["target"] = y
    df["offset"] = 0.5

    train_df = df.iloc[:10].reset_index(drop=True)
    valid_df = df.iloc[10:].reset_index(drop=True)

    cfg = {
        "actual_col": "target",
        "predicted_col": "pred",
        "base_margin_col": "offset",
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    trainer = GBMModelTrainer(object, str(cfg_path), train_df, valid_df)
    X_t, y_t, base_t = trainer._split_xy(train_df)
    X_v, y_v, base_v = trainer._split_xy(valid_df)

    assert "offset" not in X_t.columns
    assert base_t is not None and len(base_t) == len(train_df)
