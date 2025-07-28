import pandas as pd
import yaml
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from training.gbm_model_trainer import GBMModelTrainer


def test_tune_updates_params(tmp_path):
    X, y = make_classification(n_samples=40, n_features=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(4)])
    df['target'] = y
    train_df = df.iloc[:20].reset_index(drop=True)
    valid_df = df.iloc[20:30].reset_index(drop=True)

    cfg = {
        'actual_col': 'target',
        'predicted_col': 'pred',
        'tuning': {
            'search_space': {'n_estimators': [5, 10]},
            'n_iter': 2,
            'scoring': 'accuracy'
        }
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    trainer = GBMModelTrainer(RandomForestClassifier, str(cfg_path), train_df, valid_df)
    try:
        trainer.tune()
    except ImportError:
        # scikit-optimize not installed in test environment
        return

    assert 'n_estimators' in trainer.config.hyperparameters
    assert any('Trial' in line for line in trainer.log_lines)
    trainer.train()
    assert trainer.model.n_estimators == trainer.config.hyperparameters['n_estimators']

