from xgboost.callback import TrainingCallback, EarlyStopping
import logging
from typing import Optional

class LogEvalCallback(TrainingCallback):
    def __init__(self, period: int = 50, logger: Optional[logging.Logger] = None):
        self.period = period
        self.logger = logger or logging.getLogger(__name__)

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        if epoch % self.period == 0 and evals_log:
            for data_name, metrics in evals_log.items():
                for metric_name, values in metrics.items():
                    msg = f"[{epoch}] {data_name}-{metric_name}: {values[-1]:.5f}"
                    if self.logger:
                        self.logger.info(msg)
        return False

class EarlyStoppingCallback(TrainingCallback):
    def __init__(self, stopping_rounds: int = 30, metric_name: Optional[str] = None, maximize: bool = False):
        self.stopping_cb = EarlyStopping(
            rounds=stopping_rounds,
            metric_name=metric_name,
            data_name="validation",
            save_best=True,
            maximize=maximize,
        )

    def before_training(self, model):
        return self.stopping_cb.before_training(model)

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        return self.stopping_cb.after_iteration(model, epoch, evals_log)

    def after_training(self, model):
        return self.stopping_cb.after_training(model)

# Template for custom callback
class CustomCallback(TrainingCallback):
    def __init__(self, **kwargs):
        pass  # Initialize with custom arguments

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        # Insert custom logic here
        return False

