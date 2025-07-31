from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging
import yaml
import time
from typing import Any, Dict, Tuple, Optional, List, Callable, Union
import pandas as pd
import numpy as np

if not hasattr(np, "int"):
    np.int = int
import shutil

from xgboost import DMatrix, train as xgb_train
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize

from analysis.report import ModelAnalysisReport
from analysis import plots
from scoring.callbacks import LogEvalCallback, EarlyStoppingCallback
from scoring.scorers import get_scorer
from xgboost.callback import TrainingCallback

default_report_plots = [
    {"plot": "gain_curve_with_gini", "title": "Gain Curve / Lorenz Curve"},
    {
        "plot": "partial_gini_plot",
        "title": "Partial Gini (Top 15%)",
        "kwargs": {"top_percent": 15},
    },
    {"plot": "lift_chart", "title": "Lift Chart"},
    {"plot": "crunched_residual_plot", "title": "Crunched Residuals"},
    {
        "plot": "plot_residual_fit",
        "title": "Std and Avg of Normalized Residuals",
        "kwargs": {"residual_type": "normalized"},
    },
]

@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""

    search_space: Dict[str, Any] = field(default_factory=dict)
    n_iter: int = 10
    objective_func: Optional[Callable[[pd.Series, np.ndarray], float]] = None
    output_tuning: bool = False
    tuning_file: str = "tuning.csv"


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation and reporting."""

    output_report: bool = False
    report_file: str = "model_analysis.html"
    report_params: Dict[str, Any] = field(default_factory=dict)
    plots_to_add: List[Dict[str, Any]] = field(default_factory=list)
    tabulate_vars: List[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    actual_col: str
    predicted_col: str
    output_dir: str = "outputs"
    log_file: str = "training.log"
    model_file: str = "model_obj.json"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feval: Optional[Union[str, Callable[[pd.Series, np.ndarray], float]]] = None
    output_log: bool = True
    base_margin_col: str | None = None

    log_eval_period: int = 50
    early_stopping_rounds: Optional[int] = None
    early_stopping_metric: Optional[str] = None
    early_stopping_maximize: bool = False

    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)

class XGBModelTrainer:
    def __init__(
        self,
        config_path: str,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        holdout_df: pd.DataFrame | None = None,
    ) -> None:
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.train_df = train_df
        self.valid_df = valid_df
        self.holdout_df = holdout_df
        self.model = None
        self.log_lines = []
        self.logger = self._initialize_logger()

    def _load_config(self, config_path: str) -> TrainingConfig:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        if "tuning" in cfg:
            if isinstance(cfg["tuning"].get("search_space"), dict):
                cfg["tuning"]["search_space"] = parse_search_space(cfg["tuning"]["search_space"])
                cfg["tuning"]["objective_func"] = get_scorer(cfg["tuning"]["objective_func"])
            cfg["tuning"] = TuningConfig(**cfg["tuning"])

        if "evaluation" in cfg and isinstance(cfg["evaluation"], dict):
            cfg["evaluation"] = EvaluationConfig(**cfg["evaluation"])

        return TrainingConfig(**cfg)

    def _split_xy(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series | None, Optional[pd.Series]]:
        """Split dataframe into features, target, and optional base margin."""

        df_proc = df.copy()
        base_margin = None
        if self.config.base_margin_col and self.config.base_margin_col in df_proc.columns:
            base_margin = df_proc.pop(self.config.base_margin_col)

        actual_col = self.config.actual_col
        if actual_col and actual_col in df_proc.columns:
            y = df_proc.pop(actual_col)
            X = df_proc
            return X, y, base_margin

        return df_proc, None, base_margin

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure object columns are cast to categorical."""
        df_clean = df.copy()
        for col in df_clean.select_dtypes(include=["object", "string"]).columns:
            df_clean[col] = df_clean[col].astype("category")

        return df_clean

    def _ensure_parent_dir(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_output_path(self, filename: str | Path) -> Tuple[Path, str | None]:
        """
        Returns a tuple of (local_path, s3_dest) — even if output_dir is S3, file is written locally first.
        """
        filename = Path(filename).name
        output_dir = self.config.output_dir

        if is_s3_path(output_dir):
            local_dir = Path("/tmp/model_outputs")
            local_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_dir / filename
            s3_path = f"{output_dir.rstrip('/')}/{filename}"
            return local_path, s3_path
        else:
            local_path = Path(output_dir) / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            return local_path, None
        
    def _initialize_logger(self, name="xgb-trainer") -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            log_path, _ = self._resolve_output_path(self.config.log_file)
            file_handler = logging.FileHandler(log_path, mode='w')  # overwrite log file
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        if not any(isinstance(h, ListLogHandler) for h in logger.handlers):
            memory_handler = ListLogHandler(self.log_lines)
            memory_handler.setFormatter(formatter)
            logger.addHandler(memory_handler)

        return logger

    def _write_log(self) -> str:
        local_path, s3_path = self._resolve_output_path(self.config.log_file)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.log_lines))

        if s3_path:
            upload_file_to_s3(local_path, s3_path)
            self.logger.info(f"Log uploaded to {s3_path}")

        return str(local_path)

    def _predict(self, X: pd.DataFrame, base_margin: Optional[pd.Series] = None) -> np.ndarray:
        dmatrix = DMatrix(X, base_margin=base_margin, enable_categorical=True)
        preds = self.model.predict(dmatrix, output_margin=False)

        # For logistic regression return probability of positive class
        if preds.ndim == 1:
            return preds.astype(float)
        elif preds.ndim == 2:
            return preds[:, 1]
        else:
            raise ValueError(f"Unexpected prediction dimension: {preds.ndim}")

    def _prepare_callbacks(self) -> List[TrainingCallback]:
        callbacks = []

        # Always include logging
        callbacks.append(LogEvalCallback(
            period=self.config.log_eval_period,
            logger=self.logger,
        ))

        # Optionally include early stopping
        if isinstance(self.config.early_stopping_rounds, int):
            callbacks.append(EarlyStoppingCallback(
                stopping_rounds=self.config.early_stopping_rounds,
                metric_name=self.config.early_stopping_metric,
                maximize=self.config.early_stopping_maximize,
            ))

        return callbacks
    
    def _fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        base_margin_train: Optional[pd.Series] = None,
        base_margin_valid: Optional[pd.Series] = None,
    ) -> None:
        params = dict(self.config.hyperparameters or {})
        num_round = int(params.pop("num_boost_round", params.pop("n_estimators", 200)))

        dtrain = DMatrix(X_train, label=y_train, base_margin=base_margin_train, enable_categorical=True)
        evals = []
        if y_valid is not None:
            dvalid = DMatrix(
                X_valid,
                label=y_valid,
                base_margin=base_margin_valid,
                enable_categorical=True,
            )
            evals = [(dvalid, "validation")]

        # Extract custom objective if provided (must be callable)
        objective = self.config.hyperparameters["objective"]
        custom_obj = objective if callable(objective) else None
        if isinstance(objective, str):
            params["objective"] = objective  # set in params only if it's a string

        self.model = xgb_train(
            params,
            dtrain,
            num_boost_round=num_round,
            evals=evals,
            callbacks=self._prepare_callbacks(),
            obj=custom_obj,
            verbose_eval=False,
        )

    def tune(self, search_ranges: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Hyperparameter tuning using Bayesian optimization.

        Parameters
        ----------
        search_ranges:
            Optional dictionary mapping parameter names to either
            ``(low, high)`` tuples/lists or lists of categorical values.
            When omitted, the ranges defined in the configuration file are used.
        """
        self.logger.info("---"*25)
        self.logger.info("BEGINNING BAYESIAN HYPERPARAMETER TUNING")
        tuning_cfg = self.config.tuning
        search_space = parse_search_space(search_ranges) if search_ranges else tuning_cfg.search_space
        if not search_space:
            self.logger.info("No tuning search space provided; skipping tuning")
            return pd.DataFrame()

        (
            X_train,
            y_train,
            base_margin_train,
        ) = self._split_xy(self._prepare_dataframe(self.train_df))
        (
            X_valid,
            y_valid,
            base_margin_valid,
        ) = self._split_xy(self._prepare_dataframe(self.valid_df))

        param_names = list(search_space.keys())
        dimensions = list(search_space.values())

        def objective(values):
            params = {k: v for k, v in zip(param_names, values)}
            self.config.hyperparameters.update(params)
            self._fit_model(
                X_train,
                y_train,
                X_valid,
                y_valid,
                base_margin_train,
                base_margin_valid,
            )
            preds = self._predict(X_valid, base_margin=base_margin_valid)
            if tuning_cfg.objective_func is None:
                score = float(np.mean((y_valid - preds) ** 2))
            else:
                score = float(tuning_cfg.objective_func(y_valid, preds))
            self.logger.info(f"Trial {objective.counter}: score={score:.5f} params={params}")
            objective.counter += 1
            return score

        objective.counter = 1
        result = gp_minimize(objective, dimensions, n_calls=tuning_cfg.n_iter)
        best_params = {name: val for name, val in zip(param_names, result.x)}
        self.config.hyperparameters.update(best_params)

        # Output df containing tuning results
        tuning_df = pd.DataFrame(result.x_iters, columns=param_names)
        tuning_df["score"] = result.func_vals
        tuning_df = tuning_df.sort_values("score", ascending=True).reset_index(drop=True)

        if tuning_cfg.output_tuning and tuning_cfg.tuning_file:
            tuning_local_path, tuning_s3_path = self._resolve_output_path(
                tuning_cfg.tuning_file
            )
            tuning_df.to_csv(tuning_local_path, index=False)
            self.logger.info(f"Tuning DF saved to {tuning_local_path.resolve()}")
            if tuning_s3_path:
                upload_file_to_s3(tuning_local_path, tuning_s3_path)
                self.logger.info(f"Tuning DF uploaded to {tuning_s3_path}")

        return tuning_df

    def train(self) -> None:
        """Main training routine."""
        
        self.logger.info("---"*25)
        self.logger.info("BEGINNING MODEL TRAINING")

        start = time.time()
        (
            X_train,
            y_train,
            base_margin_train,
        ) = self._split_xy(self._prepare_dataframe(self.train_df))
        (
            X_valid,
            y_valid,
            base_margin_valid,
        ) = self._split_xy(self._prepare_dataframe(self.valid_df))
        
        self._fit_model(
            X_train,
            y_train,
            X_valid,
            y_valid,
            base_margin_train,
            base_margin_valid,
        )
        fit_time = time.time() - start

        self.logger.info(f"Training completed in {fit_time:.4f} seconds")
        self.logger.info("Model: XGBoost Booster")
        self.logger.info(f"Hyperparameters: {self.config.hyperparameters}")

        n_rows, n_cols = self.train_df.shape
        self.logger.info(f"Training data shape: {self.train_df.shape}")
        self.logger.info(f"Validation data shape: {self.valid_df.shape}")
        self.logger.info(
            f"Number of predictors: {n_cols - 1 if self.config.actual_col in self.train_df.columns else n_cols}"
        )
        self.logger.info(f"Target column: '{self.config.actual_col}'")
        self.logger.info(f"Prediction column: '{self.config.predicted_col}'")

        # Save model and log path
        model_path, model_s3_path = self._resolve_output_path(self.config.model_file)
        self.model.save_model(model_path)
        self.logger.info(f"Model saved to {model_path.resolve()}")
        if model_s3_path:
            upload_file_to_s3(model_path, model_s3_path)
            self.logger.info(f"Model uploaded to {model_s3_path}")

        # Save a copy of the config
        config_local_path, config_s3_path = self._resolve_output_path("config.yaml")
        shutil.copy(self.config_path, config_local_path)
        self.logger.info(f"Saved config to {config_local_path}")

        if config_s3_path:
            upload_file_to_s3(config_local_path, config_s3_path)
            self.logger.info(f"Config uploaded to {config_s3_path}")

        if self.config.output_log:
            self._write_log()

    def evaluate(self) -> None:
        """Generate a :class:`ModelAnalysisReport` using the trained model."""

        self.logger.info("---"*25)
        self.logger.info("BEGINNING MODEL EVALUATION")

        eval_cfg = self.config.evaluation
        if not eval_cfg.output_report:
            return

        (
            X_train,
            y_train,
            base_margin_train,
        ) = self._split_xy(self._prepare_dataframe(self.train_df))
        (
            X_valid,
            y_valid,
            base_margin_valid,
        ) = self._split_xy(self._prepare_dataframe(self.valid_df))

        if self.holdout_df is not None and not self.holdout_df.empty:
            (
                X_holdout,
                y_holdout,
                base_margin_holdout,
            ) = self._split_xy(self._prepare_dataframe(self.holdout_df))
        else:
            X_holdout = y_holdout = base_margin_holdout = None

        # Predict
        train_preds = self._predict(X_train, base_margin=base_margin_train)
        valid_preds = self._predict(X_valid, base_margin=base_margin_valid)

        actual_col = self.config.actual_col
        pred_col = self.config.predicted_col

        train_df_with_preds = self.train_df.copy()
        valid_df_with_preds = self.valid_df.copy()
        train_df_with_preds[pred_col] = train_preds
        valid_df_with_preds[pred_col] = valid_preds
        train_df_with_preds["split"] = "T"
        valid_df_with_preds["split"] = "V"

        frames = [train_df_with_preds, valid_df_with_preds]

        if X_holdout is not None:
            holdout_preds = self._predict(X_holdout, base_margin=base_margin_holdout)
            holdout_df_with_preds = self.holdout_df.copy()
            holdout_df_with_preds[pred_col] = holdout_preds
            holdout_df_with_preds["split"] = "H"
            frames.append(holdout_df_with_preds)

        df = pd.concat(frames, axis=0).reset_index(drop=True)

        mar = ModelAnalysisReport(
            df,
            actual_col=actual_col,
            predicted_col=pred_col,
            **eval_cfg.report_params,
        )

        if len(eval_cfg.tabulate_vars) > 0:
            mar.add_tabulation(variables=eval_cfg.tabulate_vars)

        plots_to_add = eval_cfg.plots_to_add or default_report_plots
        for plot_cfg in plots_to_add:
            func_name = plot_cfg.get("plot")
            title = plot_cfg.get("title", func_name)
            kwargs = plot_cfg.get("kwargs", {})
            try:
                plot_func = getattr(plots, func_name)
                mar.add_plot(plot_func, title, **kwargs)
                self.logger.info(f"Added plot: {title} ({func_name})")
            except Exception as e:
                self.logger.info(f"Failed to add plot '{func_name}': {e}")

        report_local_path, report_s3_path = self._resolve_output_path(eval_cfg.report_file)
        report_local_path.parent.mkdir(parents=True, exist_ok=True)
        mar.save(report_local_path)
        self.logger.info(f"Analysis report saved to {report_local_path.resolve()}")

        if report_s3_path:
            upload_file_to_s3(report_local_path, report_s3_path)
            self.logger.info(f"Analysis report uploaded to {report_s3_path}")

        if self.config.output_log:
            self._write_log()


def parse_search_space(raw_space: dict) -> dict:
    """Convert a user-provided search space into skopt dimensions."""

    search_space: dict[str, Any] = {}
    for name, cfg in raw_space.items():
        # Allow simple list/tuple specification
        if isinstance(cfg, (list, tuple)):
            if len(cfg) == 2:
                low, high = cfg
                if all(isinstance(v, int) for v in (low, high)):
                    search_space[name] = Integer(low, high)
                else:
                    search_space[name] = Real(float(low), float(high))
            else:
                search_space[name] = Categorical(list(cfg))
            continue

        if not isinstance(cfg, dict):
            raise ValueError(f"Unsupported search space config for '{name}': {cfg}")

        param_type = cfg["type"].lower()
        if param_type == "integer":
            search_space[name] = Integer(
                low=cfg["low"],
                high=cfg["high"],
                prior=cfg.get("prior", "uniform"),
            )
        elif param_type == "real":
            search_space[name] = Real(
                low=cfg["low"],
                high=cfg["high"],
                prior=cfg.get("prior", "uniform"),
            )
        elif param_type == "categorical":
            search_space[name] = Categorical(categories=cfg["categories"])
        else:
            raise ValueError(f"Unsupported type: {param_type}")
    return search_space


def is_s3_path(path: str | Path) -> bool:
    return str(path).startswith("s3://")


def upload_file_to_s3(local_path: Path, s3_path: str) -> None:
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/") + "/" + local_path.name

    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)


class ListLogHandler(logging.Handler):
    def __init__(self, log_list: List[str]):
        super().__init__()
        self.log_list = log_list

    def emit(self, record):
        msg = self.format(record)
        self.log_list.append(msg)
