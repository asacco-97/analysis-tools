from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging
import yaml
import time
from typing import Any, Dict, Tuple, Optional, Iterable, List, Callable
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


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""

    search_space: Dict[str, Any] = field(default_factory=dict)
    n_iter: int = 10
    objective_func: Optional[Callable[[pd.Series, np.ndarray], float]] = None
    output_tuning: bool = False
    tuning_file: str | None = None


@dataclass
class GBMModelTrainerConfig:
    actual_col: str
    predicted_col: str
    output_dir: str = "outputs"
    log_file: str = "training.log"
    report_file: str = "model_analysis.html"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    callbacks: List[Callable] = field(default_factory=list)
    output_log: bool = True
    base_margin_col: str | None = None

    # Params for Model Analysis Report if outputting
    output_report: bool = False
    report_params: Dict[str, Any] = field(default_factory=dict)
    plots_to_add: List[Dict[str, Any]] = field(default_factory=list)
    tabulate_vars: List[str] = field(default_factory=list)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    callbacks: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "name": "log_eval",
                "period": 50
            },
            {
                "name": "early_stopping", 
                "stopping_rounds": 30
            },
        ]
    )


class GBMModelTrainer:
    def __init__(
        self,
        model_class: Any,
        config_path: str,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        holdout_df: pd.DataFrame | None = None,
        *,
        callbacks: Optional[List[Callable]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model_class = model_class
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.train_df = train_df
        self.valid_df = valid_df
        self.holdout_df = holdout_df
        self.model = None
        self.log_lines: list[str] = []
        self.callbacks = callbacks or self.config.callbacks
        self.logger = logger

    def _load_config(self, config_path: str) -> GBMModelTrainerConfig:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        # Parse tuning config and search space
        if "tuning" in cfg and "search_space" in cfg["tuning"]:
            raw_space = cfg["tuning"]["search_space"]
            cfg["tuning"]["search_space"] = parse_search_space(raw_space)
            cfg["tuning"] = TuningConfig(**cfg["tuning"])

        return GBMModelTrainerConfig(**cfg)

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

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_lines.append(f"[{timestamp}] {message}")
        if hasattr(self, "logger") and self.logger:
            self.logger.info(message)

    def _write_log(self) -> str:
        local_path, s3_path = self._resolve_output_path(self.config.log_file)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.log_lines))

        if s3_path:
            upload_file_to_s3(local_path, s3_path)
            self._log(f"Log uploaded to {s3_path}")

        return str(local_path)

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
        num_round = int(params.pop("num_boost_round", params.pop("n_estimators", 100)))

        dtrain = DMatrix(X_train, label=y_train, base_margin=base_margin_train)
        evals = []
        if y_valid is not None:
            dvalid = DMatrix(
                X_valid,
                label=y_valid,
                base_margin=base_margin_valid,
            )
            evals = [(dvalid, "validation")]

        self.model = xgb_train(
            params,
            dtrain,
            num_boost_round=num_round,
            evals=evals,
            callbacks=self.callbacks,
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

        tuning_cfg = self.config.tuning
        raw_space = search_ranges or tuning_cfg.search_space
        if not raw_space:
            self._log("No tuning search space provided; skipping tuning")
            return pd.DataFrame()

        search_space = parse_search_space(raw_space)

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
            preds = self.model.predict(
                DMatrix(X_valid, base_margin=base_margin_valid)
            )
            if tuning_cfg.objective_func is None:
                score = float(np.mean((y_valid - preds) ** 2))
            else:
                score = float(tuning_cfg.objective_func(y_valid, preds))
            self._log(f"Trial {objective.counter}: score={score:.5f} params={params}")
            objective.counter += 1
            return score

        objective.counter = 1
        result = gp_minimize(objective, dimensions, n_calls=tuning_cfg.n_iter)
        best_params = {name: val for name, val in zip(param_names, result.x)}
        self.config.hyperparameters.update(best_params)

        tuning_df = pd.DataFrame({"params": result.x_iters, "score": result.func_vals})

        if tuning_cfg.output_tuning and tuning_cfg.tuning_file:
            tuning_local_path, tuning_s3_path = self._resolve_output_path(
                tuning_cfg.tuning_file
            )
            tuning_df.to_csv(tuning_local_path, index=False)
            self._log(f"Tuning DF saved to {tuning_local_path.resolve()}")
            if tuning_s3_path:
                upload_file_to_s3(tuning_local_path, tuning_s3_path)
                self._log(f"Tuning DF uploaded to {tuning_s3_path}")

        return tuning_df

    def train(self) -> None:
        """Main training routine."""
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
        if self.holdout_df is not None and not self.holdout_df.empty:
            (
                X_holdout,
                y_holdout,
                base_margin_holdout,
            ) = self._split_xy(
                self._prepare_dataframe(self.holdout_df)
            )
        
        self._fit_model(
            X_train,
            y_train,
            X_valid,
            y_valid,
            base_margin_train,
            base_margin_valid,
        )
        fit_time = time.time() - start

        self._log(f"Training completed in {fit_time:.4f} seconds")
        self._log("Model: XGBoost Booster")
        self._log(f"Hyperparameters: {self.config.hyperparameters}")

        n_rows, n_cols = self.train_df.shape
        self._log(f"Training data shape: {self.train_df.shape}")
        self._log(f"Validation data shape: {self.valid_df.shape}")
        self._log(
            f"Number of predictors: {n_cols - 1 if self.config.actual_col in self.train_df.columns else n_cols}"
        )
        self._log(f"Target column: '{self.config.actual_col}'")
        self._log(f"Prediction column: '{self.config.predicted_col}'")
        if y_valid is not None:
            preds_valid = self.model.predict(
                DMatrix(X_valid, base_margin=base_margin_valid)
            )
            score = float(np.mean((y_valid - preds_valid) ** 2))
            self._log(f"Validation MSE: {score:.4f}")

        # -------------------------------------------------------
        if self.config.output_report:
            actual_col = self.config.actual_col
            pred_col = self.config.predicted_col

            # Predict on train and validation
            train_preds = self.model.predict(
                DMatrix(X_train, base_margin=base_margin_train)
            )
            valid_preds = self.model.predict(
                DMatrix(X_valid, base_margin=base_margin_valid)
            )

            # Assign predictions
            train_df_with_preds = self.train_df.copy()
            valid_df_with_preds = self.valid_df.copy()
            train_df_with_preds[pred_col] = train_preds
            valid_df_with_preds[pred_col] = valid_preds

            # Add split for reporting
            train_df_with_preds["split"] = "T"
            valid_df_with_preds["split"] = "V"

            if self.holdout_df is not None and not self.holdout_df.empty:
                holdout_preds = self.model.predict(
                    DMatrix(X_holdout, base_margin=base_margin_holdout)
                )
                holdout_df_with_preds = self.holdout_df.copy()
                holdout_df_with_preds[pred_col] = holdout_preds
                holdout_df_with_preds["split"] = "H"

                # Concatenate in same order as used for predictions
                df = pd.concat(
                    [train_df_with_preds, valid_df_with_preds, holdout_df_with_preds],
                    axis=0,
                ).reset_index(drop=True)
            else:
                # Concatenate in same order as used for predictions
                df = pd.concat(
                    [train_df_with_preds, valid_df_with_preds], axis=0
                ).reset_index(drop=True)

            mar = ModelAnalysisReport(
                df,
                actual_col=actual_col,
                predicted_col=pred_col,
                **self.config.report_params,
            )
            if self.config.tabulate_vars:
                mar.add_tabulation(variables=self.config.tabulate_vars)

            for plot_cfg in self.config.plots_to_add:
                func_name = plot_cfg.get("plot")
                title = plot_cfg.get("title", func_name)
                kwargs = plot_cfg.get("kwargs", {})

                try:
                    plot_func = getattr(plots, func_name)
                    mar.add_plot(plot_func, title, **kwargs)
                    self._log(f"✔️ Added plot: {title} ({func_name})")
                except Exception as e:
                    self._log(f"⚠️ Failed to add plot '{func_name}': {e}")

            # Save locally and (optionally) to S3
            report_local_path, report_s3_path = self._resolve_output_path(
                self.config.report_file
            )
            report_local_path.parent.mkdir(parents=True, exist_ok=True)
            mar.save(report_local_path)
            self._log(f"Analysis report saved to {report_local_path.resolve()}")

            if report_s3_path:
                upload_file_to_s3(report_local_path, report_s3_path)
                self._log(f"Analysis report uploaded to {report_s3_path}")

        # Save a copy of the config
        config_local_path, config_s3_path = self._resolve_output_path("config.yaml")
        shutil.copy(self.config_path, config_local_path)
        self._log(f"Saved config to {config_local_path}")

        if config_s3_path:
            upload_file_to_s3(config_local_path, config_s3_path)
            self._log(f"Config uploaded to {config_s3_path}")

        # -------------------------------------------------------
        log_file = None
        if self.config.output_log:
            log_file = self._write_log()


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
