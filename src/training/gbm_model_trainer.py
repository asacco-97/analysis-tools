from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging
import yaml
import json
import time
import smtplib
import inspect
from email.message import EmailMessage
from typing import Any, Dict, Tuple, Optional, Iterable, List
import pandas as pd

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from analysis.report import ModelAnalysisReport
from analysis import plots

@dataclass
class GBMModelTrainerConfig:
    actual_col: str
    predicted_col: str
    log_file: str = "training.log"
    report_output_path: str = "model_analysis.html"
    email: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    output_log: bool = True
    output_email: bool = False

    # Params for Model Analysis Report if outputting
    output_report: bool = False
    report_params: Dict[str, Any] = field(default_factory=dict)
    plots_to_add: List[Dict[str, Any]] = field(default_factory=list) 
    tabulate_vars: List[str] = field(default_factory=list)  

class GBMModelTrainer:
    def __init__(
        self,
        model_class: Any,
        config_path: str,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        holdout_df: pd.DataFrame = None
    ) -> None:
        self.model_class = model_class
        self.config = self._load_config(config_path)
        self.train_df = train_df
        self.valid_df = valid_df
        self.holdout_df = holdout_df
        self.model = None
        self.log_lines: list[str] = []

    def _load_config(self, config_path: str) -> GBMModelTrainerConfig:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return GBMModelTrainerConfig(**cfg)

    def _split_xy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series | None]:
        actual_col = self.config.actual_col
        if actual_col and actual_col in df.columns:
            X = df.drop(columns=[actual_col])
            y = df[actual_col]
            return X, y
        return df, None

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure object columns are cast to categorical if using XGBoost."""
        df_clean = df.copy()
        model_name = self.model_class.__name__
        
        if "XGB" in model_name or "LGBM" in model_name:
            for col in df_clean.select_dtypes(include=["object", "string"]).columns:
                df_clean[col] = df_clean[col].astype("category")
        return df_clean

    def _ensure_parent_dir(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log_lines.append(f"[{timestamp}] {message}")

    def _write_log(self) -> str:
        log_path = self._ensure_parent_dir(self.config.log_file)
        with open(log_path, "w") as f:
            f.write("\n".join(self.log_lines))
        return log_path

    def _send_email(self, log_file: str) -> None:
        email_to = self.config.email
        if not email_to:
            return

        msg = EmailMessage()
        msg["Subject"] = "Model Training Log"
        msg["From"] = "noreply@modeltraining.com"
        msg["To"] = email_to
        with open(log_file) as f:
            msg.set_content(f.read())

        try:
            with smtplib.SMTP("localhost") as server:
                server.send_message(msg)
            self._log(f"Sent log to {email_to}")
        except Exception as exc:
            self._log(f"Failed to send email: {exc}")

    def _accepts_kwargs(self, cls) -> bool:
        sig = inspect.signature(cls.__init__)
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    
    def _filter_valid_kwargs(self, cls, candidate_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter kwargs to only those accepted by the model's __init__ method."""
        valid_keys = inspect.signature(cls.__init__).parameters
        valid_kwargs = {}
        for k, v in candidate_kwargs.items():
            if k in valid_keys:
                valid_kwargs[k] = v
            else:
                self._log(
                    f"⚠️ '{k}' is not a valid hyperparameter for {cls.__name__} and will be ignored. "
                )
        return valid_kwargs

    def _instantiate_model(self) -> Any:
        raw_params = dict(self.config.hyperparameters or {})
        
        if "XGB" in self.model_class.__name__:
            raw_params["enable_categorical"] = True
            model_params = raw_params

        else:
            # Model-specific defaults
            if "LGBM" in self.model_class.__name__:
                raw_params["verbose"] = False

            # Filter model parameters for CATBoost and LightGBM
            model_params = self._filter_valid_kwargs(self.model_class, raw_params)

        model = self.model_class(**model_params)
        return model

    def _fit_model(self, X_train, y_train, X_valid, y_valid) -> None:
        fit_kwargs = {}
        
        # Detect cat features for CatBoost
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        if isinstance(self.model, (CatBoostClassifier, CatBoostRegressor)):
            fit_kwargs["cat_features"] = cat_cols
            fit_kwargs["verbose"] = False
            
        fit_sig = inspect.signature(self.model.fit)

        # Eval set
        if "eval_set" in fit_sig.parameters:
            eval_pair = (X_valid, y_valid) if y_valid is not None else (X_valid,)
            fit_kwargs["eval_set"] = [eval_pair]

        # Special handling
        if isinstance(self.model, (XGBClassifier, XGBRegressor)):
            fit_kwargs["verbose"] = False
        
        # Fit
        if y_train is not None:
            self.model.fit(X_train, y_train, **fit_kwargs)
        else:
            self.model.fit(X_train, **fit_kwargs)

    def train(self) -> None:
        """Main training routine."""
        start = time.time()
        X_train, y_train = self._split_xy(self._prepare_dataframe(self.train_df))
        X_valid, y_valid = self._split_xy(self._prepare_dataframe(self.valid_df))
        if self.holdout_df:
            X_holdout, y_holdout = self._split_xy(self._prepare_dataframe(self.holdout_df))

        self.model = self._instantiate_model()
        self._log(f"Instantiated model: {self.model.__class__.__name__}")

        self._fit_model(X_train, y_train, X_valid, y_valid)
        fit_time = time.time() - start

        self._log(f"Training completed in {fit_time:.4f} seconds")
        self._log(f"Model: {self.model.__class__.__name__}")
        self._log(f"Hyperparameters: {self.config.hyperparameters}")
        
        n_rows, n_cols = self.train_df.shape
        self._log(f"Training data shape: {self.train_df.shape}")
        self._log(f"Validation data shape: {self.valid_df.shape}")
        self._log(f"Number of predictors: {n_cols - 1 if self.config.actual_col in self.train_df.columns else n_cols}")
        self._log(f"Target column: '{self.config.actual_col}'")
        self._log(f"Prediction column: '{self.config.predicted_col}'")
        if y_valid is not None:
            score = self.model.score(X_valid, y_valid)
            self._log(f"Validation score (default metric): {score:.4f}")

        # -------------------------------------------------------
        if self.config.output_report:
            actual_col = self.config.actual_col
            pred_col = self.config.predicted_col

            # Predict on train and validation
            if hasattr(self.model, "predict_proba"):
                train_preds = self.model.predict_proba(X_train)[:, 1]
                valid_preds = self.model.predict_proba(X_valid)[:, 1]
            else:
                train_preds = self.model.predict(X_train)
                valid_preds = self.model.predict(X_valid)
            
            # Assign predictions
            train_df_with_preds = self.train_df.copy()
            valid_df_with_preds = self.valid_df.copy()
            train_df_with_preds[pred_col] = train_preds
            valid_df_with_preds[pred_col] = valid_preds

            # Add split for reporting
            train_df_with_preds["split"] = "T"
            valid_df_with_preds["split"] = "V"
            
            if self.holdout_df:
                if hasattr(self.model, "predict_proba"):
                    holdout_preds = self.model.predict_proba(X_holdout)[:, 1]
                else:
                    holdout_preds = self.model.predict(X_holdout)
                holdout_df_with_preds = self.valid_df.copy()
                holdout_df_with_preds[pred_col] = holdout_preds
                holdout_df_with_preds["split"] = "H"

                # Concatenate in same order as used for predictions
                df = pd.concat([train_df_with_preds, valid_df_with_preds, holdout_df_with_preds], axis=0).reset_index(drop=True)
            else:   
                # Concatenate in same order as used for predictions
                df = pd.concat([train_df_with_preds, valid_df_with_preds], axis=0).reset_index(drop=True)

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
            
            report_path = self._ensure_parent_dir(self.config.report_output_path)
            mar.save(report_path)
            self._log(f"Analysis report saved to {report_path}")

        # -------------------------------------------------------
        log_file = None
        if self.config.output_log:
            log_file = self._write_log()

        if self.config.output_email and log_file:
            self._send_email(log_file)

