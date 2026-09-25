import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Callable, Type, TypedDict, Optional, cast
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.xgboost as mlflow_xgb
import mlflow.lightgbm as mlflow_lgb
import mlflow.sklearn as mlflow_sklearn
from mlflow.models import infer_signature

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def resolve_data_path(data_dir: str, processed_file: str) -> str:
    """Robustly resolves dataset path regardless of current working directory or Hydra override."""
    candidate_path = os.path.join(data_dir, processed_file)
    if os.path.exists(candidate_path):
        return os.path.abspath(candidate_path)

    # Check relative to src/climate_ml/
    climate_ml_dir = Path(__file__).resolve().parent.parent
    candidate_climate = climate_ml_dir / data_dir / processed_file
    if candidate_climate.exists():
        return str(candidate_climate)

    # Check relative to repo root
    repo_root = climate_ml_dir.parent.parent
    candidate_root = repo_root / data_dir / processed_file
    if candidate_root.exists():
        return str(candidate_root)

    return os.path.abspath(candidate_path)


class DataPreprocessor:
    """Responsible for loading, cleaning, imputing, and splitting data for training and evaluation."""

    def __init__(self, data_path: str, target_column: str, test_size: float = 0.2, random_state: int = 42):
        self.data_path = data_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names: list[str] = []
        self.imputer_means: Optional[pd.Series] = None

    def load_and_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logger.info(f"Loading dataset from: {self.data_path}")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}. Run ingestion pipeline first.")

        df = pd.read_csv(self.data_path)
        
        # 1. Target column verification - fail early and clearly, avoid silent fallbacks
        if self.target_column not in df.columns:
            raise KeyError(
                f"Configured target column '{self.target_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        # 2. Drop rows with missing target values to ensure ground truth integrity
        df = df.dropna(subset=[self.target_column])

        # 3. Separate features and target
        X = df.drop(columns=[self.target_column, 'district', 'grid_id'], errors='ignore')
        y = df[self.target_column]

        # 4. Filter only numeric features
        X = X.select_dtypes(include=[np.number])
        self.feature_names = list(X.columns)

        # 5. Split train/test BEFORE computing imputation statistics to prevent data leakage
        logger.info(f"Splitting data with test size: {self.test_size}, random state: {self.random_state}")
        X_train, X_test, y_train, y_test = cast(
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
            train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        )

        # 6. Fit imputer on X_train only and apply to both train and test
        self.imputer_means = cast(pd.Series, X_train.mean(numeric_only=True))
        X_train = X_train.fillna(self.imputer_means)
        X_test = X_test.fillna(self.imputer_means)

        return X_train, X_test, y_train, y_test


ModelRegistryEntry = TypedDict(
    "ModelRegistryEntry",
    {
        "class": Type[Any],
        "params_extractor": Callable[[DictConfig], Dict[str, Any]],
    },
)


class DynamicModelBuilder:
    """Responsible for dynamically instantiating any machine learning model based on config."""

    REGISTRY: Dict[str, ModelRegistryEntry] = {
        "xgboost": {
            "class": xgb.XGBRegressor,
            "params_extractor": lambda cfg: {
                "n_estimators": cfg.model.xgboost.n_estimators,
                "max_depth": cfg.model.xgboost.max_depth,
                "learning_rate": cfg.model.xgboost.learning_rate,
                "subsample": cfg.model.xgboost.subsample,
                "colsample_bytree": cfg.model.xgboost.colsample_bytree,
                "random_state": cfg.model.random_state
            }
        },
        "lightgbm": {
            "class": lgb.LGBMRegressor,
            "params_extractor": lambda cfg: {
                "n_estimators": cfg.model.lightgbm.n_estimators,
                "max_depth": cfg.model.lightgbm.max_depth,
                "num_leaves": cfg.model.lightgbm.num_leaves,
                "learning_rate": cfg.model.lightgbm.learning_rate,
                "subsample": cfg.model.lightgbm.subsample,
                "random_state": cfg.model.random_state,
                "verbose": -1
            }
        },
        "random_forest": {
            "class": RandomForestRegressor,
            "params_extractor": lambda cfg: {
                "n_estimators": cfg.model.random_forest.n_estimators,
                "max_depth": None if cfg.model.random_forest.max_depth == "null" else cfg.model.random_forest.max_depth,
                "min_samples_split": cfg.model.random_forest.min_samples_split,
                "min_samples_leaf": cfg.model.random_forest.min_samples_leaf,
                "random_state": cfg.model.random_state
            }
        }
    }

    @classmethod
    def get_model(cls, config: DictConfig) -> Any:
        model_name = config.model.model_name.lower()
        logger.info(f"Dynamically building model architecture: {model_name}")

        if model_name not in cls.REGISTRY:
            raise ValueError(f"Unsupported model name: '{model_name}'. Available models: {list(cls.REGISTRY.keys())}")

        model_info = cls.REGISTRY[model_name]
        model_class = model_info["class"]
        params = model_info["params_extractor"](config)

        return model_class(**params)


class ModelTrainer:
    """Responsible for orchestrating model training and evaluation metrics logging."""

    def __init__(self, config: DictConfig):
        self.config = config
        data_file = resolve_data_path(config.data.data_dir, config.data.processed_dir)
        self.preprocessor = DataPreprocessor(
            data_path=data_file,
            target_column=config.data.target_column,
            test_size=config.data.test_size,
            random_state=config.data.random_state
        )
        self.model = DynamicModelBuilder.get_model(config)
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

    def train_and_evaluate(self) -> Dict[str, float]:
        logger.info(f"Starting training for experiment: {self.config.experiment_name}")
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.load_and_split()

        self.model.fit(self.X_train, self.y_train)
        logger.info("Model training completed successfully.")

        # Compute training metrics
        predictions = self.model.predict(self.X_train)
        mse = mean_squared_error(self.y_train, predictions)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(self.y_train, predictions))
        r2 = float(r2_score(self.y_train, predictions))

        metrics = {
            "train_rmse": rmse,
            "train_mse": float(mse),
            "train_mae": mae,
            "train_r2_score": r2
        }

        logger.info(f"Training Metrics -> RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2 Score: {r2:.4f}")
        return metrics


class MLflowTracker:
    """Responsible for logging parameters, metrics, and models to MLflow."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def log_run(
        self,
        config: DictConfig,
        metrics: Dict[str, float],
        model: Any,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """Logs hyperparameters, metrics, and model artifact with signature to MLflow."""
        active_run = mlflow.active_run()
        if active_run:
            return self._log_artifacts_and_metrics(config, metrics, model, X_train, y_train, active_run.info.run_id)
        elif run_id:
            with mlflow.start_run(run_id=run_id):
                return self._log_artifacts_and_metrics(config, metrics, model, X_train, y_train, run_id)
        else:
            with mlflow.start_run(run_name=f"{config.model.model_name}_run") as run:
                return self._log_artifacts_and_metrics(config, metrics, model, X_train, y_train, run.info.run_id)

    def _log_artifacts_and_metrics(
        self,
        config: DictConfig,
        metrics: Dict[str, float],
        model: Any,
        X_train: Optional[pd.DataFrame],
        y_train: Optional[pd.Series],
        run_id: str,
    ) -> str:
        # Log hyperparameters
        mlflow.log_param("model_name", config.model.model_name)
        mlflow.log_param("seed", config.seed)
        mlflow.log_param("target_column", config.data.target_column)

        # Log model-specific parameters
        model_name = config.model.model_name.lower()
        if hasattr(config.model, model_name):
            for param_key, param_val in getattr(config.model, model_name).items():
                mlflow.log_param(f"{model_name}_{param_key}", param_val)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Prepare signature and input example if data provided
        log_kwargs: Dict[str, Any] = {"artifact_path": "model"}
        if X_train is not None:
            input_example = X_train.head(5)
            log_kwargs["input_example"] = input_example
            try:
                preds = model.predict(input_example)
                log_kwargs["signature"] = infer_signature(input_example, preds)
            except Exception as e:
                logger.warning(f"Could not infer model signature: {e}")

        # Log the trained model artifact
        if "xgboost" in model_name:
            mlflow_xgb.log_model(model, **log_kwargs)
        elif "lightgbm" in model_name:
            mlflow_lgb.log_model(model, **log_kwargs)
        else:
            mlflow_sklearn.log_model(model, **log_kwargs)

        logger.info(f"Run {run_id} successfully logged to MLflow under experiment '{self.experiment_name}'.")
        return run_id


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    logger.info(f"Project: {config.project_name}")
    trainer = ModelTrainer(config)
    metrics = trainer.train_and_evaluate()
    tracker = MLflowTracker(config.experiment_name)
    tracker.log_run(config, metrics, trainer.model, trainer.X_train, trainer.y_train)


if __name__ == "__main__":
    main()