import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from mlflow import MlflowClient
import mlflow.pyfunc

# Ensure repo root is available in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import DataPreprocessor and path resolver from trainer (reusable, single source of truth)
from src.climate_ml.src.trainer import DataPreprocessor, resolve_data_path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Responsible for loading a trained model and evaluating its performance on test data."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.data_path = resolve_data_path(config.data.data_dir, config.data.processed_dir)
        self.preprocessor = DataPreprocessor(
            data_path=self.data_path,
            target_column=config.data.target_column,
            test_size=config.data.test_size,
            random_state=config.data.random_state
        )

    def load_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Reuses DataPreprocessor to obtain the exact test set without data leakage."""
        _, X_test, _, y_test = self.preprocessor.load_and_split()
        return X_test, y_test

    @staticmethod
    def load_latest_model_from_mlflow(experiment_name: str, run_id: Optional[str] = None) -> Tuple[Any, str]:
        """Locates the latest run (or specific run_id) in MLflow and loads the saved model artifact."""
        client = MlflowClient()

        if run_id:
            logger.info(f"Loading model artifact from specified run_id: {run_id}")
            model_uri = f"runs:/{run_id}/model"
            return mlflow.pyfunc.load_model(model_uri), run_id

        logger.info(f"Searching for the latest model in MLflow experiment: {experiment_name}")
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        if not runs:
            raise ValueError(f"No runs found for experiment '{experiment_name}'.")

        latest_run_id = runs[0].info.run_id
        model_uri = f"runs:/{latest_run_id}/model"
        logger.info(f"Loading model artifact from URI: {model_uri}")
        return mlflow.pyfunc.load_model(model_uri), latest_run_id

    def evaluate(
        self,
        model: Optional[Any] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        run_id: Optional[str] = None
    ) -> Dict[str, float]:
        """Performs evaluation on the test set and logs metrics to MLflow."""
        # 1. Obtain test data if not passed in-memory
        if X_test is None or y_test is None:
            X_test, y_test = self.load_test_data()

        # 2. Obtain model if not passed in-memory
        if model is None:
            loaded_model, loaded_run_id = self.load_latest_model_from_mlflow(self.config.experiment_name, run_id=run_id)
            model = loaded_model
            if run_id is None:
                run_id = loaded_run_id

        if model is None:
            raise ValueError("Model could not be loaded for evaluation.")

        logger.info("Generating predictions on the test set...")
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))

        metrics = {
            "test_rmse": rmse,
            "test_mse": float(mse),
            "test_mae": mae,
            "test_r2_score": r2
        }

        logger.info("Evaluation Complete:")
        for metric_name, value in metrics.items():
            logger.info(f" -> {metric_name.upper()}: {value:.4f}")

        # 3. Log evaluation metrics to MLflow
        active_run = mlflow.active_run()
        if active_run:
            mlflow.log_metrics(metrics)
            logger.info(f"Evaluation metrics logged directly to active MLflow run: {active_run.info.run_id}")
        elif run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(metrics)
            logger.info(f"Evaluation metrics successfully logged to MLflow run {run_id}.")
        else:
            mlflow.set_experiment(self.config.experiment_name)
            with mlflow.start_run(run_name="evaluation_run"):
                mlflow.log_metrics(metrics)
            logger.info("Evaluation metrics logged to new evaluation run.")

        return metrics


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    logger.info(f"Starting evaluation phase for project: {config.project_name}")
    evaluator = ModelEvaluator(config)
    evaluator.evaluate()


if __name__ == "__main__":
    main()