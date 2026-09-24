import os
import sys
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
import mlflow

# Ensure repo root and src root are available in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.climate_ml.src.trainer import ModelTrainer, MLflowTracker
from src.climate_ml.src.evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Orchestrates the end-to-end training and evaluation workflow within a single MLflow run context."""

    def __init__(self, config: DictConfig):
        self.config = config

    def run(self) -> None:
        logger.info(f"Starting Training Pipeline for project: {self.config.project_name}")
        logger.info(f"Active Experiment: {self.config.experiment_name}")

        mlflow.set_experiment(self.config.experiment_name)

        # Wrap both phases inside a single cohesive MLflow run
        run_name = f"{self.config.model.model_name}_{self.config.experiment_name}"
        with mlflow.start_run(run_name=run_name) as active_run:
            run_id = active_run.info.run_id
            logger.info(f"Active MLflow Run ID: {run_id}")

            # Phase 1: Model Training & Artifact Logging
            logger.info("--- Phase 1: Model Training & Tracking ---")
            trainer = ModelTrainer(self.config)
            train_metrics = trainer.train_and_evaluate()

            tracker = MLflowTracker(self.config.experiment_name)
            tracker.log_run(
                config=self.config,
                metrics=train_metrics,
                model=trainer.model,
                X_train=trainer.X_train,
                y_train=trainer.y_train,
                run_id=run_id
            )
            logger.info("Phase 1 completed successfully.")

            # Phase 2: Model Evaluation on Test Set
            logger.info("--- Phase 2: Model Evaluation & Test Metrics Logging ---")
            evaluator = ModelEvaluator(self.config)
            evaluator.evaluate(
                model=trainer.model,
                X_test=trainer.X_test,
                y_test=trainer.y_test,
                run_id=run_id
            )
            logger.info("Phase 2 completed successfully.")

            logger.info(f"Training pipeline executed end-to-end successfully! Run ID: {run_id}")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    pipeline = TrainingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()