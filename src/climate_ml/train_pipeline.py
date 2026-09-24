import os
import sys
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
import mlflow
from dotenv import load_dotenv

# Load .env from repo root so MLFLOW_TRACKING_URI, AWS credentials, etc. are available
_REPO_ENV = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_REPO_ENV, override=False)

# Fix Windows cp1256 locale crashing on MLflow's emoji-containing run URL output
import sys as _sys
import io as _io
if hasattr(_sys.stdout, 'buffer'):
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
    _sys.stderr = _io.TextIOWrapper(_sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# Ensure repo root and src root are available in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from src.trainer import ModelTrainer, MLflowTracker
    from src.evaluation import ModelEvaluator
except ImportError:
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

        # Apply remote MLflow tracking URI from env (DagsHub) if configured
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        else:
            logger.warning("MLFLOW_TRACKING_URI not set — logging to local mlruns/")

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