import logging
import pickle
from typing import List

import mlflow
import pandas as pd
from sklearn.base import BaseEstimator

from modeling.config import get_cfg

# Set up logging
logger = logging.getLogger("modeling")


def log_model(model: BaseEstimator) -> None:
    """Log the model as pkl file in mlflow

    Args:
        model (BaseEstimator): model instance
    """
    with open("./runs/model.pkl", "wb") as f:
        pickle.dump(model, f)
    mlflow.log_artifact("./runs/model.pkl")


def log_config_datasets() -> None:
    """Log artifacts"""
    mlflow.log_artifact("./runs/config.yaml")
    mlflow.log_artifact("./runs/features_train.pq", artifact_path="data")
    mlflow.log_artifact("./runs/labels_train.pq", artifact_path="data")
    mlflow.log_artifact("./runs/meta_columns_train.pq", artifact_path="data")
    mlflow.log_artifact("./runs/train_predictions.pq", artifact_path="data")


def validate_configs(run_ids: List[str]) -> None:
    """Look at the labels used by the config to train the meta model and check the base models have been trained on
    the same labelset

    Args:
        run_ids (List[str]): list of mlflow run ids
    """
    for i, run_id in enumerate(run_ids):
        # load config
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="config.yaml")
        cfg = get_cfg()
        cfg.merge_from_file(path)

        # get labels
        if i == 0:
            labels = cfg["sql_parameters"]["labels"]
        else:
            if set(labels) != set(cfg["sql_parameters"]["labels"]):
                logger.warning("The runs have been trained on different labels.")


def create_meta_labels(run_ids: List[str]) -> pd.DataFrame:
    """Pull training scores from a list of run ids and concatenate them

    Args:
        run_ids (List[str]): list of mlflow run ids

    Returns:
        pd.DataFrame: concatenated scores to use as training data
    """
    labels = []
    for run_id in run_ids:
        path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="data/train_predictions.pq")
        df = pd.read_parquet(path)
        run = mlflow.get_run(run_id)
        run_name = run.info.run_name
        df = df.rename(columns={"SCORES": run_name})
        labels.append(df)
    return pd.concat(labels, axis=1)
