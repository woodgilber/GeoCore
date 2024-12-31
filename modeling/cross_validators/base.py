import logging
from abc import ABC, abstractclassmethod
from typing import Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd

from modeling.utils.metrics import decile_table, get_pr, get_roc

logger = logging.getLogger(__name__)


class BaseCV(ABC):
    """
    Abstract base class for cross validators.
    It implements routines that are expected for all
    future cross validators s.t. they properly work with the
    main training code, save artifacts for mlflow etc.
    """

    REQUIRED_COLUMNS = None
    REQUIRED_CONFIG_FIELDS = None
    DOWNSAMPLE_FACTOR = 1

    def __init__(self, df: pd.DataFrame, config: dict, train_labels: Optional[pd.Series] = None) -> None:
        """
        Args:
            df (pd.DataFrame): train + validation dataframe metadata
            config (dict): cross validator config
            train_labels (pd.Series): training labels

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        if (
            not all([column in df.columns.tolist() for column in self.REQUIRED_COLUMNS])
            and self.REQUIRED_COLUMNS is not None
        ):
            logger.exception(
                f"The required columns {self.REQUIRED_COLUMNS} were not found in the data frame {df.columns}!"
            )
            raise ValueError
        else:
            self.df = df

        if (
            not all([key in config.keys() for key in self.REQUIRED_CONFIG_FIELDS])
            and self.REQUIRED_CONFIG_FIELDS is not None
        ):
            logger.exception(f"The {self.REQUIRED_CONFIG_FIELDS} were not found in the passed config {config.keys()}!")
            raise ValueError
        else:
            self.config = config

        if self.config.downsample_factor:
            self.DOWNSAMPLE_FACTOR = self.config.downsample_factor

        self.train_labels = train_labels
        # Pre-compute the folds:
        self.compute_folds()
        for i, fold in enumerate(self.computed_folds):
            logger.info(
                f"Fold {i} size {len(fold['flags'])}. Train size: {len(fold['train_indices'])}. Validation size: "
                f"{len(fold['validation_indices'])}."
            )
        self.downsampled_folds = self.computed_folds[:: self.DOWNSAMPLE_FACTOR]

    def compute_folds(self):
        """
        Should compute a dictionary with the folds
        """
        return NotImplementedError

    @abstractclassmethod
    def fold_assign(self):
        """
        Assign data to the folds (like AUC, N Train data etc.)
        """
        return NotImplementedError

    @abstractclassmethod
    def save_fold_plot(self):
        return NotImplementedError

    def split(self, X: pd.DataFrame, y: np.ndarray = None, groups: np.ndarray = None):
        for fold in self.computed_folds:
            yield fold["train_indices"], fold["validation_indices"]

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.config["n_folds"]

    def evaluate_folds(self, fold_evaluation: List[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Same accross all classifiers

        Get Data Frames for train/validation ROC/PR curves accross all folds

        """
        # get train/val ROC
        train_roc_dfs = pd.concat(
            [get_roc(f["train_probabilities"], f["train_true"], f["label"])[0] for f in fold_evaluation]
        )
        validation_roc_dfs = pd.concat(
            [get_roc(f["validation_probabilities"], f["validation_true"], f["label"])[0] for f in fold_evaluation]
        )

        # get train/val PR
        train_pr_dfs = pd.concat(
            [get_pr(f["train_probabilities"], f["train_true"], f["label"])[0] for f in fold_evaluation]
        )
        validation_pr_dfs = pd.concat(
            [get_pr(f["validation_probabilities"], f["validation_true"], f["label"])[0] for f in fold_evaluation]
        )

        # get train/val lift / gain
        train_lift = []
        val_lift = []
        train_gain = []
        val_gain = []
        for f in fold_evaluation:
            train_pl = decile_table(f["train_true"], f["train_probabilities"])
            val_pl = decile_table(f["validation_true"], f["validation_probabilities"])
            train_lift.append(
                pd.DataFrame(
                    {
                        "DECILE": train_pl.decile.values,
                        "CUM_LIFT": train_pl.lift.values,
                        "LIFT": train_pl.cnt_resp.values / train_pl.cnt_resp_rndm.values,
                        "LABEL": f["label"],
                    }
                )
            )
            train_gain.append(
                pd.DataFrame(
                    {
                        "DECILE": np.append(0, train_pl.decile.values),
                        "RESP_PCT_MOD": np.append(0, train_pl.cum_resp_pct.values),
                        "RESP_PCT_WIZ": np.append(0, train_pl.cum_resp_pct_wiz.values),
                        "LABEL": f["label"],
                    }
                )
            )
            val_lift.append(
                pd.DataFrame(
                    {
                        "DECILE": val_pl.decile.values,
                        "CUM_LIFT": val_pl.lift.values,
                        "LIFT": val_pl.cnt_resp.values / val_pl.cnt_resp_rndm.values,
                        "LABEL": f["label"],
                    }
                )
            )
            val_gain.append(
                pd.DataFrame(
                    {
                        "DECILE": np.append(0, val_pl.decile.values),
                        "RESP_PCT_MOD": np.append(0, val_pl.cum_resp_pct.values),
                        "RESP_PCT_WIZ": np.append(0, val_pl.cum_resp_pct_wiz.values),
                        "LABEL": f["label"],
                    }
                )
            )
        train_lift = pd.concat(train_lift)
        val_lift = pd.concat(val_lift)
        train_gain = pd.concat(train_gain)
        val_gain = pd.concat(val_gain)

        # Crappy implementation, needs a redo:
        PR_TRAIN_AUC = pd.DataFrame({"AUC": [float(f.split("AUC = ")[1]) for f in train_pr_dfs.LABEL.unique()]})
        ROC_TRAIN_AUC = pd.DataFrame({"AUC": [float(f.split("AUC = ")[1]) for f in train_roc_dfs.LABEL.unique()]})
        PR_VALID_AUC = pd.DataFrame({"AUC": [float(f.split("AUC = ")[1]) for f in validation_pr_dfs.LABEL.unique()]})
        ROC_VALID_AUC = pd.DataFrame({"AUC": [float(f.split("AUC = ")[1]) for f in validation_roc_dfs.LABEL.unique()]})

        def mlflow_logging(df, prefix):
            mlflow.log_metric(f"cv_min_{prefix}_auc", df.AUC.min())
            mlflow.log_metric(f"cv_mean_{prefix}_auc", df.AUC.mean())
            mlflow.log_metric(f"cv_max_{prefix}_auc", df.AUC.max())
            mlflow.log_metric(f"cv_std_{prefix}_auc", df.AUC.std())

        mlflow_logging(PR_TRAIN_AUC, "pr_train_")
        mlflow_logging(ROC_TRAIN_AUC, "roc_train_")
        mlflow_logging(PR_VALID_AUC, "pr_validation_")
        mlflow_logging(ROC_VALID_AUC, "roc_validation_")

        return {
            "train_roc_dfs": train_roc_dfs,
            "train_pr_dfs": train_pr_dfs,
            "validation_roc_dfs": validation_roc_dfs,
            "validation_pr_dfs": validation_pr_dfs,
            "train_lift": train_lift,
            "val_lift": val_lift,
            "train_gain": train_gain,
            "val_gain": val_gain,
        }
