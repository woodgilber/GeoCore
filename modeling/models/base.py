import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from fvcore.common.config import CfgNode
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from skopt import BayesSearchCV
from snowflake.connector.connection import SnowflakeConnection

from modeling.cross_validators.build import build_cv

# add in queries to use for test metrics
# TODO: make this dynamic
from modeling.queries import INGENIOUS_HOT_SPRINGS
from modeling.utils.data import preprocess_data
from modeling.utils.metrics import decile_table, get_pr, get_roc
from modeling.utils.plotting import plot_fold_metrics, plot_test_data

logger = logging.getLogger(__name__)


class BaseRegression(ABC):
    """
    Abstract class for Base Regression Model
    """

    model = None
    nan_strategy = None
    preprocessing = None
    verbose = -1

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: pd.DataFrame,
        model_config: CfgNode,
        weights: List = None,
    ):
        """Binary Classifier model.

        Args:
            X (pd.DataFrame): training features
            y (pd.Series): training labels
            weights (List, optional): Label weights. Defaults to None.
            model_config (CfgNode): model config. Defaults to None.
        """
        self.X, self.y, self.metadata = X, y, metadata
        self.X, self.y, self.metadata, _ = preprocess_data(
            X, y, metadata, None, nan_strategy=self.nan_strategy, preprocessing=self.preprocessing
        )
        # placeholder for cross_validator
        self.cross_validator = None
        self.weights = weights

        # place holders
        self.model_config = model_config
        self.model_params = None
        self.optim_params = None
        self._set_params()

        # These are important for optimization:
        self.fold_evaluation = None  # Placeholder

        # Just plain initialization:
        self.regressor = None  # Placeholder for model instance
        self.optimized_params = None  # Placeholder for optimized classifier parameters.

    @abstractmethod
    def _set_params(self):
        """This method needs to set model_params and optim_params"""
        return NotImplementedError

    def build_cross_validator(self, unsupervised: bool, run_folder: str):
        """Build cross validator. We make this a method of the model class because we need the labels cleaned from
        nans.
        """
        self.cross_validator = build_cv(self.model_config, self.metadata, self.y, unsupervised, run_folder)

    def fit(self, unsupervised: bool = False):
        """
        Fits the model on the entire dataset
        """
        params = self.model_params if self.model_params else {}

        if self.optimized_params:
            params.update(self.optimized_params)

        self.regressor = self.model(**params)
        self.regressor.fit(
            self.X,
            self.y.to_numpy(),
            sample_weight=self.weights.to_numpy() if self.weights is not None else None,
        )

    def optimize(self, unsupervised: bool = False):
        """
        Same accross all regressors
        """

        if self.model_params:
            params = self.model_params
        else:
            params = {}

        params["verbose"] = self.verbose

        regressor = self.model(**params)

        fit_params = {"verbose": self.verbose}

        # Add sample weights:
        if self.weights is not None:
            fit_params = {"sample_weight": self.weights.to_numpy()}

        logger.info("Running parameter search with BayesSearchCV.")

        opt = BayesSearchCV(
            regressor,
            self.optim_params["search_spaces"],
            fit_params=fit_params,
            n_iter=self.optim_params["n_iter"] if "n_iter" in self.optim_params.keys() else 10,
            random_state=self.optim_params["random_state"] if "random_state" in self.optim_params.keys() else 0,
            n_jobs=self.optim_params["n_jobs"] if "n_jobs" in self.optim_params.keys() else 10,
            verbose=1,
            refit=False,
            cv=self.cross_validator,
            scoring=self.optim_params["scoring"] if "scoring" in self.optim_params.keys() else "roc_auc",
        )

        opt.fit(self.X, self.y)
        self.optimized_params = dict(opt.best_params_)

        # This assumes that mlflow run had been started by the caller
        mlflow.log_metric("Optimizer Score", opt.best_score_)

        # Fit the model on the entire dataset with the optimized parameters:
        self.fit()

    def predict(self, Xnew: pd.DataFrame):
        """
        Same accross all regressors
        """
        if self.regressor:
            return self.regressor.predict(Xnew)
        else:
            raise ValueError("Regression Model was not fitted! Exiting")

    @abstractmethod
    def _parameter_importance(self, folder: str = None):
        """
        This is classifier specific
        """
        return NotImplementedError

    def importance_plot(self, folder: str):
        """
        Plots parameter importance
        """
        self._parameter_importance(folder)

    @abstractmethod
    def _shapley_plot(self, folder: str):
        """
        Model specific implementation
        """
        return NotImplementedError

    def shapley_plot(self, folder: str):
        self._shapley_plot(folder)

    @abstractmethod
    def _plot_fold_diagnostics(self, folder: str):
        return NotImplementedError

    def plot_fold_diagnostics(self, folder: str):
        self._plot_fold_diagnostics(folder)

    def save_all_stats(self, folder: str = None):
        if folder:
            # Call Importance:
            self.importance_plot(folder)

            # Call Shapley:
            self.shapley_plot(folder)

            # Call fold Diagnostics:
            self.plot_fold_diagnostics(folder)

        else:
            raise ValueError("folder needs to be provided! Exiting!")


class BinaryClassifier(ABC):
    """
    Abstract class for Base Classifier
    """

    # TODO: make these properties?
    model = None
    nan_strategy = None
    preprocessing = None
    verbose = -1

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: pd.DataFrame,
        model_config: CfgNode,
        weights: List = None,
    ):
        """Binary Classifier model.

        Args:
            X (pd.DataFrame): training features
            y (pd.Series): training labels
            weights (List, optional): Label weights. Defaults to None.
            model_config (CfgNode): model config. Defaults to None.
        """
        self.X, self.y, self.metadata = X, y, metadata
        self.X, self.y, self.metadata, _ = preprocess_data(
            X, y.astype(int), metadata, None, nan_strategy=self.nan_strategy, preprocessing=self.preprocessing
        )
        self.weights = weights
        self.model_config = model_config

        # place holders
        self.model_params = None
        self.optim_params = None
        self._set_params()

        # These are important for optimization:
        # Instance since we expect the cross validators to be initialized with expensive computation.
        self.cross_validator = None
        self.fold_evaluation = None  # Placeholder

        # Just plain initialization:
        self.clf = None  # Placeholder for classifier instance
        self.optimized_params = None  # Placeholder for optimized classifier parameters.

    @abstractmethod
    def _set_params(self):
        """This method needs to set model_params and optim_params"""
        return NotImplementedError

    @abstractmethod
    def shapley_plot(self, folder: str = None):
        """
        Plots shaply parameter interpretation plot

        General Shapley plot caller
        """
        return NotImplementedError

    def build_cross_validator(self, unsupervised: bool, run_folder: str):
        """Build cross validator. We make this a method of the model class because we need the labels cleaned from
        nans.
        """
        self.cross_validator = build_cv(self.model_config, self.metadata, self.y, unsupervised, run_folder)

    def fit(self, unsupervised: bool):
        """
        Fits the model on the entire dataset
        """
        params = self.model_params if self.model_params else {}

        if self.optimized_params:
            params.update(self.optimized_params)

        self.clf = self.model(**params)
        if unsupervised:
            self.clf.fit(self.X)
        else:
            self.clf.fit(
                self.X,
                self.y.to_numpy(),
                sample_weight=self.weights.to_numpy() if self.weights is not None else None,
            )
        probabilities = self.clf.predict_proba(self.X)[:, 1]
        pd.DataFrame(index=self.X.index, data={"SCORES": probabilities}).to_parquet("./runs/train_predictions.pq")

    def optimize(self, unsupervised: bool):
        """
        Same accross all classifiers
        """
        # can't run optimization in unsupervised mode
        if not unsupervised:
            if self.model_params:
                params = self.model_params
            else:
                params = {}

            # if we perform feature selection, train a quick model and look at the feature importance
            quantile = self.model_config.feature_threshold
            to_drop = int(self.X.shape[1] * quantile)
            logger.info(f"Removing {to_drop} features")
            if to_drop > 0:
                clf = self.model(**params)
                clf.fit(
                    self.X,
                    self.y.to_numpy(),
                    sample_weight=self.weights.to_numpy() if self.weights is not None else None,
                )
                gain = clf.booster_.feature_importance("gain")
                sorted_gain = np.argsort(gain)
                to_remove = sorted_gain[:to_drop]
                self.X.drop(self.X.columns[to_remove], axis=1, inplace=True)
                logger.info(f"Dataset size after removing low gain features: {self.X.shape}.")

            clf = self.model(**params)

            fit_params = {"verbose": self.verbose}

            # Add sample weights:
            if self.weights is not None:
                fit_params = {"sample_weight": self.weights.to_numpy()}

            logger.info("Running parameter search with BayesSearchCV.")

            self.opt = BayesSearchCV(
                clf,
                self.optim_params["search_spaces"],
                fit_params=fit_params,
                n_iter=self.optim_params["n_iter"] if "n_iter" in self.optim_params.keys() else 10,
                random_state=self.optim_params["random_state"] if "random_state" in self.optim_params.keys() else 0,
                n_jobs=self.optim_params["n_jobs"] if "n_jobs" in self.optim_params.keys() else 10,
                verbose=1,
                refit=False,
                cv=self.cross_validator,
                scoring=self.optim_params["scoring"] if "scoring" in self.optim_params.keys() else "roc_auc",
            )
            self.opt.fit(self.X, self.y.to_numpy())

            pd.DataFrame(self.opt.cv_results_).to_csv("./runs/cv_results.csv")
            self.optimized_params = dict(self.opt.best_params_)

            mlflow.log_metric("Optimizer Score", self.opt.best_score_)
            mlflow.log_artifact("./runs/cv_results.csv")

        # Fit the model on the entire dataset with the optimized parameters:
        self.fit(unsupervised)

    def predict_proba(self, Xnew: pd.DataFrame):
        """
        Same accross all classifiers
        """
        if self.clf:
            return self.clf.predict_proba(Xnew)[:, 1]
        else:
            raise ValueError("Classifier was not fitted! Exiting")

    def importance_plot(self, folder: str = None):
        """
        General parameter importance
        """
        return self._parameter_importance(folder)

    # Helpers for evaluation and statistics plotting: ---------------------------------
    def _predict_folds(self):
        """
        Same across all classifiers
        We just evaluate all folds with the optimal classifier (if fitted).
        """
        self.fold_evaluation = []

        params = self.model_params if self.model_params else {}

        if self.optimized_params:
            params.update(self.optimized_params)

        params["verbose"] = self.verbose

        fl = 0
        for fold in self.cross_validator.downsampled_folds:
            clf = self.model(**params)  # New model each time:
            Xtrain_fold = self.X.to_numpy()[fold["train_indices"]]
            Ytrain_fold = list(self.y.to_numpy()[fold["train_indices"]])
            Xval_fold = self.X.to_numpy()[fold["validation_indices"]]
            Yval_fold = list(self.y.to_numpy()[fold["validation_indices"]])

            clf.fit(
                Xtrain_fold,
                Ytrain_fold,
                sample_weight=(
                    self.weights.to_numpy()[fold["train_indices"]].reshape(-1) if self.weights is not None else None
                ),
            )
            logger.info(
                f"Fold {fl} evaluation: train count = {Xtrain_fold.shape[0]} / validation count = {Xval_fold.shape[0]}"
                f" Train positive/negative: {sum(Ytrain_fold)}/{len(Ytrain_fold) - sum(Ytrain_fold)}"
                f" Val positive/negative: {sum(Yval_fold)}/{len(Yval_fold) - sum(Yval_fold)}"
            )

            train_probs = clf.predict_proba(Xtrain_fold)[:, 1]
            validation_probs = clf.predict_proba(Xval_fold)[:, 1]

            fold_stats = {
                "validation_probabilities": validation_probs,
                "validation_true": self.y.to_numpy()[fold["validation_indices"]],
                "train_probabilities": train_probs,
                "train_true": self.y.to_numpy()[fold["train_indices"]],
                "label": f"FOLD: {fl}; "
                + f"TP: {self.y.to_numpy()[fold['train_indices']].sum()}; "
                + f"TN: {len(fold['train_indices']) - self.y.to_numpy()[fold['train_indices']].sum()}; "
                + f"VP: {self.y.to_numpy()[fold['validation_indices']].sum()}; "
                + f"VN: {len(fold['validation_indices']) - self.y.to_numpy()[fold['validation_indices']].sum()}; ",
            }

            self.fold_evaluation.append(fold_stats)
            fl += 1

    def plot_fold_diagnostics(self, folder: Optional[str] = None) -> None:
        """
        Same for all Classifiers
        """
        if not self.fold_evaluation:
            self._predict_folds()

        fold_metrics = self.cross_validator.evaluate_folds(self.fold_evaluation)
        # if we use LooBuffer, we have a VERY HIGH number of folds (as many as samples in the dataset) and we can't
        # display them all in the legend
        if type(self.cross_validator).__name__ == "LooBuffer":
            legend = False
        else:
            legend = "auto"
        plot_fold_metrics(fold_metrics, folder, legend)

    def plot_test_diagnostic(self, cnx: SnowflakeConnection, database_dest: str, folder: str) -> None:
        """query data and plot

        Args:
            cnx (SnowflakeConnection): connection to snowflake db
            database_dest (str): DB_MODELING<PROD>.modeling.<RUN_NAME>_APPLICATION_SET
            folder (str): where the plots are being saved

        """
        # plot for  known fields
        queries = []
        queries.append(INGENIOUS_HOT_SPRINGS.format(APPLICATION_SET_TABLE=database_dest))

        prefixes = [
            "INGENIOUS_HOT_SPRINGS",
        ]

        for prefix, query in zip(prefixes, queries):
            df = cnx.cursor().execute(query).fetch_pandas_all()
            if df.empty:
                logger.warning(f"No {prefix} data available.")
                continue
            else:
                logger.info(f"Running eval on {prefix} data.")

            y_true = df["LABEL"].to_numpy()
            y_pred = df["PRED"].to_numpy()
            unique_labels = set(y_true)
            # Check for unique labels, if we only have one label (1), we can only compute the accuracy
            if len(unique_labels) == 1:
                # cutoff of 0.5
                y_pred = (df["PRED"].to_numpy() > 0.5).astype(int)
                y_true = df["LABEL"].to_numpy()
                accuracy = accuracy_score(y_true, y_pred)
                mlflow.log_metric(f"{prefix}_acc", accuracy)
            # If there is more than 1 we can compute other validation metrics
            else:
                table = decile_table(df["LABEL"].to_numpy(), df["PRED"].to_numpy())
                pr_df, pr_auc = get_pr(df["PRED"].to_numpy(), df["LABEL"].to_numpy(), "")
                roc_df, roc_auc = get_roc(df["PRED"].to_numpy(), df["LABEL"].to_numpy(), "")

                gain = pd.DataFrame(
                    {
                        "DECILE": np.append(0, table.decile.values),
                        "RESP_PCT_MOD": np.append(0, table.cum_resp_pct.values),
                        "RESP_PCT_WIZ": np.append(0, table.cum_resp_pct_wiz.values),
                        "LABEL": None,
                    }
                )

                lift = pd.DataFrame(
                    {
                        "DECILE": table.decile.values,
                        "CUM_LIFT": table.lift.values,
                        "LIFT": table.cnt_resp.values / table.cnt_resp_rndm.values,
                        "LABEL": None,
                    }
                )

                # save metrics
                pr_df.to_csv(os.path.join(folder, f"{prefix}_precision_recall.csv"))
                gain.to_csv(os.path.join(folder, f"{prefix}_gain.csv"))
                lift.to_csv(os.path.join(folder, f"{prefix}_lift.csv"))
                roc_df.to_csv(os.path.join(folder, f"{prefix}_roc.csv"))
                plot_test_data(gain, lift, pr_df, roc_df, prefix, folder)

                mlflow.log_metric(f"{prefix}_pr_auc", pr_auc)
                mlflow.log_metric(f"{prefix}_roc_auc", roc_auc)

    def save_all_stats(self, folder: str = None):
        if folder:
            # Call Importance:
            self.importance_plot(folder)

            # Call Shapley:
            self.shapley_plot(folder)

            # Call fold Diagnostics:
            self.plot_fold_diagnostics(folder)

        else:
            raise ValueError("folder needs to be provided! Exiting!")
