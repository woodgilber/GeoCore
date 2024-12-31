import os
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from .base import BinaryClassifier
from .build import MODEL_REGISTRY

# Set MLflow tracking URI to a local directory
mlflow.set_tracking_uri("file://" + os.path.abspath("./mlruns"))


class MLPClassifierPfa(BaseEstimator, ClassifierMixin):
    """Small rewrite of the sklearn class to add sample weight argument and make it compatible with other
    BinaryClassifier methods. Also add to rewrite the init method to enable optimization on the number of layers.
    see https://github.com/scikit-optimize/scikit-optimize/issues/653"""

    def __init__(
        self,
        n_layers: int = 1,
        n_neurons: int = 100,
        activation: str = "relu",
        solver: str = "adam",
        learning_rate_init: float = 0.001,
        learning_rate: str = "constant",
        max_iter: int = 1000,
        verbose: int = 0,
    ) -> None:
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        # placeholder
        self.classes_ = None
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.array] = None):
        hidden_layer_sizes = (2**self.n_neurons,) * self.n_layers
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
        )
        model.fit(X, y)
        self.model = model
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)


@MODEL_REGISTRY.register()
class NeuralNetworkModel(BinaryClassifier):
    model = MLPClassifierPfa
    nan_strategy = "impute_mean"
    preprocessing = MinMaxScaler

    def _parameter_importance(self, folder: str = None):
        """
        This is classifier specific
        """
        return

    def _set_params(self) -> None:
        self.optim_params = self.model_config.nn.optim_params
        self.model_params = self.model_config.nn.model_params

    def shapley_plot(self, folder: str = None):
        """
        Classifier specific Shapley Plot
        """
        if self.clf:
            shap_values = shap.KernelExplainer(self.clf.predict_proba, self.X).shap_values(self.X)
            shap.summary_plot(shap_values, self.X, show=False if folder else True)

            if folder:
                plt.gcf()
                plt.savefig(f"{folder}/shap_importance.png", bbox_inches="tight")
                plt.close()
            else:
                plt.show()

        else:
            raise ValueError("Classifier was not fitted! Exiting")
        return
