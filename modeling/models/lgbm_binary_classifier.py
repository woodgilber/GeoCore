import os

import matplotlib.pyplot as plt
import mlflow
import shap
from lightgbm import LGBMClassifier

from .base import BinaryClassifier
from .build import MODEL_REGISTRY

# Set up logging

# Set MLflow tracking URI to a local directory
mlflow.set_tracking_uri("file://" + os.path.abspath("./mlruns"))


@MODEL_REGISTRY.register()
class LGBMBinaryClassifier(BinaryClassifier):
    """
    Basic LightGBM Binary Classifier Model
    """

    model = LGBMClassifier
    nan_strategy = None

    def _set_params(self) -> None:
        self.optim_params = self.model_config.lightgbm.optim_params
        self.model_params = self.model_config.lightgbm.model_params

    def _parameter_importance(self, folder: str = None):
        """
        This is classifier specific
        """

        from lightgbm import plot_importance

        imp = plot_importance(self.clf)
        fig = imp.get_figure()
        fig.savefig(f"{folder}/importance_plots.png", bbox_inches="tight")
        fig.clear()
        del fig

    def _lift_plot(self, folder: str):
        return NotImplemented

    def shapley_plot(self, folder: str = None):
        """
        Classifier specific Shapley Plot
        """
        if self.clf:
            shap_values = shap.TreeExplainer(self.clf).shap_values(self.X)
            shap.summary_plot(shap_values, self.X, show=False if folder else True)

            if folder:
                plt.gcf()
                plt.savefig(f"{folder}/shap_importance.png", bbox_inches="tight")
                plt.close()
            else:
                plt.show()

        else:
            raise ValueError("Classifier was not fitted! Exiting")
