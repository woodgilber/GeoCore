import os

import matplotlib.pyplot as plt
import mlflow
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from .base import BinaryClassifier
from .build import MODEL_REGISTRY

# Set MLflow tracking URI to a local directory
mlflow.set_tracking_uri("file://" + os.path.abspath("./mlruns"))


@MODEL_REGISTRY.register()
class SvmModel(BinaryClassifier):
    """sklearn svm model"""

    model = SVC
    nan_strategy = "impute_mean"
    # converges faster with minmax scaler instead of robust.
    preprocessing = MinMaxScaler
    verbose = 0

    def _parameter_importance(self, folder: str = None):
        """
        This is classifier specific
        """
        return

    def _set_params(self) -> None:
        self.optim_params = self.model_config.svm.optim_params
        self.model_params = self.model_config.svm.model_params

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
