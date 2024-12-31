import logging
from typing import Union

import pandas as pd
from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry

from modeling.models.base import BaseRegression, BinaryClassifier

logger = logging.getLogger(__name__)

MODEL_REGISTRY = Registry("MODEL_REGISTRY")
MODEL_REGISTRY.__doc__ = "Registry for the models."


def build_model(
    cfg: CfgNode,
    Xtrain: pd.DataFrame,
    Ytrain: pd.Series,
    Metatrain: pd.DataFrame,
) -> Union[BinaryClassifier, BaseRegression]:
    """Build model for training

    Args:
        cfg (CfgNode): config
        Xtrain (pd.DataFrame): training features
        Ytrain (pd.Series): training labels
        Metatrain (pd.DataFrame): training metadata
        cross_validator (BaseCV): cross validator class
        unsupervised (bool): whether to train unsupervised model or not

    Returns:
        Union[BinaryClassifier, BaseRegression]: model
    """
    model_name = cfg["model"]["name"]
    model_class = MODEL_REGISTRY.get(model_name)
    model = model_class(
        X=Xtrain,
        y=Ytrain,
        metadata=Metatrain,
        weights=(
            Metatrain[[cfg["sql_parameters"]["weights_column"]]]
            if cfg["sql_parameters"]["weights_column"] is not None
            else None
        ),
        model_config=cfg["model"],
    )
    return model
