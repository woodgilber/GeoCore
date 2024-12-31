import logging
from typing import Optional

import pandas as pd
from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry

from modeling.cross_validators.base import BaseCV

logger = logging.getLogger(__name__)

XVALIDATOR_REGISTRY = Registry("XVALIDATOR_REGISTRY")
XVALIDATOR_REGISTRY.__doc__ = "Registry for the Cross Validators."


def build_cv(
    cfg: CfgNode, Metatrain: pd.DataFrame, Ytrain: pd.Series, unsupervised: bool, run_folder: str
) -> Optional[BaseCV]:
    """Create cross validator instance

    Args:
        cfg (CfgNode): model config node
        Metatrain (pd.DataFrame): training metadata
        Ytrain (pd.Series): training labels
        unsupervised (bool): whether we are running unsupervised or not
        run_folder (str): dest of plots

    Returns:
        Optional[BaseCV]: cross validator instance
    """
    cross_validator = None
    if not unsupervised:
        cross_validator = XVALIDATOR_REGISTRY.get(cfg["cross_validator"])
        cross_validator = cross_validator(
            df=Metatrain,
            config=cfg["cross_validator_parameters"],
            train_labels=Ytrain,
        )
        cross_validator.save_fold_plot(run_folder)
    return cross_validator
