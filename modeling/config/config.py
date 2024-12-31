from fvcore.common.config import CfgNode

from .defaults import _C


def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.

    Returns:
        a CfgNode instance.
    """

    return _C.clone()
