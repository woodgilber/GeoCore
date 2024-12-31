We declare all cross validators s.t. they play well with the rest of the sklearn pipeline. This means that each class needs to have split and get_n_splits routines defined. We created all that in the .base.py:BaseCV class. Every new cross validator needs to just inherit the BaseCV and you are good to go.

We use cross validator registry to make sure our custom cross validators play well with the rest of the pipeline. This means that each new cross validator needs to look something like this:

from .cross_validators import XVALIDATOR_REGISTRY
import logging

logger = logging.getLogger(__name__)

@XVALIDATOR_REGISTRY.register()
class CustomCrossValidator(BaseCV): 

    REQUIRED_COLUMNS = ["COLUMN1", "COLUMN2"]
    REQUIRED_CONFIG_FIELDS = ["a", "b", "c"]

    # class routines here. Make sure the code plays well with split and get_n_splits
If you place your cross validator in a new file, make sure you import it in the cross_validators.py (i.e. from .some_module import * )