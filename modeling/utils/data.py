import logging
from typing import Optional, Tuple

import pandas as pd
from sklearn.base import TransformerMixin

logger = logging.getLogger(__name__)
META_COLUMNS = ["WEIGHT", "LATITUDE", "LONGITUDE", "H3_CENTER", "H3_PARENT", "TYPE"]


def check_for_bad_features(features: pd.DataFrame) -> None:
    """Dumb function to make sure that we do not have features like WEIGHT or H3_PARENT

    Args:
        features (pd.DataFrame): train dataset
    """
    intersection = set(META_COLUMNS).intersection(set(features.columns))
    if intersection:
        raise ValueError(f"The following meta features leaked in the training set: {intersection}")


def preprocess_data(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    metadata: Optional[pd.DataFrame] = None,
    other_data: Optional[pd.DataFrame] = None,
    nan_strategy: Optional[str] = None,
    preprocessing: Optional[TransformerMixin] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Remove, impute, ignore nan values.

    Args:
        X (pd.DataFrame): _description_
        y (Optional[pd.Series], optional): _description_. Defaults to None.
        metadata (Optional[pd.DataFrame], optional): _description_. Defaults to None.
        other_data (Optional[pd.DataFrame], optional): _description_. Defaults to None.
        nan_strategy (Optional[str], optional): _description_. Defaults to None.
        preprocessing (Optional[TransformerMixin], optional): sklearn preprocessing function. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame]: _description_
    """
    # first, deal with nans
    if nan_strategy == "remove":
        new_df = X.join(y).join(metadata)
        new_df = new_df.dropna()
        X = new_df.loc[:, new_df.columns.isin(X.columns)]
        if y:
            y = new_df.loc[:, new_df.columns.isin([y.name])]
        if metadata:
            metadata = new_df.loc[:, new_df.columns.isin(metadata.columns)]
        if other_data is not None:
            other_data.dropna(inplace=True)
    elif nan_strategy == "impute_mean":
        mean_value = X.mean()
        X.fillna(mean_value, inplace=True)
        if other_data is not None:
            other_data.fillna(mean_value, inplace=True)
    logger.info(f"New data size after dealing with nans {X.shape}")

    # then deal with scaling when required
    if preprocessing is not None:
        transformer = preprocessing().fit(X)
        scaled_X = transformer.transform(X)
        X = pd.DataFrame(scaled_X, index=X.index, columns=X.columns)
        if other_data is not None:
            scaled_other = transformer.transform(other_data)
            other_data = pd.DataFrame(scaled_other, index=other_data.index, columns=other_data.columns)
    return X, y, metadata, other_data
