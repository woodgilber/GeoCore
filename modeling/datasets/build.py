import logging
import os
from collections import Counter
from typing import List, Optional, Tuple
import pdb

import pandas as pd
from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry
from snowflake.connector.connection import SnowflakeConnection

from modeling.datasets.base import BaseFeatures
from modeling.utils.mlflow import create_meta_labels, validate_configs
from modeling.utils.snowflake import pull_prediction_data

logger = logging.getLogger(__name__)

FEATURES_REGISTRY = Registry("FEATURES_REGISTRY")
FEATURES_REGISTRY.__doc__ = "Registry for the features."

# TODO move to a build_label.py
LABEL_REGISTRY = Registry("LABEL_REGISTRY")
LABEL_REGISTRY.__doc__ = "Registry for the labels."

TEST_REGISTRY = Registry("TEST_REGISTRY")
TEST_REGISTRY.__doc__ = "Registry for test data."


def feature_props(feature: BaseFeatures, cnx: SnowflakeConnection) -> pd.DataFrame:
    """
    Assembles a data frame with the properties of one feature column
    This info is used later for generating the left join query.
    """
    df = pd.DataFrame(
        {
            "column": feature._get_columns(cnx),
            "class_name": type(feature).__name__,
            "snowflake_table_name": f"{feature.database}.{feature.schema}.{feature.table_name}",
            "index_column": feature.index_column,
        }
    )

    return df


def sql_join_script(
    df: pd.DataFrame,
    unsupervised: bool,
    first_table: Optional[str] = None,
    test_tables: Optional[pd.DataFrame] = None,
    test_buffer: Optional[float] = None,
    label_column: Optional[str] = None,
) -> Tuple[str, str]:
    """Generate SQL query from dataframes. This behaves slightly different in supervised and unsupervised modes. In
    supervised mode, we create additional CTE for train and application data. In unsupervised, we use all the data
    available. We also set the labels to 1 and we create an irrelevant Y pd.Series that won't be used later (but makes
    the code cleaner).

    Args:
        df (pd.DataFrame): pd.DataFrame
            A pandas data frame with the following columns:
            column : that specifies the columns in a snowflake table
            class_name : the name of the feature class that establishes the link between snowflake and python
            "snowflake_table_name" : full snowflake table name with database.schema.table
            "index_column" : the column we are using for joining.
        unsupervised: whether we are training an unsupervised model or not
        first_table (Optional[str], optional): name of the first table to join on. Defaults to None.
        test_tables (Optional[pd.DataFrame], optional): Same as df but for test tables. Defaults to None.
        test_buffer (Optional[float], optional): distance at which a train sample is considered to close to a test
            sample. Defaults to None.
        label_column (Optional[str], optional): name of the label column. Defaults to None.

    Returns:
        Tuple[str, str]: query and index
    """
    internal_df = df.copy()
    # Determine the first table to which we should join:
    if first_table is None:
        first_table = internal_df.class_name.unique()[0]
    internal_df["first_table"] = first_table
    internal_df["drop_column"] = internal_df["index_column"] == internal_df["column"]
    internal_df["header"] = internal_df["class_name"] + "." + internal_df["column"]
    internal_df.loc[internal_df["class_name"] == first_table, "drop_column"] = False
    # Assemble the first cte - all the columns in our final dataframe
    cte = "WITH all_data as (SELECT \n" + ",\n".join(internal_df.loc[~internal_df["drop_column"], "header"].to_list())

    # fnish the cte with the join statements:
    tables_df = (
        internal_df[["class_name", "snowflake_table_name", "index_column", "first_table"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # TODO: this does not work if the first table is not the first table in the list of features. debug later!
    index_column = tables_df.index_column[tables_df.class_name == first_table][0]
    tables_df["footer_join"] = (
        "LEFT JOIN " + tables_df["snowflake_table_name"] + " AS " + tables_df["class_name"] + "\n"
        "ON "
        + first_table
        + "."
        + tables_df.index_column[tables_df.class_name == first_table][0]
        + " = "
        + tables_df["class_name"]
        + "."
        + tables_df["index_column"]
    )
    tables_df.loc[tables_df.class_name == tables_df.first_table, "footer_join"] = (
        "FROM " + tables_df.snowflake_table_name[tables_df.class_name == first_table][0] + " AS " + first_table
    )
    cte += "\n" + "\n".join(tables_df.footer_join.to_list()) + ")"

    # the first cte is done. if we do not have test data, the query ends here.
    if test_tables is None:
        query = "\n SELECT *, 0 as IS_TEST FROM all_data"
    else:
        if test_buffer is None:
            test_buffer = 0
            logger.warning("Test buffer is not specified, using 0.")
        # if have test data, we need to create more ctes. one for all the train data and one for the application data

        if not unsupervised:
            cte += f"""
                , train as (
                    SELECT 
                        *
                    FROM
                        all_data
                    WHERE
                        all_data.{label_column} IS NOT NULL
                )
                , application as (
                    SELECT
                        *
                    FROM
                        all_data
                    WHERE
                        all_data.{label_column} IS NULL
                )
            """

        # we create ANOTHER cte for the test data using the test tables
        internal_test_df = test_tables.copy()
        internal_test_df["header"] = (
            "SELECT "
            + internal_test_df["index_column"]
            + " AS H3_INDEX FROM "
            + internal_test_df["snowflake_table_name"]
        )
        cte += f"""
            , test as (
                SELECT
                    DISTINCT H3_INDEX
                FROM
                    ({" UNION ALL ".join(internal_test_df["header"].to_list())})
            )
            """

        if unsupervised:
            # since we are using the whole Basin and range region in unsupervised mode, we pre-computed the test_buffer
            # with a 5km distance and encoded it in the BasinAndRange feature. It took 2min to run with a 3XL warehouse
            # TODO: find a better way?
            train_cte = "all_data"
        else:
            train_cte = "train"
            # one final cte for the "bad train" data -> data that's within a certain distance of the test data
            cte += f"""
                , bad_train as (
                    SELECT DISTINCT
                        {index_column} as BAD_H3_BLOCKS
                    FROM
                        {train_cte}
                    LEFT JOIN
                        test
                    WHERE
                        ST_DWITHIN(H3_CELL_TO_POINT({train_cte}.{index_column}),
                                H3_CELL_TO_POINT(test.H3_INDEX), {test_buffer})
                )
                """

        # all the ctes are written, we can now write the query
        if unsupervised:
            # the label column does not exist in unsupervised mode (we are not using labels)
            query = f"""
                SELECT
                    all_data.*
                    , H3_CELL_TO_POINT(all_data.H3_BLOCKS) as H3_CENTER
                    , ST_X(H3_CENTER) as LONGITUDE
                    , ST_Y(H3_CENTER) as LATITUDE
                    , 1 as LABEL
                    , iff(test.H3_INDEX is not null, 1, 0) AS IS_TEST
                FROM
                    all_data
                LEFT JOIN
                    test
                ON
                    all_data.{index_column} = test.H3_INDEX
            """
        else:
            query = f"""
                SELECT
                    all_data.* EXCLUDE LABEL
                    , iff(bad_train.BAD_H3_BLOCKS is not null, NULL, all_data.LABEL) as LABEL
                    , iff(test.H3_INDEX is not null, 1, 0) AS IS_TEST
                FROM
                    all_data
                LEFT JOIN
                    test
                ON
                    all_data.{index_column} = test.H3_INDEX
                LEFT JOIN
                    bad_train
                ON
                    all_data.{index_column} = bad_train.BAD_H3_BLOCKS
            """
    return (cte + query, index_column)


def merge_labels(labels: List[BaseFeatures]) -> BaseFeatures:
    """This function does some magic to create a new BaseFeatures class with a sql_code that contains a query
    to merge the different labels together.

    Args:
        labels (List[BaseFeatures]): list of labels.

    Returns:
        BaseFeatures: single class contains the new sql code.
    """

    # define and instantiate new label
    class UnionLabel(BaseFeatures):
        index_column = "H3_BLOCKS"
        table_name = "LABELS"
        sql_code = ""
        run_checks = True
        sql_asis = True

    merger = UnionLabel()

    # we create a temp table for each label (some issues arised when using CTEs
    query_code = ""
    for i, f in enumerate(labels):
        query_code += (
            f"""CREATE OR REPLACE TEMPORARY TABLE {os.environ["SNOWFLAKE_DATABASE"]}.{os.environ["SNOWFLAKE_SCHEMA"]}.label_{i} as ("""
        )
        query_code += f.sql_code
        query_code += "); \n "

    query_code += f"""CREATE OR REPLACE TEMPORARY TABLE {os.environ["SNOWFLAKE_DATABASE"]}.{os.environ["SNOWFLAKE_SCHEMA"]}.LABELS as (
        with merged_labels as ("""
    for i in range(len(labels)):
        query_code += f"""
        SELECT
            H3_BLOCKS
            , WEIGHT
            , LABEL
            , TYPE
        FROM
            {os.environ["SNOWFLAKE_DATABASE"]}.{os.environ["SNOWFLAKE_SCHEMA"]}.label_{i}
        """
        if i < len(labels) - 1:
            query_code += " UNION ALL "
    query_code += ")"

    # in addition to the merged_labels CTE, we create a new one that partiion by h3 blocks
    # in case of multiple labels for a single h3 blocks, we select the one with the highest weight
    query_code += """
        , row_counter as (
            SELECT
                *
                , ROW_NUMBER() OVER (PARTITION BY H3_BLOCKS ORDER BY WEIGHT DESC) AS RN
            FROM
                merged_labels
            )
        SELECT
            H3_BLOCKS
            , H3_CELL_TO_POINT(H3_BLOCKS) as H3_CENTER
            , ST_X(H3_CENTER) as LONGITUDE
            , ST_Y(H3_CENTER) as LATITUDE
            , LABEL
            , WEIGHT
            , TYPE
        FROM row_counter
        WHERE RN = 1
        )"""
    merger.sql_code = query_code
    return merger


def build_dataset(
    cfg: CfgNode, cnx: SnowflakeConnection, unsupervised: bool, experiment_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build datasets from config

    Args:
        cfg (CfgNode): dataset config the defines the features and labels to pull
        cnx : Snowflake Connection
        unsupervised (bool): whether supervised learning or not
        experiment_name (str): name of mlflow experiment

    Returns:
        Xtrain : pd.DataFrame. A dataframe containing the training/validation features
        Xtest: pd.DataFrame. A dataframe containing the 2m test data. This dataset is a subset of the application set.
        Xapplication: pd.DataFrame. A dataframe containing the application data
        Ytrain: pd.Series. A pandas series with the training/validation target variable
        metatrain : pd.DataFrame. A dataframe with the metadata (i.e. used in cross validation like Lat and Long etc.)
    """
    if len(set(cfg["features"])) != len(cfg["features"]):
        logger.exception("It appears that we have repeating features")
        raise ValueError("Wrong feature list specification!")

    # Instantiate the feature instances:
    feature_instances = [FEATURES_REGISTRY.get(feature)() for feature in cfg["features"]]

    if not unsupervised:
        if len(set(cfg["labels"])) != len(cfg["labels"]):
            logger.exception("It appears that we have repeating features")
            raise ValueError("Wrong feature list specification!")
        merged_labels = merge_labels([LABEL_REGISTRY.get(label)() for label in cfg["labels"]])
        feature_instances += [merged_labels]
        unsupervised = False

    test_instances = [TEST_REGISTRY.get(data)() for data in cfg["test_data"] if data]

    # Materialize the tables:
    for table in feature_instances + test_instances:
        table.run_sql(cnx, cfg.params)

    pdb.set_trace()
    # check resolutions
    # TODO: check resolution on test instances too. I am getting some None resolutions for some reason
    resolutions = {f"{type(table).__name__}": table.resolution for table in feature_instances}
    if len(set(resolutions.values())) == 1:
        logger.info(f"All tables are using the same resolution of {list(resolutions.values())[0]}")
    else:
        most_common_res = Counter(resolutions.values()).most_common()[0][0]
        bad_tables = []
        bad_resolutions = []
        for k, v in resolutions.items():
            if v != most_common_res:
                bad_tables.append(k)
                bad_resolutions.append(v)
        raise ValueError(f"Tables {bad_tables} are using different resolutions: {bad_resolutions}")

    pdb.set_trace()
    # Get Table columns and properties etc.
    df = pd.concat([feature_props(f, cnx) for f in feature_instances])
    first_table = cfg["first_table"]
    test_tables = pd.concat([feature_props(f, cnx) for f in test_instances]) if test_instances else None
    join_sql, index_column = sql_join_script(
        df, unsupervised, first_table, test_tables, cfg.params.test_buffer, cfg.label_column
    )

    # save the sql query
    os.makedirs("./runs", exist_ok=True)
    with open("./runs/sql_join_script.sql", "w") as f:
        f.write(join_sql)

    # Pull the data
    try:
        df = cnx.cursor().execute(join_sql).fetch_pandas_all().set_index(index_column)
    except Exception:
        logger.exception(f"Errored out when executing a Snowflake join and fetch. Query: {join_sql}")
        raise ValueError
    else:
        logger.debug("Complete Feature Engineering SQL Code:")
        logger.debug("\n".join([feature.executed_sql for feature in feature_instances]) + "\n" + join_sql)
        logger.debug(df.columns.tolist())

    # Prep the output:
    if "meta_columns" in cfg:
        col_set = set(df.columns)
        meta_set = set(cfg["meta_columns"])

        if meta_set.difference(col_set):
            logger.warning(
                f"Some of the meta columns were not found in the feature set: {meta_set.difference(col_set)}"
            )

        meta = df[list(col_set.intersection(meta_set))].copy()
        drop_columns = cfg["meta_columns"] + [cfg["label_column"]]

    else:
        meta = None
        drop_columns = cfg["label_columns"]

    # get indices for test data
    test_indices = df.IS_TEST == 1
    drop_columns.append("IS_TEST")

    # create features, labels dataframes
    Y = df[cfg["label_column"]].copy()  # label column is required
    X = df.drop([col for col in drop_columns if col in df.columns.tolist()], axis=1)

    # Check for categoricals:
    if "categorical_features" in cfg.keys():
        for cf in cfg["categorical_features"]:
            X[[cf]] = X[[cf]].astype("category")

    # create the splits
    Xtrain = X[~test_indices]
    Xtest = X[test_indices]
    Ytrain = Y[~test_indices]
    metatrain = meta[~test_indices]

    if unsupervised:
        # Xapplication and Xtrain are the same
        Xapplication = X[~test_indices]
    else:
        # application set is everything without a label - does not matter test or not
        Xapplication = X  # [Y.isnull()]

        # remove data without labels from train set
        Xtrain, Ytrain, metatrain = (
            Xtrain[~Ytrain.isnull()].copy(),
            Ytrain[~Ytrain.isnull()],
            metatrain[~Ytrain.isnull()],
        )

    # if we use meta-labels, load them. we need to pull both from training and application set the set of labels may
    # have changed
    if cfg.meta_labels:
        logger.warning("Using meta labels, dropping all the features!")
        validate_configs(cfg.meta_labels)
        # we need to pull the training data predictions
        train_preds = create_meta_labels(cfg.meta_labels)
        # and the application data predictions
        application_preds = pull_prediction_data(cnx, experiment_name, cfg.meta_labels, "APPLICATION", index_column)
        # concat both
        all_data = pd.concat([train_preds, application_preds])
        # select training data
        meta_labels = all_data[all_data.index.isin(Xtrain.index)]
        Xapplication = all_data[~all_data.index.isin(Xtrain.index)]
        Xtest = Xapplication

        # we join the different tables to make sure that the index order is maintained
        Xtrain = meta_labels
        Ytrain = meta_labels.join(Ytrain).drop(meta_labels.columns, axis=1)["LABEL"]
        metatrain = meta_labels.join(metatrain).drop(meta_labels.columns, axis=1)

    return Xtrain, Xtest, Xapplication, Ytrain, metatrain
