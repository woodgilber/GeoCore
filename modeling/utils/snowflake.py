import os
from typing import List

import mlflow
import pandas as pd
import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from snowflake.connector.connection import SnowflakeConnection


def snowflake_connection(env: str) -> SnowflakeConnection:
    """Create snowflake connection using two differents method whether we are using the staging db (local dev) or the
    prod db (ci/cd dev)

    Args:
        env (str): either production or staging.
        warehouse (str): snowflake warehouse. Default to ANALYST_WH.

    Returns:
        SnowflakeConnection: connection to snowflake db
    """
    if env == "staging":
        # user needs to set password
        cnx = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            role=os.getenv("SNOWFLAKE_ROLE"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA"),
        )
    elif env == "production":
        # we use a private key for this. The key is contained in a gcp secret (see training-workflow)
        p_key = serialization.load_pem_private_key(
            os.environ["PRIVATE_KEY"].encode(),
            password=os.environ["PRIVATE_KEY_PASSPHRASE"].encode(),
            backend=default_backend(),
        )

        pkb = p_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        cnx = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            private_key=pkb,
            warehouse=warehouse,
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA"),
        )
    return cnx


def pull_prediction_data(
    cnx: SnowflakeConnection, experiment_name: str, run_ids: List[str], data_type: str, index_column: str
) -> pd.DataFrame:
    """Pull data from either the application or test table

    Args:
        cnx (SnowflakeConnection): snowflake connector
        experiment_name (str): name of mlflow experiment
        run_ids (List[str]): list of mlflow run ids
        data_type (str): either application or test
        index_column (str): index to use for join

    Returns:
        pd.DataFrame: joined dataframe
    """
    for i, run_id in enumerate(run_ids):
        # get run name
        run = mlflow.get_run(run_id)
        run_name = run.info.run_name
        run_name_fmted = run_name.replace("-", "_").upper()

        # pull data from snowflake
        table_name = f"{experiment_name}_{run_name_fmted}_{data_type}_SET"
        query = f"SELECT * FROM {table_name}"
        predictions = cnx.cursor().execute(query).fetch_pandas_all().set_index(index_column)
        predictions = predictions.rename(columns={"SCORE": run_name})
        if i == 0:
            data = predictions
        else:
            data = data.join(predictions)
    return data
