import logging
import os
import shutil
from typing import Optional

import click
import mlflow
import numpy as np
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas

from modeling.config import get_cfg
from modeling.datasets.build import build_dataset
from modeling.models.base import BaseRegression, BinaryClassifier
from modeling.models.build import build_model
from modeling.utils.data import check_for_bad_features, preprocess_data
from modeling.utils.git import check_for_commits
from modeling.utils.logging import Cprofile_and_log, flatten_dict, log_config
from modeling.utils.mlflow import log_config_datasets, log_model
from modeling.utils.plotting import application_map, train_test_map
from modeling.utils.snowflake import snowflake_connection

# fix for np.int error
np.int = int

# Set up logging
logger = logging.getLogger("modeling")

# Set MLflow tracking URI to a local directory
mlflow.set_tracking_uri("file://" + os.path.abspath("./mlruns"))

logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")


def drop_snowflake_table(cn, database, schema, table):
    """
    Drops a snowflake table if one exists

    Inputs:
        cn : snowflake connection string
        database : str
            Database name
        schema : str
            Schema name
        table : str
            table name
    """

    exec_str = f"DROP TABLE IF EXISTS {database}.{schema}.{table};"

    try:
        cn.cursor().execute(exec_str)

    except Exception:
        logger.error("Dropping table failed for command: {exec_str}")
        raise


def upload_to_snowflake(cn, df, database, schema, table, drop=False):
    try:
        if drop:
            drop_snowflake_table(cn, database, schema, table)

        success, nchunks, nrows, _ = write_pandas(
            cn,
            df,
            table_name=table,
            database=database,
            schema=schema,
            auto_create_table=drop,
        )

    except Exception as e:
        logger.error(f"Failed to push data into Snowflake table {database}.{schema}.{table}.", e)
        raise e

    finally:  # All Ok Return True:
        return True


@click.command()
@click.option("-c", "--config_path", required=True, type=str)
@click.option("-e", "--experiment_name", required=True, type=str)
@click.option("-h", "--cache", required=False, type=bool, default=False)
@click.option("-s", "--save_test", required=False, type=bool, default=True)
@click.option("-n", "--run_name", required=False, type=str)
@click.option("-d", "--debug", required=False, type=bool, default=False)
@click.option("-p", "--profile", required=False, type=bool, default=False)
@click.option("-l", "--local", required=False, type=bool, default=False)
def train(
    config_path: str,
    experiment_name: str,
    cache: bool,
    save_test: bool,
    run_name: Optional[str],
    debug: bool,
    profile: bool,
    local: bool,
) -> None:
    """Train and evaluate a model

    Args:
        config_path (str): local path to the yaml config file
        experiment_name (str): name of the mlflow experiment (eg, PFA)
        cache (bool): whether to use cached dataset for faster iteration
        save_test (bool): whether to save the test/application data
        run_name (Optional[str]): name the mlflow run instead of generating a random name
        debug (bool): allow to train a model without having pushed code changes
        profile (bool): train a model and profile the results using Cprofiler
        local (bool): skip Snowflake operations if True
    """
    # Ensure -h (cache) is True if -l (local) is True
    if local and not cache:
        logger.error("The '--cache' option must be True when '--local' is True.")
        return

    # check if files have been committed
    if not debug:
        diffs = check_for_commits()
        if diffs:
            logger.error("Files have not been committed. Commit local changes before running an experiment.")
            return

    logger.info("Loading config")
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    logger.info(cfg)

    if not local:
        # Open Snowflake Connection:
        snowflake_database = os.environ["SNOWFLAKE_DATABASE"]
        schema =  os.environ["SNOWFLAKE_SCHEMA"]
        env = "production" if snowflake_database == "DB_MODELING_PROD" else "staging"
        cnx = snowflake_connection(env)

    # check if we are training in supervised or unsupervised mode
    if cfg["sql_parameters"]["labels"]:
        unsupervised = False
    else:
        unsupervised = True
        logger.warning("No labels specified, training an unsupervised learning model.")

    # Data Cache handling
    if cache:
        logger.info("Pulling data from cache!")
        Xtrain = pd.read_parquet("./runs/features_train.pq")
        Ytrain = pd.read_parquet("./runs/labels_train.pq").iloc[:, 0]
        Metatrain = pd.read_parquet("./runs/meta_columns_train.pq")
        Xapplication = pd.read_parquet("./runs/application_set.pq")
        Xtest = pd.read_parquet("./runs/features_test.pq")

        # load cached config
        cached_cfg = get_cfg()
        cached_cfg.merge_from_file("./runs/config.yaml")
        # checked that the cached config is equal to the current config for the sql_parameters keys
        flatten_cached_cfg = flatten_dict(cached_cfg)
        flatten_cfg = flatten_dict(cfg)
        for k in flatten_cfg.keys():
            if "sql_parameters" in k:
                if flatten_cfg[k] != flatten_cached_cfg[k]:
                    logger.warning(f"{k}: cached config value {flatten_cached_cfg[k]}")
                    logger.warning(f"{k}: current config value {flatten_cfg[k]}")
        shutil.copy(config_path, "./runs/config.yaml")
    else:
        logger.info("Pulling data from Snowflake!")
        if local:
            logger.error("Cannot pull data from Snowflake when running in local mode.")
            return
        Xtrain, Xtest, Xapplication, Ytrain, Metatrain = build_dataset(
            cfg["sql_parameters"], cnx, unsupervised, experiment_name
        )

        os.makedirs("./runs", exist_ok=True)

        Xtrain.to_parquet("./runs/features_train.pq")
        pd.DataFrame(Ytrain).to_parquet("./runs/labels_train.pq")
        Metatrain.to_parquet("./runs/meta_columns_train.pq")
        Xapplication.to_parquet("./runs/application_set.pq")
        Xtest.to_parquet("./runs/features_test.pq")
        shutil.copy(config_path, "./runs/config.yaml")

    logger.info(f"Dataset size : {Xtrain.shape}")
    check_for_bad_features(Xtrain)
    mlflow.set_experiment(experiment_name)

    logger.info(f"Starting Run for Experiment: {experiment_name}.")
    logger.info(
        f"Xtrain size: {Xtrain.shape}, Ytrain size {Ytrain.shape}, Xapplication size {Xapplication.shape}, "
        f"Metatrain size: {Metatrain.shape}"
    )

    with mlflow.start_run(description=cfg["run_description"], run_name=run_name) as run:
        # log the config
        log_config(cfg)

        run_id = run.info.run_name.replace("-", "_")

        # Create run folder:
        RUN_FOLDER = f"./runs/{experiment_name}/artifacts_{run_id}"

        os.makedirs(RUN_FOLDER)  # Assumes it never exists since each run is new.

        logger.info(f"Pushing all artifacts into: {RUN_FOLDER}")

        mlflow.log_param("SNOWFLAKE_USER", os.getenv("SNOWFLAKE_USER"))
        mlflow.log_param("GIT_BRANCH", os.popen("git branch --show-current").read().replace("\n", ""))
        mlflow.log_param("GIT_COMMIT", os.popen("git show --oneline -s").read())
        mlflow.log_param("n_samples", Xtrain.shape[0])
        if Xtrain.shape[0] <= Xtrain.shape[1] * 100:
            # warning if we do not have 100x more samples than features
            logger.warning(f"You are using {Xtrain.shape[1]} features but only {Xtrain.shape[0]} samples.")

        # Save the map of the train/test/application data.
        train_test_map(
            train=Xtrain,
            ytrain=Ytrain,
            test=Xtest,
            destination_folder=RUN_FOLDER,
        )

        logger.info("Fitting the model.")
        # Initialize the cross-validator:
        logger.info(f"Label ratio: {(Ytrain == 1).sum()} positives / {(Ytrain == 0).sum()} negatives")
        logger.info(f"Test data includes {cfg.sql_parameters.test_data}")
        # Log the number of positive and negative labels
        mlflow.log_metric("positive_labels", int((Ytrain == 1).sum()))
        mlflow.log_metric("negative_labels", int((Ytrain == 0).sum()))

        # Initialize the model and CV:
        model = build_model(cfg, Xtrain, Ytrain, Metatrain)

        # If creating a cprofile output is selected - log performance
        if profile:
            # Profile and execute build_cross_validator
            model.build_cross_validator = Cprofile_and_log(
                lambda: model.build_cross_validator(unsupervised, RUN_FOLDER), "Build_Cross_Validator"
            )

            # Profile and execute optimize
            model.optimize = Cprofile_and_log(lambda: model.optimize(unsupervised), "Model_Optimization")
        else:
            model.build_cross_validator(unsupervised, RUN_FOLDER)
            model.optimize(unsupervised)

        if not unsupervised:
            model.save_all_stats(folder=RUN_FOLDER)
        mlflow.log_param("n_features", Xtrain.shape[1])
        mlflow.log_params(model.optimized_params)
        log_model(model)

        # Log config and datasets
        log_config_datasets()

        if save_test:
            logger.info("Applying the model on the application set, and pushing to Snowflake.")
            # these lines are needed if features are removed
            Xapplication = Xapplication[list(Xtrain.columns)]
            Xtest = Xtest[list(Xtrain.columns)]

            _, _, _, Xapplication = preprocess_data(
                Xtrain, None, None, Xapplication, model.nan_strategy, model.preprocessing
            )

            # Apply the model on the application set and the test set:
            if isinstance(model, BinaryClassifier):
                applied_outputs = pd.DataFrame({"SCORE": model.predict_proba(Xapplication)}, index=Xapplication.index)
                applied_outputs["PERCENTILE"] = applied_outputs["SCORE"].rank(pct=True)

            elif isinstance(model, BaseRegression):
                applied_outputs = pd.DataFrame({"PREDICTION": model.predict(Xapplication)}, index=Xapplication.index)
            else:
                logger.error("Unknown model instance. Don't know how to predict!")

            # process application set
            applied_outputs.to_parquet("./runs/application_set_predictions.pq")
            applied_outputs = applied_outputs.reset_index()
            application_map(
                applied_outputs=applied_outputs,
                h3_col="H3_BLOCKS",
                score_col="SCORE",
                destination_folder=RUN_FOLDER,
            )

            if not local:
                table_name = f'{experiment_name}_{run_id.replace("-", "_").upper()}_APPLICATION_SET'
                upload_to_snowflake(
                    cnx,
                    applied_outputs,
                    database=snowflake_database,
                    schema=schema,
                    table=table_name,
                    drop=True,
                )
                logger.info(f"Results are in table: {snowflake_database}.{schema}.{table_name}")

                # process test set
                if any(cfg["sql_parameters"]["test_data"]):
                    model.plot_test_diagnostic(cnx, f"{snowflake_database}.{schema}.{table_name}", RUN_FOLDER)

        mlflow.log_artifact(RUN_FOLDER)
        logger.info("Metrics, errors and artifacts all logged.")


if __name__ == "__main__":
    train()
