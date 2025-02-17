from snowflake import connector
import pandas as pd
import itertools
import yaml
import logging
import os
import tensorflow as tf
from sklearn import preprocessing
import numpy as np
import datetime as dt
import argparse

default_params = {"batch_size":128,
                  "num_epochs":20,
                  "max_nodes":16,
                  "max_depth":3,
                  "test_size":0.2,
                  "validation_size":0.2,
                  "optimizer":'adam'}

logging.basicConfig(format="%(levelname)s %(asctime)s %(message)s",
                    datefmt="%H:%M:%S", level = logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--table',
    type = str,
    required = True,
    help = 'Name of the table to run modeling on'
    )

parser.add_argument(
    '--params',
    type = str,
    required = True,
    help = 'pathway to hyperparameter file'
    )

args = parser.parse_args()

with open(args.params) as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def prep_data(table_name: str, params: dict) -> list:
    logging.info("Retrieving Snowflake credentials")
    try:
        scn = connector.connect(
            user = os.getenv("SNOWFLAKE_USER"),
            password = os.getenv("SNOWFLAKE_PASSWORD"),
            account = os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse = 'geocore_warehouse',
            database = 'fishing_db',
            schema = 'fishing_schema',
            insecure_mode=True
        )
        scs = scn.cursor()
    except Exception as e:
        logging.error("Error establishing Snowflake connection")
        logging.error(f"{e}")

    sql = f"SELECT * FROM {table_name}"
    scs.execute(sql)
    df = scs.fetch_pandas_all()

    logging.info(f"Successfully read {len(df)} rows of data")

    if len(params["test_size"]) > 1 or len(params["validation_size"]) > 1:
        logging.error("Only one test/validation size is allowed to be specified")

    raw_inputs = df.iloc[:,3:-1]
    labels_all = df.iloc[:,-1]

    scaled_inputs = preprocessing.scale(raw_inputs)
    shuffled_indices = np.arange(scaled_inputs.shape[0])
    np.random.shuffle(shuffled_indices)

    shuffled_inputs = scaled_inputs[shuffled_indices]
    shuffled_targets = labels_all[shuffled_indices]

    samples_count = shuffled_inputs.shape[0]
    test_samples_count = int(params["test_size"][0] * samples_count)
    validation_samples_count = int(params["validation_size"][0] * samples_count)
    train_samples_count = samples_count - test_samples_count - validation_samples_count

    train_inputs = shuffled_inputs[:train_samples_count]
    train_targets = shuffled_targets[:train_samples_count]

    validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
    validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

    test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
    test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

    np.savez('fishing_data_train', inputs=train_inputs, targets=train_targets)
    np.savez('fishing_data_validation', inputs=validation_inputs, targets=validation_targets)
    np.savez('fishing_data_test', inputs=test_inputs, targets=test_targets)

    scs.close()
    scn.close()
    return [train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets]

def run_model(params: dict, data: list, run_num: str):

    hparams = dict((k, params[k]) for k in ('max_nodes', 'max_depth', 'optimizer', 'activation'))

    run_name = f"run-{run_num}"
    logging.info(f'--- Starting trial: {run_name}')
    logging.info(', '.join([f"{key}: {hparams[key]}" for key in hparams.keys()]))

    [train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets] = data

    model = tf.keras.Sequential()
    for n in range(hparams['max_depth']):
        model.add(tf.keras.layers.Dense(units=int(hparams['max_nodes'])-hparams['max_nodes']*n/hparams['max_depth'],activation=hparams['activation']))

    model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

    model.compile(optimizer=hparams['optimizer'],
                loss='binary_crossentropy',
                metrics=['binary_accuracy',tf.keras.metrics.AUC()])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        min_delta = 0,
        patience = 3,
        verbose = 0, 
        restore_best_weights = True
    )

    log_dir = "logs/hparam_tuning/" + run_name + '_' + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(
        train_inputs,
        train_targets,
        epochs=params["num_epochs"],
        batch_size=params["batch_size"],
        callbacks = [tensorboard_callback, early_stopping], 
        validation_data = [validation_inputs, validation_targets]
        )

    loss, accuracy, auc = model.evaluate(test_inputs, test_targets)
    logging.info('Accuracy: %.2f' % (accuracy*100))
    logging.info('ROC AUC: %.2f' % (auc))

specified_keys = params.keys()
for key, value in default_params.items():
    if key not in specified_keys:
        params[key] = [value]

data = prep_data(args.table, params)

if len(params["batch_size"]) > 1 or len(params["num_epochs"]) > 1:
    logging.error("Only one batch size/number of epochs is allowed to be specified")

keys, values = zip(*params.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

run_num = 0
for run_dict in permutations_dicts:
    run_model(run_dict,data,run_num)
    run_num += 1

logging.info("Complete, run `tensorboard --logdir 'pth/to/dir'` to investigate run details")