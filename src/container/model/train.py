import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier


sys.path.append("..")  # Do not remove this
logging.basicConfig(level=logging.DEBUG)


####################
## Global defines ##
####################

CONTAINER_DIR_PREFIX = "/opt/ml/"
DATA_CHANNEL = "train/"
DATA_DIR = os.path.join(CONTAINER_DIR_PREFIX, "input/data/{}".format(DATA_CHANNEL))
SAVE_DIR = os.path.join(CONTAINER_DIR_PREFIX, "model")
HYPERPARAMS_PATH = os.path.join(
    CONTAINER_DIR_PREFIX, "input/config/hyperparameters.json"
)
FAILURE_DIR = os.path.join(CONTAINER_DIR_PREFIX, "failure")

TARGET_VARIABLE = "class"

############################
## Main Training Workflow ##
############################


def train(hyperparams_path=None):
    logging.info("Starting training.")

    logging.info("Getting training data.")
    train_fname = os.path.join(DATA_DIR, "train.csv")
    data = pd.read_csv(train_fname)
    features = data.drop(TARGET_VARIABLE, axis=1)
    targets = data[TARGET_VARIABLE]

    logging.info("Getting hyperparameters.")
    if not hyperparams_path:
        hyperparams_path = HYPERPARAMS_PATH

    hyperparams = get_hyperparameters(hyperparams_path)

    logging.info("Casting data types for hyperparameters.")
    hyperparams = cast_dtypes_for_hyperparameters(hyperparams)

    logging.info("Scaling features.")
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.joblib"))

    logging.info("Encoding target variable.")
    label_encoder = LabelEncoder()
    targets = label_encoder.fit_transform(targets)
    joblib.dump(label_encoder, os.path.join(SAVE_DIR, "label_encoder.joblib"))

    logging.info("Training model.")
    model = XGBClassifier(**hyperparams)
    model.fit(scaled, targets)
    logging.info("Model Performance Metrics:")
    logging.info("Accuracy: {}".format(model.score(scaled, targets)))
    joblib.dump(model, os.path.join(SAVE_DIR, "model.joblib"))

    logging.info("Training complete.")

    return


###############################
## Training Helper Functions ##
###############################


def get_hyperparameters(hyperparameters_path):
    try:
        hyperparameters = {}
        with open(hyperparameters_path) as json_file:
            json_string = json.load(json_file)
            json_file.close()
            hyperparameters = json_string
        for key, value in hyperparameters.items():
            hyperparameters[key] = json.loads(value)
        return hyperparameters
    except Exception as e:
        logging.exception("Exception occured while getting hyperparameters. Aborting.")


def cast_dtypes_for_hyperparameters(hyperparams):
    def cast(hyperparams_dict, key):
        casting_mapper = {
            # XGBoost hyperparameter dtypes
            "max_depth": int,
            "learning_rate": float,
            "n_estimators": int,
            "silent": bool,
            "objective": str,
            "eval_metric": str,
            "booster": str,
            "nthread": int,
            "n_jobs": int,
            "gamma": float,
            "min_child_weight": int,
            "max_delta_step": int,
            "subsample": float,
            "colsample_bytree": float,
            "colsample_bylevel": float,
            "colsample_bynode": float,
            "reg_alpha": float,
            "reg_lambda": float,
            "scale_pos_weight": float,
            "random_state": int,
            "missing": float,
            "importance_type ": str,
        }
        casting_func = casting_mapper[key]
        try:
            hyperparams_dict[key] = casting_func(hyperparams_dict[key])
        except KeyError:
            pass
        return hyperparams_dict

    for model in hyperparams.keys():
        for k, v in hyperparams[model].items():
            hyperparams[model] = cast(hyperparams[model], k)
    return hyperparams


################
## Invocation ##
################

if __name__ == "__main__":
    train(DATA_DIR, SAVE_DIR, HYPERPARAMS_PATH, FAILURE_DIR)

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
