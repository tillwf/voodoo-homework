import click
import fastparquet
import functools as ft
import json
import logging
import os
import pandas as pd

from tensorflow.keras.models import load_model

from voodoo_homework.features import USER_FEATURES
from voodoo_homework.features import TIME_SERIES_FEATURES

from voodoo_homework.config import load_config
from voodoo_homework.utils import load_data
from voodoo_homework.utils import load_datasets
from voodoo_homework.utils import load_features

from voodoo_homework.features.base_features import BaseFeatures
from voodoo_homework.features.extra_features import ExtraFeatures
from voodoo_homework.features.time_series_features import TimeSeriesFeatures

from voodoo_homework.models.losses import mean_square_error_log
from voodoo_homework.models.losses import weighted_mape_tf


CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
MODELS_ROOT = CONF["path"]["models_root"]
LOGS_ROOT = CONF["path"]["logs_root"]
INTERIM_ROOT = CONF["path"]["interim_data_root"]
OUTPUT_ROOT = CONF["path"]["output_data_root"]

FEATURE_DICT = {
    "base_features": BaseFeatures,
    "extra_features": ExtraFeatures,
    "time_series_features": TimeSeriesFeatures
}

FEATURE_DEFINITIONS = CONF["feature_definitions"]
FEATURES = CONF["features"]


@click.group()
def predict():
    pass


@predict.command()
@click.option(
    '--testset-path',
    type=str,
    default=os.path.join(OUTPUT_ROOT, "test.csv"),
    help='Path of test dataset, default is {}'.format(
        os.path.join(OUTPUT_ROOT, "test.csv")
    )
)
@click.option(
    '--models-root',
    type=str,
    default=MODELS_ROOT,
    help='Path of models folder, default is {}'.format(
        MODELS_ROOT
    )
)
@click.option(
    '--output-root',
    type=str,
    default=OUTPUT_ROOT,
    help='Path of output folder, default is {}'.format(
        OUTPUT_ROOT
    )
)
@click.option(
    '--features',
    type=str,
    multiple=True,
    default=list(FEATURE_DICT.keys()),
    help='Features used for the training, default is {}'.format(
        list(FEATURE_DICT.keys())
    )
)
def make_predictions(testset_path, models_root, output_root, features, evaluate=True):
    logging.info("Make Prediction")

    logging.info("Reading test data")
    _, _, X_test = load_datasets()
    y_test = X_test.pop("d120_rev")

    logging.info("Create features")
    data = load_features()

    # Select the feature to use based on the `features` param
    features_names = [
        f
        for feature_group in features
        for f in FEATURE_DEFINITIONS[feature_group]
        if FEATURE_DICT.get(feature_group)
    ]

    cols = []
    for filename in features_names:
        path = os.path.join(INTERIM_ROOT, f"{filename}.parquet")
        cols += fastparquet.ParquetFile(path).columns

    cols = set([c for c in cols if not c.startswith("__")])

    # Merge the "user based" features
    X_test = pd.merge(
        X_test,
        data[cols],
        on=["user_id", "cohort"]
    ).replace({False: 0, True: 1})\
     .select_dtypes(['number']).fillna(0)

    logging.info("Loading model")

    model = load_model(
        os.path.join(models_root, "final_model"),
        custom_objects={'mean_square_error_log': mean_square_error_log}
    )

    logging.info("Making predictions")
    raw_predictions = pd.DataFrame(
        model.predict(X_test),
        index=X_test.index,
        columns=["predictions"]
    )

    # Add columns to compute the metrics (Mean Rank, MAP@10, etc.)
    X_test["predictions"] = raw_predictions
    import ipdb; ipdb.set_trace()
    # Saving the predictions
    logging.info("Saving predictions")
    X_test.to_parquet(os.path.join(OUTPUT_ROOT, "raw_predictions.parquet"))
