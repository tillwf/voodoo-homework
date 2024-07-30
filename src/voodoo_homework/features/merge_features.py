import click
import functools as ft
import json
import logging
import os
import pandas as pd

from voodoo_homework.config import load_config
from voodoo_homework.features.base_features import BaseFeatures
from voodoo_homework.features.extra_features import ExtraFeatures
from voodoo_homework.features.time_series_features import TimeSeriesFeatures

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUTPUT_ROOT = CONF["path"]["output_data_root"]
INTERIM_ROOT = CONF["path"]["interim_data_root"]

FEATURE_DICT = {
    "base_features": BaseFeatures,
    "extra_features": ExtraFeatures,
    "time_series_features": TimeSeriesFeatures
}

FEATURE_DEFINITIONS = CONF["feature_definitions"]
FEATURES = CONF["features"]


@click.group()
def merge():
    pass


@merge.command()
@click.option(
    '--data-path',
    type=str,
    default=DATA_PATH,
    help='Path of train dataset, default is {}'.format(
        DATA_PATH
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
def merge_features(data_path, output_root):
    """ Merge all parquet file into one big parquet file"""
    df = pd.DataFrame()

    for feature_group, features in FEATURE_DEFINITIONS.items():
        key = FEATURES[feature_group]
        logging.info(f"Merging {feature_group}")
        for feature in features:
            logging.info(f" - {feature}")
            temp_df = pd.read_parquet(os.path.join(INTERIM_ROOT, f"{feature}.parquet"))
            if len(df) == 0:
                df = temp_df
            else:
                df = pd.merge(
                    left=df,
                    right=temp_df,
                    on=key,
                    how="left")
    
    path = os.path.join(OUTPUT_ROOT, f"features.parquet")
    logging.info(f"Saving features to {path}")
    df.to_parquet(path, index=False)
