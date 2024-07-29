import click
import functools as ft
import json
import logging
import os
import pandas as pd

from voodoo_homework.config import load_config
from voodoo_homework.utils import load_data
from voodoo_homework.features.base_features import BaseFeatures
from voodoo_homework.features.extra_features import ExtraFeatures
from voodoo_homework.features.post_popularity import PostPopularity
from voodoo_homework.features.user_post_popularity import UserPostPopularity

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUTPUT_ROOT = CONF["path"]["output_data_root"]

FEATURE_DICT = {
    "base_features": BaseFeatures,
    "extra_features": ExtraFeatures,
    "post_popularity": PostPopularity,
    "user_post_popularity": UserPostPopularity
}


@click.group()
def build():
    pass


@build.command()
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
def build_features(data_path, output_root):
    logging.info("Loading Data")

    df = load_data()

    for feature in FEATURE_DICT.values():
        feature.extract_feature(df, save=True)
