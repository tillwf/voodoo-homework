import os
import pandas as pd

from voodoo_homework.config import load_config

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUPUT_ROOT = CONF["path"]["output_data_root"]


def load_data():
    return pd.read_json(
        path_or_buf=DATA_PATH,
        lines=True,
        compression="gzip"
    ).drop_duplicates(subset=[
        "trackable_id",
        "user_id",
        "tracker_created_at"
    ])


def load_features():
    return pd.read_parquet(os.path.join(OUPUT_ROOT, "features.parquet"))


def load_datasets():
    return (
        pd.read_parquet(os.path.join(OUPUT_ROOT, "train.parquet")),
        pd.read_parquet(os.path.join(OUPUT_ROOT, "eval.parquet")),
        pd.read_parquet(os.path.join(OUPUT_ROOT, "test.parquet")),
    )
