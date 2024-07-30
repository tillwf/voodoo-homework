import os
import pandas as pd

from voodoo_homework.config import load_config

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUPUT_ROOT = CONF["path"]["output_data_root"]


def load_data():
    df = pd.read_parquet(
        path=DATA_PATH
    )
    return df


def load_features():
    return pd.read_parquet(os.path.join(OUPUT_ROOT, "features.parquet"))


def load_datasets():
    return (
        pd.read_parquet(os.path.join(OUPUT_ROOT, "train.parquet")),
        pd.read_parquet(os.path.join(OUPUT_ROOT, "eval.parquet")),
        pd.read_parquet(os.path.join(OUPUT_ROOT, "test.parquet")),
    )
