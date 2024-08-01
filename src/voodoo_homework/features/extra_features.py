import logging
import os

from voodoo_homework.config import load_config
from voodoo_homework.features.feature import Feature

IDS = [
    "user_id",
]

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUTPUT_ROOT = CONF["path"]["interim_data_root"]


class ExtraFeatures(Feature):

    @classmethod
    def extract_feature(cls, df, save=False):
        """Compute extra features"""
        logging.info("Adding extra features")

        # Extract information from `install_date`
        df["install_month"] = df.install_date.dt.month
        df["install_day_of_month"] = df.install_date.dt.day
        df["install_day_of_week"] = df.install_date.dt.dayofweek

        EXTRA_COLS = [
            "install_month",
            "install_day_of_month",
            "install_day_of_week"
        ]

        if save:
            os.makedirs(OUTPUT_ROOT, exist_ok=True)
            df[IDS + EXTRA_COLS].to_parquet(
                os.path.join(OUTPUT_ROOT, f"extra_features.parquet"),
                index=False
            )

        return df[IDS + EXTRA_COLS]
