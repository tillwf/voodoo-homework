import logging
import os
import pandas as pd

from voodoo_homework.config import load_config
from voodoo_homework.features.feature import Feature

IDS = [
    "user_id",
    "cohort",
]

USER_FEATURES = [
    "app_id",
    "platform",
    "is_optin",
    # "install_date"  ## Cardinality too high
    # "game_type",  ## Only one value
    "country",
    "manufacturer",
    "mobile_classification",
    "city",
]

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUTPUT_ROOT = CONF["path"]["interim_data_root"]


class BaseFeatures(Feature):

    @classmethod
    def extract_feature(cls, df, save=False):          
        logging.info("Keeping the base features")

        if save:
            os.makedirs(OUTPUT_ROOT, exist_ok=True)
            df_cat = df[IDS + USER_FEATURES]
            # This step is useful to identify categorical features
            df_cat[USER_FEATURES] = df_cat[USER_FEATURES].astype(str)
            df_cat.to_parquet(
                os.path.join(OUTPUT_ROOT, "base_features.parquet"),
                index=False
            )

        return df[IDS + USER_FEATURES]
