import logging
import os
import pandas as pd

from voodoo_homework.config import load_config
from voodoo_homework.features.feature import Feature

IDS = [
    "trackable_id",
    "user_id",
    "tracker_created_at"
]

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUTPUT_ROOT = CONF["path"]["interim_data_root"]


class ExtraFeatures(Feature):

    @classmethod
    def extract_feature(cls, df, save=False):
        """Compute extra features"""
        logging.info("Adding extra features")

        # User children age in year to match with `author_children_age_year`
        df["user_children_age_year"] = df.user_children_age_month.apply(lambda x: [i//12 for i in x])

        # Author has same age children
        df["author_has_same_age_children"] = df.apply(
            lambda x: len(set(x.author_children_age_year).intersection(set(x.user_children_age_year))) > 0,
            axis=1
        )

        # Author has same age children
        df["author_has_same_age_month_children"] = df.apply(
            lambda x: len(set(x.author_children_age_month).intersection(set(x.user_children_age_month))) > 0,
            axis=1
        )

        # Author has older children than user
        df["author_has_older_children"] = df.apply(
            lambda x: max(x.author_children_age_year, default=-1) > max(x.user_children_age_year, default=-1),
            axis=1
        )

        # Time since first commit

        EXTRA_COLS = [
            "author_has_same_age_children",
            "author_has_same_age_month_children",
            "author_has_older_children"
        ]

        if save:
            os.makedirs(OUTPUT_ROOT, exist_ok=True)
            df[IDS + EXTRA_COLS].to_parquet(
                os.path.join(OUTPUT_ROOT, f"extra_features.parquet"),
                index=False
            )

        return df[IDS + EXTRA_COLS]
