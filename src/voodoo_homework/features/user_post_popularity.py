import logging
import os
import pandas as pd

from voodoo_homework.config import load_config
from voodoo_homework.features.feature import Feature

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUTPUT_ROOT = CONF["path"]["interim_data_root"]


class UserPostPopularity(Feature):

    @classmethod
    def extract_feature(cls, df, save=False, windows=["1d", "7d", "28d"]):
        """Compute the popularity of a post for each user using windows"""

        logging.info("Computing the features UserPostPopularity")

        def add_post_past_popularity(window):
            logging.info(f"\tComputing user's past {window} popularity")
            return (
                df[["tracker_created_at", "trackable_id", "user_id", "has_been_opened"]]
                .set_index('tracker_created_at')
                .sort_index()
                .groupby(['trackable_id', "user_id"], sort=False)
                .has_been_opened
                .rolling(window, closed='left')
                .agg({
                    f"user_post_last_{window}_views_count": len,
                    f"user_post_last_{window}_clicks_count": sum,
                    f"user_post_last_{window}_ratio": lambda x: sum(x)/len(x)
                })
                .fillna(0)
                .reset_index())
         
        df_res = pd.DataFrame()
        for window in windows:
            df_temp = add_post_past_popularity(window)
            df_res = pd.concat([df_res, df_temp], axis=0)
            if save:
                os.makedirs(OUTPUT_ROOT, exist_ok=True)
                df_temp.to_parquet(
                    os.path.join(OUTPUT_ROOT, f"user_post_popularity_{window}.parquet"),
                    index=False
                )

        return df_res
