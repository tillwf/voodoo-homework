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
    "install_date",
    "platform",
    "is_optin",
    "game_type",
    "country",
    "user_country_code",
    "manufacturer",
    "mobile_classification",
    "city",
]

ENGAGEMENT_FEATURES = [
    "iap_count_d0",
    "iap_coins_rev_d0",
    "iap_coins_count_d0",
    "iap_ads_rev_d0",
    "iap_ads_count_d0",
    "ad_count_d0",
    "session_count_d0",
    "game_count_d0",
    "current_level_d0",
    "session_length_d0",
    "coins_spend_sum_d0",
    "booster_used_count_d0",
    "rv_shown_count_d0",
]

REVENUE_AD = [f"d{t}_ad_rev" for t in (0)]  # , 3, 7, 14, 30, 60, 90)]
REVENUE_IAP = [f"d{t}_iap_rev" for t in (0)]  # , 3, 7, 14, 30, 60, 90)]
REVENUE_TOTAL = [f"d{t}_rev" for t in (0)]  # , 3, 7, 14, 30, 60, 90)]

REVENUE = REVENUE_AD + REVENUE_IAP + REVENUE_TOTAL

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUTPUT_ROOT = CONF["path"]["interim_data_root"]


class BaseFeatures(Feature):

    @classmethod
    def extract_feature(cls, df, save=False):          
        logging.info("Keeping the base features")

        if save:
            os.makedirs(OUTPUT_ROOT, exist_ok=True)
            df[IDS + USER_FEATURES + POST_FEATURES].to_parquet(
                os.path.join(OUTPUT_ROOT, f"base_features.parquet"),
                index=False
            )

        return df[IDS + USER_FEATURES + POST_FEATURES]
