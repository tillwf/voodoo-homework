import os
import pandas as pd

from functools import reduce
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

from voodoo_homework.config import load_config
from voodoo_homework.features.feature import Feature

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUTPUT_ROOT = CONF["path"]["interim_data_root"]

IDS = [
    "user_id",
]

ENGAGEMENT_FEATURES = [
    "iap_count",
    "iap_coins_rev",
    "iap_coins_count",
    "iap_ads_rev",
    "iap_ads_count",
    "ad_count",
    "session_count",
    "game_count",
    "current_level",
    "session_length",
    "coins_spend_sum",
    "booster_used_count",
    "rv_shown_count",
]


class TimeSeriesFeatures(Feature):

    @classmethod
    def extract_feature(cls, df, save=False):
        historical_data_points = [
            int(dp)
            for dp in CONF["historical_data_points"].split(",")
        ]

        time_series_features = []
        for feature in ENGAGEMENT_FEATURES:
            columns = sorted([
                f"{feature}_d{d}"
                for d in historical_data_points
            ])
            intermediate_df = df[IDS + columns]
            intermediate_df.columns = IDS + historical_data_points
            # Convert dataframe like
            #    user_id               cohort     0      3
            # 0   408426 -7300573791086353391  10.0   20.0
            # 1  1485764 -5356601982058702779  100.0  30.0
            # To
            #    user_id               cohort    time    value
            # 0   408426 -7300573791086353391       0     10.0
            # 2   408426 -7300573791086353391       3     20.0
            # 3  1485764 -5356601982058702779       0    100.0
            # 4  1485764 -5356601982058702779       3     30.0

            melted_df = pd.melt(
                intermediate_df,
                id_vars=['user_id'],
                var_name='data_point',
                value_name=feature
            ).dropna()

            settings = MinimalFCParameters()
            extracted_features = extract_features(
                melted_df,
                column_id='user_id',
                column_sort='data_point',
                default_fc_parameters=settings
            ).reset_index(drop=True)
            extracted_features["user_id"] = melted_df["user_id"]
            if save:
                os.makedirs(OUTPUT_ROOT, exist_ok=True)
                extracted_features.to_parquet(
                    os.path.join(OUTPUT_ROOT, f"{feature}.parquet"),
                    index=False
                )

            time_series_features.append(extracted_features)

        time_series_features = reduce(
            lambda x, y: pd.merge(x, y, on='user_id'),
            time_series_features
        )

        return time_series_features
