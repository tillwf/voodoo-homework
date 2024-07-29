import click
import json
import logging
import os
import pandas as pd

from voodoo_homework.config import load_config
from voodoo_homework.utils import load_data

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
OUPUT_ROOT = CONF["path"]["output_data_root"]

TRAIN_START_DATE = CONF["dataset"]["train_start_date"]
TRAIN_END_DATE = CONF["dataset"]["train_end_date"]

EVAL_START_DATE = CONF["dataset"]["eval_start_date"]
EVAL_END_DATE = CONF["dataset"]["eval_end_date"]

TEST_START_DATE = CONF["dataset"]["test_start_date"]
TEST_END_DATE = CONF["dataset"]["test_end_date"]


@click.group()
def dataset():
    pass


@dataset.command()
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
    default=OUPUT_ROOT,
    help='Path of output folder, default is {}'.format(
        OUPUT_ROOT
    )
)
@click.option(
    '--train_start_date',
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=TRAIN_START_DATE,
    help='Starting day of the trainset, default is {}'.format(
        TRAIN_START_DATE
    )
)
@click.option(
    '--train_end_date',
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=TRAIN_END_DATE,
    help='Ending day of the trainset, default is {}'.format(
        TRAIN_START_DATE
    )
)
@click.option(
    '--eval_start_date',
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=EVAL_START_DATE,
    help='Starting day of the evalset, default is {}'.format(
        EVAL_START_DATE
    )
)
@click.option(
    '--eval_end_date',
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=EVAL_END_DATE,
    help='Ending day of the evalset, default is {}'.format(
        EVAL_START_DATE
    )
)
@click.option(
    '--test_start_date',
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=TEST_START_DATE,
    help='Starting day of the testset, default is {}'.format(
        TEST_START_DATE
    )
)
@click.option(
    '--test_end_date',
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=TEST_END_DATE,
    help='Ending day of the testset, default is {}'.format(
        TEST_END_DATE
    )
)
def make_dataset(
        data_path,
        output_root,
        train_start_date,
        train_end_date,
        eval_start_date,
        eval_end_date,
        test_start_date,
        test_end_date):
    logging.info("Making Dataset")
    logging.info(data_path)
    logging.info(output_root)

    logging.info("Loading raw data")
    df = load_data()

    logging.info("Saving Files")

    col_to_keep = ["trackable_id", "user_id", "tracker_created_at", "has_been_opened"]

    # TRAINSET
    logging.info("\tTrainset")
    os.makedirs(output_root, exist_ok=True)
    train_path = os.path.join(output_root, "train.parquet")
    trainset = df[(
        (df["tracker_created_at"] >= TRAIN_START_DATE) &
        (df["tracker_created_at"] <= TRAIN_END_DATE)
    )][col_to_keep]

    # Filtering
    # We remove users with few signals which are moreover only negative
    N_SIGNALS = 4

    user_vc = trainset.user_id.value_counts()
    few_signal_users = (user_vc[user_vc <= N_SIGNALS]).index

    pos_signal = trainset[trainset.user_id.isin(few_signal_users)]

    pos_signal.loc[:, "sum_hbo"] = pos_signal.groupby("user_id").has_been_opened.transform(sum)
    few_signal_users_negative = pos_signal[pos_signal["sum_hbo"] == 0].user_id
    trainset = trainset[~trainset.user_id.isin(few_signal_users_negative)]

    # Saving the data to parquet
    trainset.to_parquet(train_path, index=False)

    # VALIDATION SET
    logging.info("\tValidation set")
    eval_path = os.path.join(output_root, "eval.parquet")
    
    # Saving the data to parquet
    df[(
        (df["tracker_created_at"] >= EVAL_START_DATE) &
        (df["tracker_created_at"] <= EVAL_END_DATE)
    )][col_to_keep].to_parquet(eval_path, index=False)

    # TESTSET
    logging.info("\tTestset")
    test_path = os.path.join(output_root, "test.parquet")

    # Keep one day before the starting date to generate the negative post
    testset = df[(
        (df["tracker_created_at"] >= EVAL_END_DATE) &
        (df["tracker_created_at"] <= TEST_END_DATE)
    )]

    # Keep only last day articles
    testset["post_creation_date"] = (testset.tracker_created_at - testset.post_age_in_minutes.apply(lambda x: pd.Timedelta(minutes=x))).dt.date
    testset["post_creation_date_plus_one"] = testset["post_creation_date"].apply(lambda x: x + pd.Timedelta(days=1))
    testset["post_from_yesterday"] = testset.tracker_created_at.dt.date == testset.post_creation_date_plus_one

    ## Keep only line with has_been_opened == 1 and post is from last day
    ## => Divide by 8 the number of post
    pos_testset = testset[
        (testset.has_been_opened) &
        (testset.post_from_yesterday)
    ]

    ## Add all other posts from the day before as negative
    pos_and_neg = []
    for i, row in pos_testset.iterrows():
        current_user_id = row["user_id"]
        current_pos_date = row["tracker_created_at"]
        yesterdays_posts = testset[
            testset.post_creation_date_plus_one == current_pos_date.date()
        ].trackable_id.unique()

        for post_id in yesterdays_posts:
            pos_and_neg.append({
                "user_id": current_user_id,
                "tracker_created_at": current_pos_date,
                "trackable_id": post_id,
                "has_been_opened": 0
            })

    testset = pd.concat([pos_testset[col_to_keep], pd.DataFrame(pos_and_neg)], axis=0)
    
    testset = testset[(
        testset["tracker_created_at"] >= TEST_START_DATE
    )]

    # Saving the data to parquet
    testset[col_to_keep].to_parquet(test_path, index=False)
    
    # Sanity Check
    train_users = pd.read_parquet(train_path)
    logging.info(f"Train Size: {len(train_users)} ({train_users.user_id.nunique()} distinct users)")

    eval_users = pd.read_parquet(eval_path)
    logging.info(f"eval Size: {len(eval_users)} ({eval_users.user_id.nunique()} distinct users)")

    test_users = pd.read_parquet(test_path)
    logging.info(f"test Size: {len(test_users)} ({test_users.user_id.nunique()} distinct users)")
