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
    logging.info(f"Loading from {data_path}")
    logging.info(f"Outputting to {output_root}")

    logging.info("Loading raw data")
    df = load_data()

    logging.info("Adding `cohort` column")
    df["cohort"] = df.apply(
        lambda x: hash((
            x["install_date"],
            x["country"],
            x["ad_network_id"],
            x["campaign_id"]
        )),
        axis=1
    )

    logging.info("Saving Files")

    # TRAINSET
    logging.info("\tTrainset")
    os.makedirs(output_root, exist_ok=True)
    train_path = os.path.join(output_root, "train.parquet")
    trainset = df[(
        (df["install_date"] >= TRAIN_START_DATE) &
        (df["install_date"] <= TRAIN_END_DATE)
    )]

    # Saving the data to parquet
    trainset.to_parquet(train_path, index=False)

    # VALIDATION SET
    logging.info("\tValidation set")
    eval_path = os.path.join(output_root, "eval.parquet")
    
    # Saving the data to parquet
    evaluation_set = df[(
        (df["install_date"] >= EVAL_START_DATE) &
        (df["install_date"] <= EVAL_END_DATE)
    )]

    # Saving the data to parquet
    evaluation_set.to_parquet(eval_path, index=False)

    # TESTSET
    logging.info("\tTestset")
    test_path = os.path.join(output_root, "test.parquet")

    # Keep one day before the starting date to generate the negative post
    testset = df[(
        (df["install_date"] >= TEST_START_DATE) &
        (df["install_date"] <= TEST_END_DATE)
    )]

    # Saving the data to parquet
    testset.to_parquet(test_path, index=False)
    
    # Sanity Check
    train_users = pd.read_parquet(train_path)
    logging.info(f"Train Size: {len(train_users)} ({train_users.user_id.nunique()} distinct users)")

    eval_users = pd.read_parquet(eval_path)
    logging.info(f"eval Size: {len(eval_users)} ({eval_users.user_id.nunique()} distinct users)")

    test_users = pd.read_parquet(test_path)
    logging.info(f"test Size: {len(test_users)} ({test_users.user_id.nunique()} distinct users)")
