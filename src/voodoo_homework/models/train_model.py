import click
import fastparquet
import json
import logging
import os
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from voodoo_homework.config import load_config
from voodoo_homework.utils import load_datasets
from voodoo_homework.utils import load_features

from voodoo_homework.features.base_features import BaseFeatures
from voodoo_homework.features.extra_features import ExtraFeatures
from voodoo_homework.features.post_popularity import PostPopularity
from voodoo_homework.features.user_post_popularity import UserPostPopularity

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
MODELS_ROOT = CONF["path"]["models_root"]
LOGS_ROOT = CONF["path"]["logs_root"]
INTERIM_ROOT = CONF["path"]["interim_data_root"]
OUTPUT_ROOT = CONF["path"]["output_data_root"]

FEATURE_DICT = {
    "base_features": BaseFeatures,
    "extra_features": ExtraFeatures,
    "post_popularity": PostPopularity,
    "user_post_popularity": UserPostPopularity
}

FEATURE_DEFINITIONS = CONF["feature_definitions"]

EPOCH = CONF["model"]["epoch"]
PATIENCE = CONF["model"]["early_stopping_patience"]


@click.group()
def train():
    pass


@train.command()
@click.option(
    '--output-root',
    type=str,
    default=OUTPUT_ROOT,
    help='Path of output folder, default is {}'.format(
        OUTPUT_ROOT
    )
)
@click.option(
    '--models-root',
    type=str,
    default=MODELS_ROOT,
    help='Path of models folder, default is {}'.format(
        MODELS_ROOT
    )
)
@click.option(
    '--logs-root',
    type=str,
    default=LOGS_ROOT,
    help='Path of logs folder, default is {}'.format(
        LOGS_ROOT
    )
)
@click.option(
    '--features',
    type=str,
    multiple=True,
    default=list(FEATURE_DICT.keys()),
    help='Features used for the training, default is {}'.format(
        list(FEATURE_DICT.keys())
    )
)
def train_model(models_root, output_root, logs_root, features):
    logging.info("Training Model")

    X_train, X_validation, _ = load_datasets()
    y_train = X_train.pop("has_been_opened").astype(int)
    y_validation = X_validation.pop("has_been_opened").astype(int)

    # Load all the features
    data = load_features()

    # Select the features base on the `features` parameter
    features_names = [
        f
        for feature_group in features
        for f in FEATURE_DEFINITIONS[feature_group]
        if FEATURE_DICT.get(feature_group)
    ]
    cols = []
    for filename in features_names:
        path = os.path.join(INTERIM_ROOT, f"{filename}.parquet")
        cols += fastparquet.ParquetFile(path).columns

    cols = set([c for c in cols if not c.startswith("__")])

    # Construct the train and validation set with the features
    X_train = pd.merge(
        X_train,
        data[cols],
        on=["trackable_id", "user_id", "tracker_created_at"]
    ).replace({False: 0, True: 1})\
     .select_dtypes(['number']).fillna(0)
   
    X_validation = pd.merge(
        X_validation,
        data[cols],
        on=["trackable_id", "user_id", "tracker_created_at"]
    ).replace({False: 0, True: 1})\
     .select_dtypes(['number']).fillna(0)

    # Define the model
    # Normalize the numerical features
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_train)

    # Simple Logistic regression
    linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(
            units=1,
            activation='sigmoid',
            input_dim=X_train.shape[1]
        )
    ])
    linear_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
    )

    # Add callbacks to be able to restart if a process fail, to
    # save the best model and to create a TensorBoard
    callbacks = []

    os.makedirs(models_root, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=logs_root,
        histogram_freq=1,
        write_graph=True,
        write_images=False,
        update_freq=100,
        profile_batch=2,
        embeddings_freq=1
    )
    callbacks.append(tensorboard)

    best_model_file = os.path.join(models_root, "best_model_so_far")
    best_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        best_model_file,
        monitor='val_loss',
        verbose=1,
        save_best_only=True
    )
    callbacks.append(best_model_checkpoint)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=PATIENCE,
        monitor="val_loss"
    )
    callbacks.append(early_stopping)

    # Launch the train and save the loss evolution in `history`
    history = linear_model.fit(
        X_train,
        y_train.values,
        callbacks=callbacks,
        epochs=EPOCH,
        validation_data=(
            X_validation,
            y_validation.values
        )
    )

    # Save the model
    logging.info("Saving Model")
    linear_model.load_weights(best_model_file)
    linear_model.save(os.path.join(models_root, "final_model"))
