import click
import fastparquet
import json
import logging
import numpy as np
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
from voodoo_homework.features.time_series_features import TimeSeriesFeatures

CONF = load_config()
DATA_PATH = CONF["path"]["input_data_path"]
MODELS_ROOT = CONF["path"]["models_root"]
LOGS_ROOT = CONF["path"]["logs_root"]
INTERIM_ROOT = CONF["path"]["interim_data_root"]
OUTPUT_ROOT = CONF["path"]["output_data_root"]

FEATURE_DICT = {
    "base_features": BaseFeatures,
    "extra_features": ExtraFeatures,
    "time_series_features": TimeSeriesFeatures
}

FEATURE_DEFINITIONS = CONF["feature_definitions"]

EPOCH = CONF["model"]["epoch"]
PATIENCE = CONF["model"]["early_stopping_patience"]

EPSILON = tf.keras.backend.epsilon()


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

    def mean_square_error_log(y_true, y_pred):
        y_true = tf.maximum(tf.cast(y_true, tf.float32), EPSILON)
        #y_pred = tf.maximum(tf.cast(y_pred, tf.float32), EPSILON)

        y_true_log = tf.math.log(y_true)
        y_pred_log = tf.math.log(y_pred)

        return tf.keras.losses.mean_squared_error(y_true_log, y_pred_log)

    def weighted_mape_tf(y_true, y_pred):
        tot =tf.cast(tf.reduce_sum(y_true), tf.float32)
        tot = tf.clip_by_value(tot, clip_value_min=1, clip_value_max=10)
        wmape = tf.realdiv(
            tf.reduce_sum(
                tf.abs(
                    tf.subtract(
                        tf.cast(y_true, tf.float32),
                        tf.cast(y_pred, tf.float32)
                    )
                )
            ),
            tf.cast(tot, tf.float32)
        ) * 100

        return wmape

    logging.info("Training Model")
    X_train, X_validation, _ = load_datasets()
    y_train = X_train.pop("d120_rev").astype(int)
    y_validation = X_validation.pop("d120_rev").astype(int)

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
        on=["user_id", "cohort"]
    ).replace({False: 0, True: 1})\
     .select_dtypes(['number']).fillna(0)
   
    X_validation = pd.merge(
        X_validation,
        data[cols],
        on=["user_id", "cohort"]
    ).replace({False: 0, True: 1})\
     .select_dtypes(['number']).fillna(0)

    # Define the model
    # Normalize the numerical features
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_train)

    # Simple Logistic regression
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(units=32, activation='relu'),
        layers.Dense(units=16, activation='relu'),
        layers.Dense(units=1),
        layers.Lambda(lambda y_pred: tf.clip_by_value(y_pred, EPSILON, 1e10))
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=mean_square_error_log
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
    history = model.fit(
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
    model.load_weights(best_model_file)
    model.save(os.path.join(models_root, "final_model"))
