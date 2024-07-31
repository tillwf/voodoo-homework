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
from tensorflow.keras.layers import Input, Dense, Concatenate, Embedding, Flatten, BatchNormalization, Lambda
from tensorflow.keras.layers import Normalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers.experimental.preprocessing import Normalization, StringLookup, CategoryEncoding

from voodoo_homework.config import load_config
from voodoo_homework.utils import load_datasets
from voodoo_homework.utils import load_features

from voodoo_homework.features.base_features import BaseFeatures
from voodoo_homework.features.extra_features import ExtraFeatures
from voodoo_homework.features.time_series_features import TimeSeriesFeatures

from voodoo_homework.models.losses import mean_squared_error_log
from voodoo_homework.models.losses import weighted_mape_tf
from voodoo_homework.models.utils import dataframe_to_dict


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

    cols = set(
        [c for c in cols if not c.startswith("__")] +
        ["user_id", "cohort"]
    )

    # Construct the train and validation set with the features
    numeric_cols = data[cols].select_dtypes(include=['number']).columns.difference(["user_id", "cohort", "game_type"]).tolist()
    categorical_cols = data[cols].select_dtypes(exclude=['number']).columns.difference(["user_id", "cohort", "game_type"]).tolist()

    X_train = pd.merge(
        X_train,
        data[cols],
        on=["user_id"]
    )
    y_train = y_train

    X_validation = pd.merge(
        X_validation,
        data[cols],
        on=["user_id"]
    )

    # Extract column names
    numeric_cols = data.select_dtypes(include=['number']).columns.difference(["user_id", "cohort"]).tolist()
    categorical_cols = data.select_dtypes(exclude=['number']).columns.difference(["user_id", "cohort"]).tolist()

    # Create input layers for numerical and categorical columns
    inputs = []
    encoded_features = []

    # Normalize numerical features
    for col in numeric_cols:
        logging.info(col)
        numeric_input = Input(shape=(1,), name=col)
        normalization_layer = Normalization()(numeric_input)
        inputs.append(numeric_input)
        encoded_features.append(normalization_layer)

    # Encode categorical features
    for col in categorical_cols:
        logging.info(col)
        categorical_input = Input(shape=(1,), name=col, dtype=tf.string)

        # Create and adapt the StringLookup layer
        lookup_layer = StringLookup(output_mode='int', vocabulary=X_train[col].unique())
        encoded_indices = lookup_layer(categorical_input)

        # Create the CategoryEncoding layer
        encoding_layer = CategoryEncoding(num_tokens=X_train[col].nunique() + 1, output_mode='one_hot')
        encoded_feature = encoding_layer(encoded_indices)

        inputs.append(categorical_input)
        encoded_features.append(encoded_feature)

    # Concatenate all features
    all_features = Concatenate()(encoded_features)

    # Define the rest of the model
    x = Dense(128, activation='relu')(all_features)
    x = Dense(64, activation='relu')(x)
    output = Dense(1)(x)
    clipped_output = Lambda(lambda y_pred: tf.clip_by_value(y_pred, EPSILON, 1e10))(output)

    # Create the model
    model = Model(inputs=inputs, outputs=clipped_output)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=mean_squared_error_log
    )

    # Model summary
    model.summary()

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

    # Prepare data for training (convert X_train to dictionary)
    X_train_dict = dataframe_to_dict(X_train)
    X_validation_dict = dataframe_to_dict(X_validation)

    # Launch the train and save the loss evolution in `history`
    history = model.fit(
        X_train_dict,
        y_train.values,
        callbacks=callbacks,
        epochs=2,
        validation_data=(
            X_validation_dict,
            y_validation.values
        )
    )

    # Save the model
    logging.info("Saving Model")
    model.load_weights(best_model_file)
    model.save(os.path.join(models_root, "final_model"))
