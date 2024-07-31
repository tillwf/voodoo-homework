# Voodoo

## Installation

Setup your virtual environment using `pyenv` ([pyenv installer](https://github.com/pyenv/pyenv-installer))

```bash
pyenv install 3.11.5
pyenv local 3.11.5
python -m venv venv
source venv/bin/activate
```

Then install the requirements and the package locally


```
make install
```

## Setup

The files `train_samples.parquet` and `test_samples.parquet` must be in the folder `data/raw`

## Problem description

We want to predict the user LTV for an app at 120 days after the installation using only its features and what the user did during the first 24 hours.

### Context


### Metric

- Business Metric / Online Metric


- ML Metric / Offline Metric


### Data

The data consist in 6 month of d0 metrics and the d120 total revenue (also split between ad and iap)

#### Unique Id

In the data, the are no users in more than on app. The constructed `cohort` value using a hash of the tuple (`install_date`, `app_id`, `campaign_id`, `advertiser_id`) is not used.

####  Label

`d120_rev`
Can be also split in `d120_iap_rev` and `d120_ad_rev`

#### Raw Features

- User features

| **Feature** | **Type** | **Description** |
|-------------|----------|-----------------|
| **app_id**  | STRING | Unique identifier for each application |
| **install_date** | STRING | Date when the user installed the app |
| **platform** | STRING | Operating system platform of the user's device (iOS or Android) |
| **is_optin** | STRING | Indicates if the user opted in for personalized ads or other services |
| **game_type** | STRING |   |
| **country** | STRING | Country where the user downloaded the app |
| **manufacturer** | STRING | Manufacturer of the user's device |
| **mobile_classification** | STRING | Classification of the mobile device |
| **city** | STRING | City where the user downloaded the app |


- Extra features

| **Feature** | **Type** | **Description** |
|-------------|----------|-----------------|
| **install_month**  | INTEGER | month of installation |
| **install_day_of_month**  | INTEGER | day of month of installation |
| **install_day_of_week**  | INTEGER | day of week of installation |


- Engagement features

| **Feature** | **Type** | **Description** |
|-------------|----------|-----------------|
| **iap_count** | INTEGER | Time series features from: Number of items bought by the user |
| **iap_coins_rev** | INTEGER | Time series features from: Revenue from in-app purchases of
coins |
| **iap_coins_count** | INTEGER | Time series features from: Number of coin items bought by
the user |
| **iap_ads_rev** | INTEGER | Time series features from: Revenue from in-app purchases related to
ads (for instance paying to remove ads) |
| **iap_ads_count** | INTEGER | Time series features from: Number of ad items bought by the
user |
| **ad_count** | INTEGER | Time series features from: Number of ads viewed by the user |
| **session_count** | INTEGER | Time series features from: Number of sessions done by the
user, a session is defined by the moment a user opens the app |
| **game_count** | INTEGER | Time series features from: Number of games played; you can have
multiple games played within a session |
| **current_level** | INTEGER | Time series features from: User's current game level |
| **session_length** | INTEGER | Time series features from: Sum of every session length |
| **coins_spend_sum** | INTEGER | Time series features from: Total amount of coins spent
by the user |
| **booster_used_count** | INTEGER | Time series features from: Number of boosters
used by the user |
| **rv_shown_count** | INTEGER | Time series features from: Number of rewarded videos
shown to the user |

- Targets

| **Feature** | **Type** | **Description** |
|-------------|----------|-----------------|
| **d120_rev** | FLOAT | Total revenue generated on days 120 |
| d120_iap_rev | FLOAT | Cumulated IAP revenue generated on days 120 |
| d120_ad_rev | FLOAT | Cumulated ad revenue generated on days 120 |

## Commands

### Help

```bash
python -m voodoo_homework
```

Will display all possible commands with their description. You can display each command documentation with:

```bash
python -m voodoo_homework <command> --help
```

### Dataset Creation

Using the raw data we want to make a train/validation/test split based on the column `install_date`
For the default values you can do:

```bash
make dataset
```

The train and validation set are all the lines in the time range specified in the configuration file.

No features are computed yet.

### Feature Engineering

We want to be able to compute our features separatly by "category". It enables to compute them in parallel and join them only at the end.

To build every group of features run:

```
make build-features
```

and to merge them into a unified parquet file launch:

```
make merge-features
``` 

#### User Features

Here we just transform the type of the user's feature to be handle as categorical value later in the training part.

#### Extra Features

These feature are there to show how features can be computed separatly and merge afterwards. We just extract the month, day of month and day of week from the `install_date` feature.

#### Engagement Features

Engagement features are computed using time series package `tsfresh` and the config file value `historical_data_points` which is the list of data points to use:

- "0,3,10": to use `d0`, `d3` and `d10`
- "0": to use only `d0` point

it will be useful when we will train models on more data than the first 24h.


### Train the model

The Linear Regression is implemented using Tensorflow to be able to visualize easily the training process using Tensorboard, to save and use the model quickly and to be able to complexify it without changing too much the code. It eases also the normalization of numerical features and the handling of categorical features as it will be embed in the graph.

```bash
make train
```

### Make predictions

Save the predictions and print the performance

```bash
make predictions
```

### Make tests

```bash
make tests
```

or

```bash
pytest tests
```

## Future Work


## Alternatives


## Deployement
