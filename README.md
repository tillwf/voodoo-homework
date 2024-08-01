# Voodoo

## Installation

Setup your virtual environment using `pyenv` ([pyenv installer](https://github.com/pyenv/pyenv-installer))

```bash
pyenv install 3.11.5
pyenv local 3.11.5
python -m venv venv
source venv/bin/activate
```

Then install the requirements and the package locally:


```
make install
```

## Setup

The files `train_samples.parquet` and `test_samples.parquet` must be in the folder `data/raw`

## Problem description

We want to predict the user LTV for an app at 120 days after the installation using only its features and what the user did during the first 24 hours.

### Context

In the context of gaming apps, revenue can come from ads watched or in-app purchases.

### Metric

- Business Metric / Online Metric

The business metric would be the total amount of revenue generated by actions made based on the predictions. In this use case, it would be impossible to compute them.


- ML Metric / Offline Metric

The metric to evaluate the performance of the algorithm would be the difference between the actual d120 revenue and the prediction.

In the trainset, 20% of users generate 0$.
The other revenue are spread from 0.0002$ to thousands of dollars. That is why we will use the log of the revenue to compute the loss of the algorithm (using a simple MSE).


### Data

The data consist in 7 month of d0 metrics, user features and the d120 total revenue (also split between ad and iap) (see Raw Features Section)

#### Unique Id

In the data, the are no users in more than on app. The constructed `cohort` value using a hash of the tuple (`install_date`, `app_id`, `campaign_id`, `advertiser_id`) will not be used.

####  Label

In this approach we will only use the `d120_rev`. As this revenue can be split in `d120_iap_rev` and `d120_ad_rev`, these two value could be seperate targets.

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
| **iap_coins_rev** | INTEGER | Time series features from: Revenue from in-app purchases of coins |
| **iap_coins_count** | INTEGER | Time series features from: Number of coin items bought by the user |
| **iap_ads_rev** | INTEGER | Time series features from: Revenue from in-app purchases related to ads (for instance paying to remove ads) |
| **iap_ads_count** | INTEGER | Time series features from: Number of ad items bought by the user |
| **ad_count** | INTEGER | Time series features from: Number of ads viewed by the user |
| **session_count** | INTEGER | Time series features from: Number of sessions done by the user, a session is defined by the moment a user opens the app |
| **game_count** | INTEGER | Time series features from: Number of games played; you can have multiple games played within a session |
| **current_level** | INTEGER | Time series features from: User's current game level |
| **session_length** | INTEGER | Time series features from: Sum of every session length |
| **coins_spend_sum** | INTEGER | Time series features from: Total amount of coins spent by the user |
| **booster_used_count** | INTEGER | Time series features from: Number of boosters used by the user |
| **rv_shown_count** | INTEGER | Time series features from: Number of rewarded videos shown to the user |

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

The neural network is implemented using Tensorflow to be able to visualize easily the training process using Tensorboard, to save and use the model quickly and to be able to complexify it without changing too much the code. It eases also the normalization of numerical features and the handling of categorical features as it will be embed in the graph.

```bash
make train
```

To check the training live, launch Tensorboard:

```
tensorboard --logdir logs
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

- Predict ad revenue and iap revenue. Then use a ML layer above to make the final prediction
- Change the loss to more meaningful metrics like WMAPE
- Fine-tune parameters: leaning rate, network architecture etc.
- Complete unit tests


## Deployement


Let's say we already have a data warehouse where the event are stored and updated live if needed.
The different steps for a complete (Kubeflow) pipeline and a deployment would be:

- **Feature Engineering**: from the data warehouse, construct all the feature needed. It could be done with SQL queries (for instance using `BigQuery`'s power) or `Beam` with `Spark` to have a more sustainable codebase. The features will be store in databases like SQL for the training part, or in file like `Parquet` or `TFRecords`. For the serving part, the features are computed on the fly using fast reading database (like in memory database).

- **Model Training**: once we have our features, if we use a neural network, we could deploy our code in a kubernetes pod using our Dockerfile, and launch multiple trainings (for GridSearch for instance). The evolution of the trainings could be follow using `Tensorboard`. At the end, the models would be saved in `GCS` or `S3` to be call later.

- **Offline evaluation**: once our model is trained, we want to observe its performances on a testset. We could use the `ML.PREDICT` function of `BigQuery` to apply our model to our testset stored in `BigQuery` and then plug a `Google Data Studio` on the results to have a proper dashboard. With this, we could easily compared multiple algorithms on the same dashboard and choose the best one to put in production.

- **Serving**: the serving could be done in two different ways. Either you make your predictions in batch every night and store them in a big key-value database which will be called each time we need a prediction. Or the predictions are computed online.
  - *Offline approach*: with few users, this could be a good solution with very good response time. The generation of the feature for the testset will be very easy as it would be exactly the same code as the traiset generation. The only problem would be the lack of contextual feature: time of the day, device, live popularity.
  - *Online approach*: slower (compute the feature then query the model) and more complicated to maintain (multiple services), this approach will be usually more accurate and would use every live and contextual signals. One problem will be the computation of the features live which can be tricky.

- **Online evaluation**
  - Monitor the services: using `Grafana` dashboard or `Datadog` we could observe the number of query to our service, the response time per step (feature construction, predictions), the total response time and the ressources usage.
  - AB Test: to assess the performance of our approach we have to perform an AB Test which will compare our metric to the previous version of the algorithm. It has to be well calibrated to be able to make valuable conclusion (randomization unit, sample size, etc.). The conclusions should be made at the right moment: we should be careful with the first days results and wait to have statistical significance.
