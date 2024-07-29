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

The files `train_samples.parquet` and `test_samples.parquet` must be in the folder `data/raw`

## Problem description

### Context


### Metric

- Business Metric / Online Metric


- ML Metric / Offline Metric


### Data

#### Unique Id



####  Label


#### Raw Features

- User features

| **Feature** | **Type** | **Description** |
|-------------|----------|-----------------|

- Engagement features

| **Feature** | **Type** | **Description** |
|-------------|----------|-----------------|

- Label

| **Feature** | **Type** | **Description** |
|-------------|----------|-----------------|


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

The train and validation set are all the lines in the time rage specified in the configuration file.

### Feature Engineering

```
make build-features
```

and to merge them into a unified parquet file launch:

```
make merge-features
``` 

### Train the model

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
