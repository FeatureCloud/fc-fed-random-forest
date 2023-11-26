# Histogram-Based Federated Random Forest FeatureCloud App

## Description
A Random Forest FeautureCloud app, allowing a federated computation of the random forest algorithm.
Supports both classification and regression.

## Input
- train.csv containing the local training data (columns: features; rows: samples)
- test.csv containing the local test data


## Output
- pred.csv containing the predicted class or value 
- train.csv containing the local training data
- test.csv containing the local test data

## Workflows
Can be combined with the following apps:
- Pre: Cross Validation, Normalization, Feature Selection
- Post: Regression Evaluation, Classification Evaluation

## Config
Use the config file to customize your training. Just upload it together with your training data as `config.yml`
```
fc_random_forest:
  input:
    train: "train.csv"
    test: "test.csv"
  output:
    pred: "pred.csv"
    test: "test.csv"
  format:
    sep: ","
    label: "target"
  split:
    mode: directory # directory if cross validation was used before, else file
    dir: data # data if cross validation app was used before, else .
  mode: classification # classification or regression
  n_estimators: 100 # number of trees in the forest
  max_depth: 10 # maximum depth of the tree
  min_samples_split: 2 # minimum number of samples required to split an internal node
  min_samples_leaf: 1 # minimum number of samples required to be at a leaf node
  max_features: 'sqrt' # number of features to consider when looking for the best split
  max_samples: 0.75 # number of samples to draw from X to train each base estimator
  bootstrap: True # whether bootstrap samples are used when building trees
  n_bins: 10 # number of bins used for each feature
  quantile: 0,1 # feature indices for quantile binning
  random_state: 42 # random state for reproducibility
  oob: True # use out-of-bag error to weight decision trees
```
