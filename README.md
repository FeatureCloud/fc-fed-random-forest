# Random Forest FeatureCloud App with really federated execution

## Description
A Random Forest FeautureCloud app, allowing a federated computation of the random forest algorithm.
Supports both only classification for now.

## Input
- train.csv containing the local training data (columns: features; rows: samples)
- test.csv containing the local test data


## Output
- pred.csv containing the predicted class or value
- test.csv containing the local test data

## Workflows
Can be combined with the following apps:
- Pre: Cross Validation, Normalization, Feature Selection
- Post: Regression Evaluation, Classification Evaluation

## Config
Use the config file to customize your training. Just upload it together with your training data as `config.yml`
```
fc-rand-forest:
  input:
    - train: 'train.csv'
    - test: 'test.csv'
  output:
    - pred: 'pred.csv'
    - test: 'test.csv'
  format:
    - sep: ','
    - label_col: 'target'
  split:
    - mode: 'file'
    - dir: .
  n_estimators: 10
  max_depth: 5
  max_features: 'sqrt'
  max_samples: 1.0
  bootstrap: True
  mode: 'classification'
  random_state: 42

```
