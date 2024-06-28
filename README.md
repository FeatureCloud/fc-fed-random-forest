# Histogram-Based Federated Random Forest FeatureCloud App

## Description
A Random Forest FeautureCloud app, allowing an iterative federated computation of the random forest algorithm.
Supports single-class classification. The multiclass classification and regression are to be added. 

## Input
- train.csv containing the local training data (columns: features; rows: samples)
- test.csv containing the local test data
### Important Note:
- For single-class classification problems, all the entries of both train and test data must be converted to numerical values like float or int. The "Str" type is not compatible. 


## Output
- `pred.csv` containing the predicted class or value 
- `train.csv` containing the local training data
- `test.csv` containing the local test data
- `rfmodel.py` containing the class RandomForest. 
  This class used for the model has the predict(self, X) method that can be used 
  to do your own predictions. It must be loaded (e.g. imported) to be able to
  load the model instance (`rf_model.joblib`)
- `rf_model.joblib` containing the trained mode instance. It has a predict(self, X)
  method to predict a dataset with the same features as the given data for training
  Please note that RandomForest from `rfmodel.py` must be loaded to be able to
  load this class via joblib.load(`rf_model.joblib`)

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
  use_weighted_classes: False # if set to True, all classes are given the same 
                              # importance, no matter how many samples 
                              # are of each class
                              # e.g. if a one class has 1/3 of all samples and
                              # the other class 2/3 of all samples, then
                              # the first class is given a weight of 3
                              # and the second class is given a weight of 1.5
```
