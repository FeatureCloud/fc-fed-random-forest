import os
import bios
import pandas as pd
import numpy as np
from typing import Tuple, List

INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'

def read_config():
    config_name = "config.yml"
    if not os.path.exists(f'{INPUT_DIR}/{config_name}'):
        config_name = "config.yaml"
    config = bios.read(f'{INPUT_DIR}/{config_name}')['fc-rand-forest']
    config_input = config['input']
    train = config_input['train']
    test_input = config_input['test']

    config_output = config['output']
    pred = config_output['pred']
    test_output = config_output['test']

    config_format = config['format']
    sep = config_format.get('sep', ',')
    label_col = config_format['label_col']

    config_split = config['split']
    split_mode = config_split['mode']
    split_dir = config_split['dir']

    # Parameters RandomForest
    n_estimators = int(config.get('n_estimators', 100))
    criterion = config.get('criterion', 'gini')
    max_depth = int(config.get('max_depth', 10))
    min_samples_split = config.get('min_samples_split', 2)
    min_samples_leaf = config.get('min_samples_leaf', 1)
    max_features = config.get('max_features', 'sqrt')
    bootstrap = config.get('bootstrap', True)
    max_samples = config.get('max_samples', None)
    random_state = int(config.get('random_state', 0))
    weight_classes_bool = config.get('use_weighted_classes', False)

    prediction_mode = config['mode']
    quantile = config.get('quantile', [])
    n_bins = config['n_bins']

    oob = config.get('oob', False)

    output_mode = config.get('output_mode', 'model')
    if output_mode not in ['model', 'pred', 'model+pred']:
        raise ValueError('Output mode must be either "model" or "pred" or "model+pred".')

    return train, test_input, pred, test_output, sep, label_col, split_mode, split_dir, \
            n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, \
            max_features, bootstrap, max_samples, random_state, prediction_mode, quantile, \
            n_bins, oob, weight_classes_bool, output_mode

def convert_to_np(data):
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.to_numpy()
    else:
        raise ValueError("Input data is not a Pandas Series or DataFrame.")

def read_files(train: str, test_input: str, sep: str, label_col: str) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index[str]]:
    """
    Reads the train and test data files and returns the data as numpy arrays.
    Ensures that the feature names in the train and test data match, if not
    raises an ValueError.
    Args:
        train: str: Name of the train file.
        test_input: str: Name of the test file.
        sep: str: Delimiter used in the files.
        label_col: str: Name of the label column.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
            X: np.ndarray: Train data.
            y: np.ndarray: Train labels.
            X_test: np.ndarray: Test data.
            y_test: np.ndarray: Test labels.
            feature_names: List[str]: Feature names.
    Raises:
        ValueError: If the feature names in the train and test data do not match.
    """
    train_df = pd.read_csv(f'{INPUT_DIR}/{train}', sep=sep)
    test = pd.read_csv(f'{INPUT_DIR}/{test_input}', sep=sep)
    X_train = train_df.drop(label_col, axis=1)
    X_test = test.drop(label_col, axis=1)
    y_train = train_df.loc[:, label_col]
    y_test = test.loc[:, label_col]
    # check if we have any missing values and raise an error if yes
    if X_train.isnull().values.any() or y_train.isnull().any():
        raise ValueError("Missing values in train data.")
    if X_test.isnull().values.any() or y_test.isnull().any():
        raise ValueError("Missing values in test data.")
    feature_names = X_train.columns
    if feature_names != X_test.columns:
        raise ValueError("Feature names in train and test data do not match.")
    # MISSING_VALUES_SUPPORT: remove columns without ANY values, also
    # remove them from feature_names
    X = convert_to_np(X_train)
    y = convert_to_np(y_train)
    X_test = convert_to_np(X_test)
    y_test = convert_to_np(y_test)

    return X, y, X_test, y_test, feature_names