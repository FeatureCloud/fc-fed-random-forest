import os
import bios
import pandas as pd

INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'

def read_config():
    config_name = "config.yml"
    if not os.path.exists(f'{INPUT_DIR}/{config_name}'):
        config_name = "config.yaml"
    config = bios.read(f'{INPUT_DIR}/{config_name}')['fc-rand-forest']
    config_input = config['input']
    train = config_input[0]['train']
    test_input = config_input[1]['test']

    config_output = config['output']
    pred = config_output[0]['pred']
    test_output = config_output[1]['test']

    config_format = config['format']
    sep = config_format[0].get('sep', ',')
    label_col = config_format[1]['label_col']

    config_split = config['split']
    split_mode = config_split[0]['mode']
    split_dir = config_split[1]['dir']

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

    prediction_mode = config['mode']
    quantile = str(config.get('quantile', ''))
    n_bins = config['n_bins']

    oob = config.get('oob', False)

    return train, test_input, pred, test_output, sep, label_col, split_mode, split_dir, \
            n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, \
            max_features, bootstrap, max_samples, random_state, prediction_mode, quantile, \
            n_bins, oob

def convert_to_np(data):
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.to_numpy()
    else:
        raise ValueError("Input data is not a Pandas Series or DataFrame.")

def read_files(train: str, test_input: str, sep: str, label_col: str):
    train = pd.read_csv(f'{INPUT_DIR}/{train}', sep=sep)
    test = pd.read_csv(f'{INPUT_DIR}/{test_input}', sep=sep)
    X_train = train.drop(label_col, axis=1)
    X_test = test.drop(label_col, axis=1)
    y_train = train.loc[:, label_col]
    y_test = test.loc[:, label_col]

    X = convert_to_np(X_train)
    y = convert_to_np(y_train)
    X_test = convert_to_np(X_test)
    y_test = convert_to_np(y_test)

    return X, y, X_test, y_test