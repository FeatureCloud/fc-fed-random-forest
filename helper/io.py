import bios
import pandas as pd
from skops.io import dump
import shutil

INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'


def read_config():
    config = bios.read(f'{INPUT_DIR}/config.yml')['fc-rand-forest']
    config_input = config['input']
    train = config_input[0]['train']
    test_input = config_input[1]['test']

    config_output = config['output']
    pred = config_output[0]['pred']
    test_output = config_output[1]['test']

    config_format = config['format']
    sep = config_format[0]['sep']
    label_col = config_format[1]['label_col']

    config_split = config['split']
    split_mode = config_split[0]['mode']
    split_dir = config_split[1]['dir']

    n_estimators = int(config['n_estimators'])
    max_depth = int(config['max_depth'])
    max_features = config['max_features']
    bootstrap = bool(config['bootstrap'])
    prediction_mode = config['mode']
    random_state = config['random_state']

    return train, test_input, pred, test_output, sep, label_col, split_mode, split_dir, n_estimators, max_depth, max_features, bootstrap, prediction_mode, random_state


def read_files(train: str, test_input: str, label_col: str):
    train = pd.read_csv(f'{INPUT_DIR}/{train}')
    test = pd.read_csv(f'{INPUT_DIR}/{test_input}')
    X_train = train.drop(label_col, axis=1)
    X_test = test.drop(label_col, axis=1)
    y_train = train.loc[:, label_col]
    y_test = test.loc[:, label_col]

    if isinstance(X_train, pd.DataFrame):
        X = X_train.values
    if isinstance(y_train, pd.Series):
        y = y_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    return X, y, X_test, y_test


def write_model(rf_model):
    dump(rf_model, f'{OUTPUT_DIR}/rf_model.skops')


def write_config():
    shutil.copyfile(f'{INPUT_DIR}/config.yml', f'{OUTPUT_DIR}/config.yml')


def write_pred(y_pred, y_true, output_pred_filename, sep):
    df = pd.concat([pd.DataFrame(y_pred), y_true], axis=1)
    df.columns = ['pred', 'true']
    df.to_csv(f'{OUTPUT_DIR}/{output_pred_filename}', sep=sep, index=False)


def write_test(input_test_filename, sep, output_test_filename, target_col):
    test = pd.read_csv(f'{INPUT_DIR}/{input_test_filename}', sep=sep).drop(target_col, axis=1)
    test.to_csv(f'{OUTPUT_DIR}/{output_test_filename}', sep=sep, index=False)


def read_test(input_test_filename, sep):
    return pd.read_csv(f'{INPUT_DIR}/{input_test_filename}', sep=sep)
