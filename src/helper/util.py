"""
Contains small helper functions for the project.
"""
import numpy as np
import pandas as pd
# pylint: disable=invalid-name

def validate_input_data(X_: np.ndarray, y_: np.ndarray, X_test_: np.ndarray, y_test_: np.ndarray, num_features: int):
    """
    Checks if the input data is valid.
    Args:
        X_: Train data
        y_: Train labels
        X_test_: Test data
        y_test_: Test labels
        num_features: Number of features in the data (in X_ and X_test_)
    Raises:
        ValueError: If the number of features in train and test data is not equal.
        ValueError: If the data is not completely numerical.
        ValueError: If the labels are not completely numerical and 1d arrays.
        ValueError: If the number of features in all splits is not equal
            (not equal to given num_features).
    Returns:
        None if no errors are raised.
    """
     # Ensure that test and train are compatible
    if X_.shape[1] != X_test_.shape[1]:
        raise ValueError('Number of features in train and test data must be equal.')
    # Ensure that X_ and X_test_ are completely numerical
    if not np.issubdtype(X_.dtype, np.number) or not np.issubdtype(X_test_.dtype, np.number):
        raise ValueError('All columns in test and train data must be numerical.')
    # Ensure that y_ and y_test_ are completely numerical and 1d arrays
    if not np.issubdtype(y_.dtype, np.number) or not np.issubdtype(y_test_.dtype, np.number):
        raise ValueError('All columns in test and train labels must be numerical.')
    if y_.ndim != 1 or y_test_.ndim != 1:
        raise ValueError('label column could not be extracted to an 1d array.')
    if X_.shape[1] != num_features or X_test_.shape[1] != num_features:
        raise ValueError('Number of features in all splits must be equal.')

def convert_to_np(data):
    """
    Simple function to convert a Pandas Series or DataFrame to a numpy array.
    Failed conversion raises a ValueError.
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.to_numpy()
    else:
        raise ValueError("Input data is not a Pandas Series or DataFrame.")
