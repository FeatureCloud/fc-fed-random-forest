import numpy as np
from typing import Union

def split_score(X, y, feat_idxs, n_bins, mode, classes, weights=None):
    n_classes = len(classes)
    local_score = []
    for feat_idx in feat_idxs:
        X_column = X[:, feat_idx]
        tmp_feat = []
        for thr in range(n_bins):
            left_idxs, right_idxs = _split(X_column, thr)
            left_y = y[left_idxs]
            right_y = y[right_idxs]
            if mode == 'classification':
                score = _gini_split(left_y, right_y, n_classes, len(y), weights=weights)
            else:
                if weights is not None:
                    raise ValueError('Weights are not supported for regression.')
                score = _mse_split(y, left_y, right_y)
            tmp_feat.append(score)
        local_score.append(tmp_feat)
    return local_score

def _split(X_column, split_thr):
    left_idxs = np.where(X_column <= split_thr)[0]
    right_idxs = np.where(X_column > split_thr)[0]
    return left_idxs, right_idxs

def _gini_split(left_y: np.ndarray, right_y: np.ndarray,
                n_classes: int, len_y: int,
                weights: Union[None, np.ndarray]=None) -> np.floating:
    """
    Calculate the gini impurity of a split.
    Formula is:
        gini = #samples_left / #samples * gini_left + #samples_right / #samples * gini_right
        gini_left and right are calculated via the gini impurity formula:
        gini_impurity = 1 - sum(p_i^2), where p_i is the probability of class i
    practically, this is calculated as:
        gini_impurity = 1 - sum_class_i((#samples_of_class_i / #samples)^2)
    if weights are given, the formula is:
        gini_impurity = 1 - sum_class_i((#samples_of_class_i * weight_of_class_i / sum_class_i(#samples_of_class_i * weight_of_class_i))^2)
    Args:
        left_y: y values of the left node
        right_y: y values of the right node
        n_classes: number of classes
        len_y: number of samples in the parent node
        weights: weights of the samples, if None, no weighting is done
    Returns:
        gini score of the split
    """
    if len(left_y) == 0 or len(right_y) == 0:
        return np.float64(1.0)
    left_weights = None if weights is None else np.vectorize(weights.get)(left_y)
    gini_left = 1.0 - np.sum((np.bincount(left_y.astype('int'), minlength=n_classes, weights=left_weights) / len(left_y.astype('int'))) ** 2)
    right_weights = None if weights is None else np.vectorize(weights.get)(right_y)
    gini_right = 1.0 - np.sum((np.bincount(right_y.astype('int'), minlength=n_classes, weights=right_weights) / len(right_y.astype('int'))) ** 2)
    gini = (len(left_y) / len_y) * gini_left + (len(right_y) / len_y) * gini_right
    return gini

def _mse_split(y, left_y, right_y):
    parent_mse = _mse(y)
    if len(left_y) == 0 or len(right_y) == 0:
        return parent_mse
    left_mse = _mse(left_y)
    right_mse = _mse(right_y)
    mse = parent_mse - 1/len(y) * (len(left_y) * left_mse + len(right_y) * right_mse)
    return mse

def _mse(y):
    return np.mean(np.square(y - np.mean(y)))
