import numpy as np
from typing import Union, Dict
def split_score(X, y, feat_idxs, n_bins, mode, classes, weights=None):
    """
    Calculate the score of a split for each feature and threshold.

    Args:
        X: input data
        y: target data
        feat_idxs: indices of the features to be considered
        n_bins: number of bins to consider for each feature
        mode: classification or regression
        classes: classes of the classification
        weights: weights of the samples, if None, no weighting is done
            If weights is given, should be a dictionary with the class as key
            and the weight as value
    """
    n_classes = len(classes)
        # these are the global classes
    local_score = []
        # dimensionality is feat_idxs x n_bins
    for feat_idx in feat_idxs:
        X_column = X[:, feat_idx]
        tmp_feat = []
        for thr in range(n_bins):
            left_idxs, right_idxs = _split(X_column, thr)
            left_y = y[left_idxs]
            right_y = y[right_idxs]
            if mode == 'classification':
                # we assume only 2 classes exist
                # this is why we simply have left and right
                len_y = len(y) if weights is None else np.sum(np.vectorize(weights.get)(y))
                    # if we have weights, we sum up the weights of the samples
                    # otherwise we just take the number of samples
                score = _gini_split(left_y, right_y, n_classes, len_y, weights=weights)
            else:
                if weights is not None:
                    raise ValueError('Weights are not supported for regression.')
                score = _mse_split(y, left_y, right_y)
            tmp_feat.append(score)
        local_score.append(tmp_feat)
    return local_score

def _split(X_column, split_thr):
    # the values in X mean membership of the sample to the bin (0, 1, 2, ...)
    # the left_idxes are the indices part of this bin and all bins to the left
    # the right_idxes are the indices part of all bins to the right
    left_idxs = np.where(X_column <= split_thr)[0]
    right_idxs = np.where(X_column > split_thr)[0]
    return left_idxs, right_idxs

def _gini_split(left_y: np.ndarray, right_y: np.ndarray,
                n_classes: int, total_y: int,
                weights: Union[None, Dict[Union[int, str], int]]=None) -> np.floating:
    """
    Calculate the gini impurity of a split.
    Formula is:
        gini = #samples_left / #samples * gini_left + #samples_right / #samples * gini_right
        gini_left and right are calculated via the gini impurity formula:
        gini_impurity = 1 - sum(p_i^2), where p_i is the probability of class i
    practically, this is calculated as:
        gini_impurity = 1 - sum_class_i((#samples_of_class_i / #samples)^2)
    if weights are given, then we go from each sample being represented by a one
    to each sample being represented by it's weight.
    This changes especially the calculations that previously only took the number
    of samples into using the accumulated weight of these samples.
    1. The gini impurity calculation changes:
        gini_impurity = 1 - sum_class_i((sum_weight_of_samples_of_class_i / sum_weight_of_samples)^2)
    2. The gini score formula changes:
        gini = sum_weight_of_samples_left / sum_weight_of_samples * gini_left + sum_weight_of_samples_right / sum_weight_of_samples * gini_right

    Args:
        left_y: y values of the left node
        right_y: y values of the right node
        n_classes: number of classes
        total_y: number of samples in the parent node. If weights are given,
            this is the sum of the weights over all samples
        weights: weights of the samples, if None, no weighting is done
    Returns:
        gini score of the split
    """
    if len(left_y) == 0 or len(right_y) == 0:
        return np.float64(1.0)
    left_weights = None if weights is None else np.vectorize(weights.get)(left_y)
        # runs weights.get on each element of left_y constructing an np.array
        # we therefore get an np.array of length samples with the weight for
        # each sample as values
    total_left = len(left_y) if left_weights is None else np.sum(left_weights)
    gini_left = 1.0 - np.sum((np.bincount(left_y.astype('int'), minlength=n_classes, weights=left_weights) / total_left) ** 2)
        # with np.bincount we get an array of length n_classes with the number of
        # samples of each class as values and the index as the class
        # we then divide by the number of samples to get the probability of each class
        # if we have weights, we add up the weights of the samples of each class
        # instead of adding one for each sample
    right_weights = None if weights is None else np.vectorize(weights.get)(right_y)
    total_right = len(right_y) if right_weights is None else np.sum(right_weights)
    gini_right = 1.0 - np.sum((np.bincount(right_y.astype('int'), minlength=n_classes, weights=right_weights) / total_right) ** 2)
    gini = (total_left / total_y) * gini_left + (total_right / total_y) * gini_right
        # we add the two gini impurities weighted by the number of samples/weights by total weight
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
