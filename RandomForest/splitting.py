import numpy as np

def split_score(X, y, feat_idxs, n_bins, mode, classes):
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
                score = _gini_split(left_y, right_y, n_classes, len(y))
            else:
                score = _mse_split(y, left_y, right_y)
            tmp_feat.append(score)
        local_score.append(tmp_feat)
    return local_score

def _split(X_column, split_thr):
    left_idxs = np.where(X_column <= split_thr)[0]
    right_idxs = np.where(X_column > split_thr)[0]
    return left_idxs, right_idxs

def _gini_split(left_y, right_y, n_classes, len_y):  
    if len(left_y) == 0 or len(right_y) == 0:
        return 1
    # Now, the code works even if the target variable is float like 1.0
    # Still, the code does NOT work if the Target variable is str like "Yes"
    gini_left = 1.0 - np.sum((np.bincount(left_y.astype('int'), minlength=n_classes) / len(left_y.astype('int'))) ** 2)
    gini_right = 1.0 - np.sum((np.bincount(right_y.astype('int'), minlength=n_classes) / len(right_y.astype('int'))) ** 2)
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
