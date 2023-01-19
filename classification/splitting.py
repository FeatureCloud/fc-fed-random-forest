import random
import numpy as np
import pandas as pd


def gini_split(X, y, n_classes, feat_indices):
    m = len(y)

    best_local_splits = []
    for feat_id in feat_indices:
        temp = []
        sorted_column = sorted(set(X[:, feat_id]))
        threshold_values = [np.mean([a, b]) for a, b in zip(sorted_column, sorted_column[1:])]

        for threshold in threshold_values:

            left_y = y[X[:, feat_id] < threshold]
            right_y = y[X[:, feat_id] > threshold]

            num_class_left = [np.sum(left_y == c) for c in range(n_classes)]
            num_class_right = [np.sum(right_y == c) for c in range(n_classes)]

            gini_left = 1.0 - sum((n / len(left_y)) ** 2 for n in num_class_left)
            gini_right = 1.0 - sum((n / len(right_y)) ** 2 for n in num_class_right)

            gini = (len(left_y) / m) * gini_left + (len(right_y) / m) * gini_right

            temp.append([feat_id, threshold, gini])

        df = pd.DataFrame(temp, columns=['feature_id', 'threshold', 'gini_index']).set_index('feature_id')
        try:
            best = df[df.gini_index == df.gini_index.min()].iloc[0, :].to_list()
        except IndexError:
            best = [None, None]

        best_local_splits.append(best)

    return best_local_splits
