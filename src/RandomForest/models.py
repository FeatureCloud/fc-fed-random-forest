from typing import Optional, Union, Dict, Any

import numpy as np
from scipy import stats
# pylint: disable=too-many-instance-attributes, invalid-name

class RandomForest:
    """
    A random forest model for classification and regression to be used for federated histogram
    based random forest learning and prediction.
    """
    def __init__(self,
                 n_estimators: int,
                 max_samples: float,
                 feat_idcs: np.ndarray, # List of feature indices for each tree,
                                        # n_estimators x n_features
                 max_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 n_patients_local: int, # Number of patients in the local dataset
                 n_patients_global: int, # Number of patients in the global, aggregated dataset
                 bootstrap: bool,
                 random_state: int,
                 quantile: np.ndarray, # List of feature indices using quantile binning
                 global_mean: np.ndarray, # Global mean for each feature
                 global_stddev: np.ndarray, # global standard deviation for each feature
                 split_points: np.ndarray, #  n_features x n_bins - 1,  #TODO: fix them
                 prediction_mode: str, # 'classification' or 'regression'
                 global_classes: np.ndarray, # global classes. Should be the same in the
                                        # same order for all clients
                 oob: bool,
                 class_weights: Optional[Dict[Any, float]] = None,
                    # weights of the classes (dict[class]=weight)
                ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.feat_idcs = feat_idcs
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_patients_local = n_patients_local
        self.n_patients_global = n_patients_global
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.decision_trees = []
        np.random.seed(random_state)
        self.quantile = quantile
        self.global_mean = global_mean
        self.global_stddev = global_stddev
        self.split_points = split_points
        self.prediction_mode = prediction_mode
        self.global_classes = global_classes
        self.class_weights = class_weights
        self.oob = oob
        self.finished = False

        # init the trees
        for _ in range(self.n_estimators):
            sample_idcs = self._bootstrap_samples()
            tree = DecisionTree(samples_idcs=sample_idcs,
                                max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf,
                                feat_idcs=self.feat_idcs,
                                mode=self.prediction_mode,
                                global_classes=self.global_classes,
                                class_weights=self.class_weights)
            self.decision_trees.append(tree)

    def _bootstrap_samples(self):
        sample_size = max(round(self.n_patients_local * self.max_samples), 1)
        sample_idcs = np.random.choice(self.n_patients_local, sample_size, replace=self.bootstrap)
        return sample_idcs

    def predict(self, X):
        bucket_idcs = np.setdiff1d(np.arange(len(X[0])), self.quantile)

        if len(bucket_idcs) > 0:
            # Bucket Binning
            bucket_split_points = self.split_points[bucket_idcs, :]

            X_T_bucket = np.transpose(X[:, bucket_idcs])
            # Assign data points to bins
            X_hist_bucket = np.array([np.digitize(X_T_bucket[i], bucket_split_points[i]) \
                                        for i in range(X_T_bucket.shape[0])]) - 1

        if len(self.quantile) > 0:
            # Quantile Binning
            a = (X[:, self.quantile] - self.global_mean)
            b = self.global_stddev
            normalized = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
            normalized[normalized == np.inf] = 0
            normalized[normalized == -np.inf] = 0
            normalized[normalized == np.nan] = 0

            quantile_split_points = self.split_points[self.quantile, :]
            X_T_quantile = np.transpose(normalized)
            # Assign data points to bins
            X_hist_quantile = np.array([np.digitize(X_T_quantile[i], quantile_split_points[i]) \
                                        for i in range(X_T_quantile.shape[0])])

        if len(bucket_idcs) > 0 and len(self.quantile) > 0:
            X_hist = np.concatenate((X_hist_quantile, X_hist_bucket))
            # Place the values of array at specified indices
            X_hist[self.quantile] = X_hist_quantile
            X_hist[bucket_idcs] = X_hist_bucket
            X_hist = np.transpose(X_hist)

        elif len(bucket_idcs) > 0:
            X_hist = np.transpose(X_hist_bucket)

        else:
            X_hist = np.transpose(X_hist_quantile)

        # Make predictions with every tree in the forest
        y = np.array([tree.predict(X_hist) for tree in self.decision_trees])
        # Reshape so we can find the most common value
        y = np.swapaxes(y, axis1=0, axis2=1)

        if self.prediction_mode == 'classification':
            if not self.oob:
                # Use majority voting for the final prediction
                predicted_values = stats.mode(y, axis=1, keepdims=True)[0].reshape(-1)
            else:
                predicted_values = []
                classes = np.unique(y)
                for i in range(len(X_hist)):
                    counter = []
                    for c in classes:
                        indices = np.where(y[i] == c)[0]
                        counter.append(np.sum([self.decision_trees[j].weight for j in indices]))
                    predicted_values.append(classes[np.argmax(counter)])
        else:
            predicted_values = np.mean(y, axis=0)

        return predicted_values


class DecisionTree:
    """
    A decision tree model for classification and regression to be used for federated histogram
    based random forest learning and prediction.
    """
    def __init__(self,
                 samples_idcs: np.ndarray, # 1d array of indices of the samples to be used
                 max_depth: int,
                 min_samples_split: int,
                 min_samples_leaf: int,
                 feat_idcs: np.ndarray, # List of feature indices for each tree,
                                        # n_estimators x n_features
                 mode: str, # classification or regression
                 global_classes: np.ndarray, # global classes. Should be the same in the
                                        # same order for all clients
                 class_weights: Optional[Dict[Any, float]] = None,
                    # weights of the classes (dict[class]=weight)
                 ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.feat_idcs = feat_idcs
        self.finished = False
        self.leaves = []
        self.weight = 1
        self.mode = mode
        self.global_classes = global_classes
        self.class_weights = class_weights

        # init the root node
        self.root = Node(depth=0,
                         sample_idcs=samples_idcs,
                         mode=mode,
                         global_classes=global_classes,
                         class_weights=class_weights,
                         feature_idcs=feat_idcs)

        self.cur_depth_nodes = [self.root]

    def predict(self,
                X):
        """
        Predict the target values for the input data X (2d array).
        Assumes the same features with the same indices as the training data.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to find the leaf node for the input data x, effectively making a prediction.
        """
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class Node:
    """
    A single decision tree Node.
    """
    def __init__(self,
                 depth: int,
                 sample_idcs: np.ndarray, # 1d array of indices of the samples to be used
                 mode: str, # classification or regression
                 global_classes: np.ndarray, # global classes. Should be the same in the
                                             # same order for all clients
                 feature_idcs: np.ndarray, # List of feature indices for each tree,
                 class_weights: Optional[Dict[Any, float]] = None, # weights of the classes (dict[class]=weight)
                 feature: Optional[int] = None, # the feature index used in this node
                 threshold: Optional[Union[float, int]] = None,
                    # the threshold whether to go left or right. The decision is x <= threshold #TODO: right?
                 score: Optional[float] = None, # the score (e.g. gini impurity) of the split
                 parent: Optional["Node"] = None, # pointer to the parent node
                 left: Optional["Node"] = None, # pointer to the left child
                 right: Optional["Node"] = None, # pointer to the right child
                 global_leaf: bool = False, # whether the node is a leaf globally
                 local_leaf: bool = False, # whether the node is a leaf locally #TODO: how is the local/global leaf determined?
                 value=None):
        self.depth = depth
        self.sample_idcs = sample_idcs
        self.feature = feature
        self.threshold = threshold
        self.score = score
        self.parent = parent
        self.left = left
        self.right = right
        self.global_leaf = global_leaf
        self.local_leaf = local_leaf
        self.value = value
        self.global_classes = global_classes
        self.feature_idcs = feature_idcs
        if len(set(global_classes)) != len(global_classes):
            raise ValueError('Classes must be unique.')
        self.num_classes = len(global_classes)
        if mode not in ['classification', 'regression']:
            raise ValueError('Mode must be either classification or regression.')
        self.mode = mode
        for cl in global_classes:
            if cl not in class_weights:
                raise ValueError('Class weights must be given for all classes.')
        self.class_weights = class_weights

    def is_leaf_node(self):
        """
        Whether the node is GLOBALLY considered a leaf node.
        """
        return self.global_leaf

    def get_split_score(self, X, y):
        """
        Calculate the score of a split for each feature and threshold.
        """
        for feature_idx in self.feature_idcs:
            # TODO: CONTINUE HERE



#TODO: include the following functions in the Node class
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